"""DLB-native autoregressive sampler.

Uses DL-Backtrace for all forward passes and mirrors HuggingFace sampling/beam
semantics. Supports multiple Transformers versions via compatibility fallbacks.
"""

from __future__ import annotations

import gc
import json
import time
import io
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, cast, Any, Dict

import torch
import torch.nn.functional as F

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.beam_search import BeamSearchScorer

# Stopping criteria with fallback for older Transformers versions
try:
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        MaxTimeCriteria,
        MaxNewTokensCriteria,
        EosTokenCriteria,
    )
    HAS_EOS_CRITERIA = True
except ImportError:
    try:
        from transformers.generation.stopping_criteria import (
            StoppingCriteriaList,
            MaxTimeCriteria,
            MaxNewTokensCriteria,
        )
        HAS_EOS_CRITERIA = False
    except ImportError:
        from transformers.generation.stopping_criteria import (
            StoppingCriteriaList,
            MaxTimeCriteria,
            StoppingCriteria,
        )
        HAS_EOS_CRITERIA = False

        class MaxNewTokensCriteria(StoppingCriteria):
            def __init__(self, start_length: int, max_new_tokens: int):
                self.start_length = int(start_length)
                self.max_new_tokens = int(max_new_tokens)
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                cur = input_ids.shape[1]
                return (cur - self.start_length) >= self.max_new_tokens


# Deterministic math for reproducibility across CUDA runs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False


class DLBAutoSampler:
    """
    DLB-native text generation (single-prompt => B=1) supporting:
      • Greedy (temperature/top_k/top_p all None)
      • Sampling: temperature / top-k / top-p (when num_beams == 1)
      • Deterministic beam search via HF BeamSearchScorer (when num_beams > 1)

    All logits come from DLB:
        io = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
        logits = io["output"]["output_values"]  # [batch, T, V]

    Returns:
      Always a single sequence with shape [1, T_total]
      (If return_scores=True on the sampling path, returns (sequence, scores_trace))
    """

    def __init__(self, dlb, tokenizer):
        self.dlb = dlb
        self.tokenizer = tokenizer

    def _clear_dlb_memory(self):
        """Clear DLB intermediate storage BEFORE the next allocation, not after."""
        # Explicitly delete tensor contents before dropping the dict reference,
        # so Python's refcount drops to zero immediately without waiting for GC.
        node_io = getattr(self.dlb, 'node_io', None)
        if node_io:
            for v in node_io.values():
                if isinstance(v, dict):
                    for vv in v.values():
                        if torch.is_tensor(vv):
                            del vv
                elif torch.is_tensor(v):
                    del v
            node_io.clear()
        self.dlb.node_io = {}

        all_wt = getattr(self.dlb, 'all_wt', None)
        if all_wt:
            for v in all_wt.values():
                if isinstance(v, (list, tuple)):
                    for vv in v:
                        if torch.is_tensor(vv):
                            del vv
                elif torch.is_tensor(v):
                    del v
            all_wt.clear()
        self.dlb.all_wt = {}
        # gc.collect()

    def _get_gpu_memory_usage_pct(self) -> float:
        """Return current GPU memory usage as a percentage (0-100). Returns 0 if no CUDA."""
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return (allocated / total) * 100.0

    # -- Native Forward with KV-Cache (Fast Path) --------------------------

    def _check_native_forward_support(self):
        """Check if the underlying model supports native forward with KV-cache.

        Walks the wrapper chain (e.g., ModelWrapper.model → AutoModelForCausalLM)
        looking for a model whose forward() accepts `use_cache` and
        `past_key_values` parameters.
        """
        model = self.dlb.model
        inner = getattr(model, 'model', None)
        if inner is None:
            return False
        try:
            import inspect
            sig = inspect.signature(inner.forward)
            return 'use_cache' in sig.parameters and 'past_key_values' in sig.parameters
        except (ValueError, TypeError):
            return False

    def _ensure_model_on_device(self, target_device):
        """Move the inner model to *target_device* if it isn't there already.

        DLB's execution engine operates on its own extracted-weight copies, so
        the original nn.Module may still be on CPU even when generation runs on
        CUDA.  This one-time move makes the native forward path possible.

        Returns the device the model actually ended up on (may differ from
        *target_device* if OOM prevented the move).
        """
        inner = self.dlb.model.model
        try:
            model_device = next(inner.parameters()).device
        except StopIteration:
            return target_device

        if str(model_device) == str(target_device):
            return model_device

        try:
            inner.to(target_device)
            return target_device
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Not enough VRAM to hold both extracted weights AND model params
            return model_device

    def _native_forward_with_cache(self, input_ids, attention_mask,
                                    past_key_values=None, target_device=None):
        """Fast forward pass using the native model with KV-cache.

        When *past_key_values* is ``None``, runs a full "prefill" pass over the
        entire sequence.  On subsequent calls with valid *past_key_values*,
        only processes the **last token** (incremental decode), reusing cached
        key / value states from all previous positions.

        Parameters
        ----------
        input_ids : Tensor [batch, seq_len]
            Full token sequence (including all previously generated tokens).
        attention_mask : Tensor [batch, seq_len]
            Full attention mask covering every position.
        past_key_values : tuple | None
            Cached KV states from a previous call, or ``None`` for prefill.
        target_device : str | torch.device | None
            Device the generation loop is running on.  Used to ensure the
            model is on the correct device on the first call.

        Returns
        -------
        (logits, new_past_key_values)
            *logits*: ``[batch, 1, vocab]`` for incremental /
            ``[batch, seq_len, vocab]`` for prefill.
            *new_past_key_values*: updated cache for the next call.
        """
        inner_model = self.dlb.model.model

        # ── One-time device alignment ──
        if not hasattr(self, '_native_fwd_device_ok'):
            if target_device is not None:
                self._ensure_model_on_device(target_device)
            self._native_fwd_device_ok = True

        # Detect model device for input alignment
        try:
            _dev = next(inner_model.parameters()).device
        except StopIteration:
            _dev = input_ids.device

        if past_key_values is not None:
            # Incremental decode — only feed the LAST token
            model_input = input_ids[:, -1:].to(_dev)
        else:
            # Prefill — feed the entire sequence
            model_input = input_ids.to(_dev)

        mask_input = attention_mask.to(_dev)

        with torch.no_grad():
            output = inner_model(
                input_ids=model_input,
                attention_mask=mask_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Extract logits (handles CausalLMOutput, plain Tensor, or tuple)
        if hasattr(output, 'logits'):
            logits = output.logits
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            logits = output[0]

        new_past_kv = getattr(output, 'past_key_values', None)
        return logits, new_past_kv

    def _save_to_disk(
        self,
        data,
        *,
        cache_dir: Path,
        filename: str,
        use_compression: bool = True,
        compression_method: str = "lz4",
        pickle_protocol: int = 4,
    ) -> str:
        """Save tensor/dict data to disk with optional lz4 compression. Returns the file path."""
        base_path = cache_dir / filename

        def _to_cpu_async(obj):
            if torch.is_tensor(obj):
                t = obj.detach()
                return t.to('cpu', non_blocking=True) if t.is_cuda else t
            if isinstance(obj, np.ndarray):
                return torch.from_numpy(obj)
            if isinstance(obj, dict):
                return {k: _to_cpu_async(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_cpu_async(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_cpu_async(v) for v in obj)
            return obj

        cpu_data = _to_cpu_async(data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if use_compression and compression_method == "lz4":
            if not HAS_LZ4:
                raise ImportError("lz4 library required. Install with: pip install lz4")
            file_path = Path(str(base_path) + '.lz4')
            buf = io.BytesIO()
            torch.save(cpu_data, buf, pickle_protocol=pickle_protocol)
            compressed = lz4.frame.compress(buf.getvalue())
            with open(file_path, 'wb') as f:
                f.write(compressed)
            del buf, compressed
        else:
            file_path = base_path
            torch.save(cpu_data, file_path, pickle_protocol=pickle_protocol)

        del cpu_data
        gc.collect()
        return str(file_path)

    def _print_generated_sequence(self, generated: torch.Tensor, prefix: str = ""):
        """Print decoded sequence using self.tokenizer."""
        text = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        if prefix:
            print(f"{prefix}: {text}")
        else:
            print(text)

    # ---------- small dtype helpers ----------

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.LongTensor:
        return cast(torch.LongTensor, x.long())

    @staticmethod
    def _as_float(x: torch.Tensor) -> torch.FloatTensor:
        return cast(torch.FloatTensor, x.float())

    @staticmethod
    def _criteria_true(x) -> bool:
        """Robustly coerce stopping-criteria output (bool or tensor) to a Python bool."""
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return False
            if x.numel() == 1:
                return bool(x.item())
            return bool(x.any().item())
        return bool(x)

    # ---------- model / config helpers ----------

    def _get_causallm(self, model_like):
        """Walk `.model` chain until we find a GenerationMixin-style CausalLM."""
        obj = model_like
        seen = set()
        for _ in range(8):
            if hasattr(obj, "_prepare_generation_config") and hasattr(obj, "_get_logits_processor"):
                return obj
            i = id(obj)
            if i in seen:
                break
            seen.add(i)
            if hasattr(obj, "model"):
                obj = obj.model
            else:
                break
        raise TypeError(
            f"{type(model_like).__name__} isn't a GenerationMixin model. "
            "Pass AutoModelForCausalLM / LlamaForCausalLM (not base LlamaModel)."
        )

    @staticmethod
    def _attach_token_tensors(gen_config, device):
        """Populate private token tensors expected by HF internals (4.52.x)."""
        def to_tensor(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return torch.tensor(list(x), device=device, dtype=torch.long)
            return torch.tensor([int(x)], device=device, dtype=torch.long)
        gen_config._eos_token_tensor = to_tensor(getattr(gen_config, "eos_token_id", None))
        gen_config._bos_token_tensor = to_tensor(getattr(gen_config, "bos_token_id", None))
        gen_config._pad_token_tensor = to_tensor(getattr(gen_config, "pad_token_id", None))

    @staticmethod
    def _extract_last_logits(io_data: dict) -> torch.Tensor:
        """
        Find logits tensor in DLB io_data.
        Expect: io_data["output"]["output_values"] with shape [N, T, V].
        """
        if isinstance(io_data, dict):
            out = io_data.get("output", None)
            if isinstance(out, dict) and isinstance(out.get("output_values", None), torch.Tensor):
                return out["output_values"]
            for k in reversed(list(io_data.keys())):
                v = io_data[k]
                if isinstance(v, dict) and isinstance(v.get("output_values", None), torch.Tensor):
                    return v["output_values"]
        raise KeyError("DLB io_data does not contain 'output' -> 'output_values' tensor.")

    @staticmethod
    def _clean_sampling_knobs(
        temp: Optional[float], top_k: Optional[int], top_p: Optional[float]
    ) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        T = float(temp) if (temp is not None and temp != 1.0) else None
        K = int(top_k) if (top_k is not None and top_k > 0) else None
        P = float(top_p) if (top_p is not None and 0.0 < top_p < 1.0) else None
        return T, K, P

    @staticmethod
    def _decide_do_sample(T: Optional[float], K: Optional[int], P: Optional[float]) -> bool:
        # HF rule: any knob triggers sampling
        return (T is not None) or (K is not None) or (P is not None)

    @staticmethod
    def _build_warper(
        temperature: Optional[float], top_k: Optional[int], top_p: Optional[float]
    ) -> LogitsProcessorList:
        # Match HF non-beam order: Temperature -> TopK -> TopP
        w = LogitsProcessorList()
        keep = 1
        if temperature is not None:
            if temperature <= 0:
                raise ValueError("temperature must be > 0 when do_sample=True")
            w.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k > 0:
            w.append(TopKLogitsWarper(top_k, min_tokens_to_keep=keep))
        if top_p is not None and top_p < 1.0:
            w.append(TopPLogitsWarper(top_p, min_tokens_to_keep=keep))
        return w

    @staticmethod
    def _build_stopping(
        start_len: int,
        max_new_tokens: Optional[int],
        max_time: Optional[float],
        eos_token_id: Optional[int | List[int]] = None,
    ) -> StoppingCriteriaList:
        sc = StoppingCriteriaList()
        if max_new_tokens is not None:
            try:
                sc.append(MaxNewTokensCriteria(max_new_tokens=int(max_new_tokens)))
            except TypeError:
                sc.append(MaxNewTokensCriteria(start_length=int(start_len), max_new_tokens=int(max_new_tokens)))
        if max_time is not None:
            sc.append(MaxTimeCriteria(max_time=float(max_time)))
        if eos_token_id is not None and HAS_EOS_CRITERIA:
            sc.append(EosTokenCriteria(eos_token_id=eos_token_id))
        return sc

    def _compute_relevance(
        self,
        target_token_ids,
        *,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        task="generation",
        debug=False,
    ):
        """
        Compute relevance for the *actual* chosen token(s), not greedy argmax.

        target_token_ids:
            torch.Tensor shape [1] or [beam] or list[int]
            We turn this into a Python list[int] and pass it through.
        """
        # normalize to list[int] or None
        if target_token_ids is None:
            tti = None
        elif torch.is_tensor(target_token_ids):
            tti = [int(x) for x in target_token_ids.view(-1).tolist()]
        elif isinstance(target_token_ids, (list, tuple)):
            tti = [int(x) for x in target_token_ids]
        else:
            tti = [int(target_token_ids)]

        rel_dict = self.dlb.evaluation(
            mode=mode,
            start_wt=[],
            multiplier=multiplier,
            scaler=scaler,
            thresholding=thresholding,
            task=task,                   # <- "generation" during decoding
            target_token_ids=tti,        # <- this is now plumbed all the way down
            debug=debug,
        )

        return rel_dict

    def _summarize_relevance(self, rel_dict):
        """
        Turn self.dlb.all_wt (rel_dict) into a lightweight scalar so we don't
        explode memory every token.

        Currently: sum of all entries.
        You can change this to anything you want.
        """
        total = 0.0

        def add_val(x):
            nonlocal total
            if torch.is_tensor(x):
                total += float(x.detach().cpu().sum().item())
            elif hasattr(x, "sum") and hasattr(x, "shape"):
                # numpy-like
                total += float(x.sum())
            elif isinstance(x, (list, tuple)):
                for v in x:
                    add_val(v)
            elif isinstance(x, dict):
                for v in x.values():
                    add_val(v)

        add_val(rel_dict)
        return total

    @staticmethod
    def _resolve_torch_dtype(dtype_hint):
        if dtype_hint is None:
            return None
        if isinstance(dtype_hint, torch.dtype):
            return dtype_hint
        if isinstance(dtype_hint, str):
            key = dtype_hint.strip().lower()
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float64": torch.float64,
                "fp64": torch.float64,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported relevance dtype hint: {dtype_hint}")


    def _prepare_cache_dir(self, base_dir: Optional[str], policy: str):
        if policy != "disk":
            return None
        if not base_dir:
            raise ValueError("relevance_cache_dir is required when relevance_cache_policy='disk'")
        root = Path(base_dir).expanduser()
        timestamp = int(time.time() * 1000)
        run_dir = root / f"relevance_cache_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _store_relevance_entry(
        self,
        rel_dict,
        *,
        policy: str,
        step_idx: int,
        cache_dir: Optional[Path],
        target_dtype,
        move_to_cpu: bool,
        use_compression: bool = True,
        compression_method: str = "lz4",
        pickle_protocol: int = 4,
    ):
        """Store relevance: filter → GPU concat → ONE DMA → free GPU → raw-bytes save.

        File format (.dlbr):
          [8-byte header_len LE] [JSON metadata] [LZ4-compressed raw float32 bytes]

        Use DLBAutoSampler.load_relevance_step() to reload.
        """
        normalized_policy = (policy or "full").lower()
        if normalized_policy == "none":
            return None

        _t0 = time.perf_counter()

        # Filter weight/buffer nodes by name prefix
        original_count = 0
        original_bytes = 0
        kept_keys = []
        if isinstance(rel_dict, dict):
            original_count = len(rel_dict)
            for k, v in rel_dict.items():
                if torch.is_tensor(v):
                    original_bytes += v.nelement() * v.element_size()
            kept_keys = [k for k in rel_dict
                         if not (k.startswith("p_model_") or k.startswith("b_model_"))]

        # Flatten all kept tensors into a single GPU tensor, then do one DMA transfer
        meta_entries = []
        flat_parts = []
        offset = 0
        cast_dtype = self._resolve_torch_dtype(target_dtype) if target_dtype else torch.float32

        for k in kept_keys:
            v = rel_dict[k]
            if torch.is_tensor(v):
                t = v.detach()
                orig_dtype = str(t.dtype)
                flat = t.to(dtype=cast_dtype).reshape(-1)
                n = flat.numel()
                meta_entries.append((k, list(t.shape), orig_dtype, n))
                flat_parts.append(flat)
                offset += n
            elif isinstance(v, (list, tuple)):
                for i, sub in enumerate(v):
                    if torch.is_tensor(sub):
                        t = sub.detach()
                        orig_dtype = str(t.dtype)
                        flat = t.to(dtype=cast_dtype).reshape(-1)
                        n = flat.numel()
                        meta_entries.append((f"{k}[{i}]", list(t.shape), orig_dtype, n))
                        flat_parts.append(flat)
                        offset += n

        if not flat_parts:
            rel_dict.clear()
            return None

        # Cat directly on CPU — avoids allocating a large contiguous GPU tensor
        # that frequently triggers OOM at long sequence lengths.
        flat_cpu = torch.cat([t.cpu() for t in flat_parts])
        del flat_parts

        # Free GPU memory before serialisation
        rel_dict.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if normalized_policy == "summary":
            cpu_dict = self._flat_to_dict(flat_cpu, meta_entries)
            return {"summary": self._summarize_relevance(cpu_dict)}

        if normalized_policy == "full":
            return self._flat_to_dict(flat_cpu, meta_entries)

        if normalized_policy != "disk":
            raise ValueError(
                "relevance_cache_policy must be one of "
                "{'full', 'summary', 'disk', 'none'}"
            )

        if cache_dir is None:
            raise ValueError(
                "relevance_cache_dir must be provided when "
                "relevance_cache_policy='disk'"
            )

        cpu_dict = self._flat_to_dict(flat_cpu, meta_entries)
        summary_val = self._summarize_relevance(cpu_dict)
        del cpu_dict

        header = {
            "format": "dlbr_v1",
            "dtype": str(cast_dtype),
            "total_elements": int(flat_cpu.numel()),
            "entries": [
                {"key": k, "shape": s, "orig_dtype": d, "count": n}
                for k, s, d, n in meta_entries
            ],
        }
        header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
        header_len = len(header_bytes)

        raw_bytes = flat_cpu.numpy().tobytes()
        del flat_cpu

        file_path = cache_dir / f"step_{step_idx:05d}.dlbr"
        if use_compression and compression_method == "lz4" and HAS_LZ4:
            compressed = lz4.frame.compress(raw_bytes)
            with open(file_path, 'wb') as f:
                f.write(header_len.to_bytes(8, 'little'))
                f.write(header_bytes)
                f.write(compressed)
            del compressed
        else:
            with open(file_path, 'wb') as f:
                f.write(header_len.to_bytes(8, 'little'))
                f.write(header_bytes)
                f.write(raw_bytes)
        del raw_bytes

        gc.collect()
        return {
            "summary": summary_val,
            "path": str(file_path),
            "compression": compression_method if use_compression else "none",
        }

    @staticmethod
    def _flat_to_dict(flat_cpu, meta_entries):
        """Reconstruct a {key: tensor} dict from a flat buffer + metadata."""
        result = {}
        offset = 0
        for key, shape, orig_dtype, count in meta_entries:
            t = flat_cpu[offset:offset + count].reshape(shape)
            result[key] = t
            offset += count
        return result

    @staticmethod
    def load_relevance_step(file_path: str, device: str = "cpu"):
        """Load a .dlbr relevance file. Returns a dict mapping node keys to tensors."""
        path = Path(file_path)
        with open(path, 'rb') as f:
            header_len = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(header_len).decode('utf-8'))
            payload = f.read()

        dtype_str = header.get("dtype", "torch.float32")
        dtype_map = {
            "torch.float32": (torch.float32, np.float32),
            "torch.float16": (torch.float16, np.float16),
            "torch.bfloat16": (torch.float32, np.float32),  # no numpy equiv for bfloat16
        }
        torch_dtype, np_dtype = dtype_map.get(dtype_str, (torch.float32, np.float32))

        if HAS_LZ4:
            try:
                payload = lz4.frame.decompress(payload)
            except Exception:
                pass  # uncompressed fallback

        flat = torch.frombuffer(bytearray(payload), dtype=torch_dtype)
        result = {}
        offset = 0
        for entry in header["entries"]:
            key = entry["key"]
            shape = entry["shape"]
            count = entry["count"]
            t = flat[offset:offset + count].reshape(shape).clone()
            if device != "cpu":
                t = t.to(device)
            result[key] = t
            offset += count
        return result

    # ---------- public API ----------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        # sampling knobs
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        # lengths / stopping
        max_new_tokens: int = 50,
        min_new_tokens: Optional[int] = None,
        max_time: Optional[float] = None,
        early_stopping: Optional[bool | str] = None,   # used only for beams
        # constraints
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        bad_words_ids: Optional[List[List[int]]] = None,
        # ids
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int | List[int]] = None,
        pad_token_id: Optional[int] = None,
        # beams
        num_beams: int = 1,
        num_return_sequences: Optional[int] = None,
        length_penalty: float = 1.0,
        # misc
        return_scores: bool = False,
        return_layerwise_output: bool = False,
        return_relevance: bool = False,
        explain_tokens = "all",
        debug: bool = False,
        relevance_cache_policy: str = "full",
        relevance_cache_dir: Optional[str] = None,
        relevance_compress_dtype: Optional[Any] = "float16",
        relevance_move_to_cpu: bool = True,
        relevance_use_compression: bool = True,
        relevance_compression_method: str = "lz4",
        relevance_pickle_protocol: int = 4,
    ):
        """
        Always returns:
            - [1, T_total] (top-1 sequence)
            - Or (sequence, scores_trace) for sampling when return_scores=True

        Args:
            explain_tokens: Which generated tokens to compute DLB relevance for.
                - "all" (default): Compute relevance for every generated token
                - "none": Skip relevance for all tokens
                - int N: Compute relevance for the first N tokens only
                - list[int]: Compute relevance for these specific step indices only
                  (e.g. [0, 4, 7] → only tokens at step 0, 4, and 7)

            relevance_cache_policy: "full" (default), "summary", "disk", or "none".
            relevance_cache_dir: base directory for on-disk caching (policy="disk").
            relevance_compress_dtype: dtype hint (str or torch.dtype) for stored tensors.
            relevance_use_compression: If True, use lz4 compression for disk storage (default: True).
            relevance_compression_method: "lz4" (default) or "none".
            relevance_pickle_protocol: Pickle protocol (2-5). Default=4.
            relevance_move_to_cpu: move tensors to CPU before caching to reduce VRAM.
        """ 
        model = self._get_causallm(self.dlb.model)
        device = input_ids.device
        B = input_ids.size(0)
        assert B == 1, "Current implementation assumes batch size = 1."

        cache_policy = (relevance_cache_policy or "full").lower()
        allowed_policies = {"full", "summary", "disk", "none"}
        if cache_policy not in allowed_policies:
            raise ValueError(
                "relevance_cache_policy must be one of {'full', 'summary', 'disk', 'none'}"
            )
        cache_dtype = (
            self._resolve_torch_dtype(relevance_compress_dtype)
            if relevance_compress_dtype is not None
            else None
        )
        cache_dir_path = None
        if return_relevance and cache_policy == "disk":
            cache_dir_path = self._prepare_cache_dir(relevance_cache_dir, cache_policy)

        # Dtypes up-front
        input_ids = self._as_long(input_ids)

        # Build / infer attention mask
        if attention_mask is None:
            if pad_token_id is not None:
                attention_mask = self._as_long((input_ids != pad_token_id).long())
            else:
                attention_mask = self._as_long(torch.ones_like(input_ids))
        else:
            attention_mask = self._as_long(attention_mask)

        # Normalize knobs
        T, K, P = self._clean_sampling_knobs(temperature, top_k, top_p)
        do_sample = self._decide_do_sample(T, K, P)

        start_len = input_ids.shape[1]

        # If BEAM mode, **hard-disable** sampling knobs to avoid warnings
        if num_beams > 1:
            do_sample = False
            T = K = P = None

        # Early stopping for beams -> use bool for cross-version HF
        if num_beams > 1:
            if isinstance(early_stopping, bool):
                do_early_stopping = early_stopping
            elif early_stopping == "always":
                do_early_stopping = True
            else:
                do_early_stopping = False
        else:
            do_early_stopping = False

        # Prepare GenerationConfig for processors
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
        )
        generation_config, _ = model._prepare_generation_config(
            generation_config=None, use_model_defaults=True, **gen_kwargs
        )
        self._attach_token_tensors(generation_config, device)

        # EOS list / PAD
        if generation_config.eos_token_id is None:
            eos_list: List[int] = []
        elif isinstance(generation_config.eos_token_id, (list, tuple)):
            eos_list = list(generation_config.eos_token_id)
        else:
            eos_list = [int(generation_config.eos_token_id)]
        eos_set = set(eos_list)
        PAD = pad_token_id if pad_token_id is not None else (eos_list[0] if eos_list else 0)

        stopping_criteria = self._build_stopping(
            start_len,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            eos_token_id=eos_list if eos_list else None,
        )

        # Processors (HF builds them; we apply per step)
        input_ids_seq_length = input_ids.shape[1]
        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device=device,
            model_kwargs={},
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )

        # Warpers (sampling path only)
        if do_sample and num_beams == 1 and any(v is not None for v in (T, K, P)):
            logits_warper = self._build_warper(T, K, P)
        else:
            logits_warper = None

        # ======================
        # Non-beam path (B=1)
        # ======================
        if num_beams < 2:
            generated = self._as_long(input_ids.clone()).to(device)
            attn = self._as_long(attention_mask.clone()).to(device)
            scores_trace = [] if return_scores else None
            relevance_trace = [] if return_relevance else None
            io_data_trace = [] if return_layerwise_output else None
            disk_streaming = (cache_policy == "disk" and cache_dir_path is not None)

            # ── Per-step timing accumulator ──
            _step_timings = []

            stopped_by = None

            # Build DLB gating function from explain_tokens
            if isinstance(explain_tokens, str):
                if explain_tokens.lower() == "none":
                    _should_run_dlb = lambda idx: False
                else:  # "all"
                    _should_run_dlb = lambda idx: True
            elif isinstance(explain_tokens, int):
                _should_run_dlb = lambda idx: idx < explain_tokens
            elif isinstance(explain_tokens, (list, tuple)):
                _dlb_steps = set(explain_tokens)
                _should_run_dlb = lambda idx: idx in _dlb_steps
            else:
                _should_run_dlb = lambda idx: True

            # ── Fast-path: native model with KV-cache ──
            _native_fwd_ok = self._check_native_forward_support()
            _past_kv = None          # KV-cache for native forward
            if _native_fwd_ok:
                print("  Native KV-cache forward available — fast path enabled for non-DLB steps")

            for _gen_step_idx in range(max_new_tokens if max_new_tokens is not None else 10_00_000):
                _t = {}  # timing dict for this step
                _t["step"] = _gen_step_idx
                _t["seq_len"] = generated.shape[1]

                # ── Stage A: Forward pass ──
                if device == "cuda":
                    torch.cuda.synchronize()
                _ts = time.perf_counter()

                _use_dlb = _should_run_dlb(_gen_step_idx) or not _native_fwd_ok

                if _use_dlb:
                    # ── SLOW PATH: clear BEFORE allocating ──
                    # Kill old node_io/all_wt first so peak RAM
                    self._clear_dlb_memory()
                    if _past_kv is not None:
                        del _past_kv
                        _past_kv = None
                    io_data = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
                    logits = self._extract_last_logits(io_data)
                    # Drop io_data immediately unless layerwise output is needed —
                    # self.dlb.node_io and io_data are the same dict, so keeping
                    # io_data alive prevents the refcount from hitting zero
                    if not return_layerwise_output:
                        del io_data
                        io_data = None
                else:
                    # ── FAST PATH: Native model with KV-cache
                    
                    logits, _past_kv = self._native_forward_with_cache(
                        generated, attn, past_key_values=_past_kv,
                        target_device=device,
                    )
                    io_data = None

                # ── OOM guard: stop DLB loop if GPU usage > 85% ──
                _gpu_pct = self._get_gpu_memory_usage_pct()
                if _use_dlb and _gpu_pct > 85.0:
                    print(
                        f" GPU memory at {_gpu_pct:.1f}% after token {_gen_step_idx} "
                        f"— stopping generation to prevent OOM."
                    )
                    stopped_by = "gpu_oom_guard"
                    break

                if device == "cuda":
                    torch.cuda.synchronize()
                _t["predict"] = time.perf_counter() - _ts

                # ── Stage B: Logits extraction + sampling ──
                _ts = time.perf_counter()
                if logits.device != device:
                    logits = logits.to(device)
                next_logits = self._as_float(logits[:, -1, :])     # Float for processors

                # processors
                scores = logits_processor(self._as_long(generated), next_logits)
                if scores.device != device:
                    scores = scores.to(device)

                # enforce min_new_tokens by masking EOS until allowed
                if min_new_tokens is not None and (generated.shape[1] - start_len) < min_new_tokens and eos_list:
                    scores[:, eos_list] = -1e9

                # sampling vs greedy
                if do_sample and logits_warper is not None:
                    scores = logits_warper(self._as_long(generated), scores)
                    probs = F.softmax(scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)  # LongTensor
                else:
                    next_tokens = scores.argmax(dim=-1, keepdim=True)      # LongTensor

                # 🔑 force to same device as `generated`
                next_tokens = self._as_long(next_tokens).to(device)
                if device == "cuda":
                    torch.cuda.synchronize()
                _t["sampling"] = time.perf_counter() - _ts

                if not return_layerwise_output:
                    del logits
                    if io_data is not None:
                        del io_data

                # ── Stage C: Save scores to disk ──
                _ts = time.perf_counter()
                if return_scores:
                    scores_cpu = scores.detach().to("cpu")
                    if disk_streaming:
                        path = self._save_to_disk(
                            scores_cpu,
                            cache_dir=cache_dir_path,
                            filename=f"step_{_gen_step_idx:05d}_scores.pt",
                            use_compression=relevance_use_compression,
                            compression_method=relevance_compression_method,
                            pickle_protocol=relevance_pickle_protocol,
                        )
                        scores_trace.append({"path": path})
                    else:
                        scores_trace.append(scores_cpu)
                    del scores_cpu
                _t["scores_save"] = time.perf_counter() - _ts

                # ── Stage D: Save IO data to disk ──
                _ts = time.perf_counter()
                if return_layerwise_output and io_data is not None:
                    if disk_streaming:
                        path = self._save_to_disk(
                            io_data,
                            cache_dir=cache_dir_path,
                            filename=f"step_{_gen_step_idx:05d}_io.pt",
                            use_compression=relevance_use_compression,
                            compression_method=relevance_compression_method,
                            pickle_protocol=relevance_pickle_protocol,
                        )
                        io_data_trace.append({"path": path})
                        del io_data
                        gc.collect()
                    else:
                        io_data_trace.append(io_data)
                _t["io_save"] = time.perf_counter() - _ts
                
                rel_dict = None

                # ── Stage E: Backtracing (relevance propagation) ──
                _ts = time.perf_counter()
                if _should_run_dlb(_gen_step_idx):
                    rel_dict = self._compute_relevance(
                        target_token_ids=next_tokens.view(-1),
                        mode="default",
                        multiplier=100.0,
                        scaler=1.0,
                        thresholding=0.5,
                        task="generation",
                        debug=False,
                    )  
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if device == "cuda":
                    torch.cuda.synchronize()
                _t["backtrace"] = time.perf_counter() - _ts

                # ── Stage F: Save relevance ──
                _ts = time.perf_counter()
                if return_relevance and _should_run_dlb(_gen_step_idx):
                    if cache_policy == "disk":
                        entry = self._store_relevance_entry(
                            rel_dict,
                            policy=cache_policy,
                            step_idx=_gen_step_idx,
                            cache_dir=cache_dir_path,
                            target_dtype=cache_dtype,
                            move_to_cpu=relevance_move_to_cpu,
                            use_compression=relevance_use_compression,
                            compression_method=relevance_compression_method,
                            pickle_protocol=relevance_pickle_protocol,
                        )
                        relevance_trace.append(entry)
                        # _store_relevance_entry already calls rel_dict.clear()
                        # for disk policy, so nothing left to free
                    elif cache_policy == "full":
                        # Deep-copy tensors to CPU so we can safely free GPU memory
                        full_rel = self._store_relevance_entry(
                            rel_dict,
                            policy=cache_policy,
                            step_idx=_gen_step_idx,
                            cache_dir=cache_dir_path,
                            target_dtype=cache_dtype,
                            move_to_cpu=relevance_move_to_cpu,
                            use_compression=relevance_use_compression,
                            compression_method=relevance_compression_method,
                            pickle_protocol=relevance_pickle_protocol,
                        )
                        relevance_trace.clear()
                        relevance_trace.append(full_rel)
                        del full_rel
                _t["relevance_save"] = time.perf_counter() - _ts

                # ── Stage G: Memory cleanup ──
                _ts = time.perf_counter()
                if rel_dict is not None:
                    # Safe to clear — "full" policy already copied tensors above,
                    # "disk" policy already consumed the data in _store_relevance_entry
                    if isinstance(rel_dict, dict):
                        rel_dict.clear()
                    del rel_dict
                    rel_dict = None

                _t["cleanup"] = time.perf_counter() - _ts

                _t["total"] = _t["predict"] + _t["sampling"] + _t["scores_save"] + _t["io_save"] + _t["backtrace"] + _t["relevance_save"] + _t["cleanup"]

                # Print per-token line
                _path_label = "DLB " if _use_dlb else "FAST"
                print(
                    f"  ⏱ Token {_gen_step_idx:3d} [{_path_label}] (seq={_t['seq_len']:4d}) │ "
                    f"predict={_t['predict']:6.2f}s │ "
                    f"sample={_t['sampling']:5.3f}s │ "
                    f"backtrace={_t['backtrace']:5.2f}s │ "
                    f"rel_save={_t['relevance_save']:5.2f}s │ "
                    f"scores_save={_t['scores_save']:5.3f}s │ "
                    f"io_save={_t['io_save']:5.2f}s │ "
                    f"cleanup={_t['cleanup']:5.2f}s │ "
                    f"TOTAL={_t['total']:6.2f}s"
                )
                _step_timings.append(_t)

                generated = torch.cat([generated, next_tokens], dim=1) 
                attn = torch.cat(
                    [attn, torch.ones((1, 1), dtype=attn.dtype, device=attn.device)],
                    dim=1,
                )

                # early stop if EOS produced
                if eos_list:
                    tok = int(next_tokens.view(-1)[0].item())
                    if tok in eos_set:
                        stopped_by = "eos"
                        break

                # HF-native stopping criteria (max_new_tokens / max_time)
                crit = stopping_criteria(self._as_long(generated[:1, :]), None)  # slice to [1, T]
                if self._criteria_true(crit):
                    stopped_by = "stopping_criteria"
                    break
            else:
                stopped_by = "loop_exhausted"

            # ── Print timing summary ──
            if _step_timings:
                stages = ["predict", "sampling", "backtrace", "relevance_save", "scores_save", "io_save", "cleanup", "total"]
                print("\n" + "=" * 90)
                print("  ⏱ GENERATION TIMING SUMMARY")
                print("=" * 90)
                for stage in stages:
                    vals = [t[stage] for t in _step_timings]
                    total_s = sum(vals)
                    avg_s = total_s / len(vals) if vals else 0
                    pct = (total_s / sum(t["total"] for t in _step_timings) * 100) if stage != "total" else 100.0
                    print(f"  {stage:>16s}:  total={total_s:7.2f}s  avg={avg_s:6.2f}s  ({pct:5.1f}%)")
                print("=" * 90)

            if debug:
                print(f"[greedy/sampling] stopped_by={stopped_by}")

            want_extras = return_scores or return_relevance or return_layerwise_output
            if not want_extras:
                return generated

            info = {}
            if return_scores:
                info["scores_trace"] = scores_trace
            if return_relevance:
                info["relevance_trace"] = relevance_trace
                info["relevance_cache_policy"] = cache_policy
                if cache_dir_path is not None:
                    info["relevance_cache_dir"] = str(cache_dir_path)
            if return_layerwise_output:
                info["layerwise_output_trace"] = io_data_trace
            if disk_streaming and cache_dir_path is not None:
                info["cache_dir"] = str(cache_dir_path)
            return generated, info  # ([1, T], dict)


        # ======================
        # Beam search path (deterministic, no sampling) — always return top-1
        # ======================
        beams = int(num_beams)
        assert beams >= 2, "num_beams must be >= 2 for beam search."

        # Some HF versions require max_length when early stopping is configured.
        max_len_for_beam = int(start_len + (max_new_tokens if max_new_tokens is not None else 1024))

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=do_early_stopping,
            num_beam_hyps_to_keep=1,
            max_length=max_len_for_beam,
        )

        generated = self._as_long(input_ids.expand(beams, -1).contiguous()).to(device)
        attn = self._as_long(attention_mask.expand(beams, -1).contiguous()).to(device)

        # Local beam scores
        beam_scores = torch.zeros((1, beams), dtype=torch.float32, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1).to(device)   # [beams] on device

        cur_len = start_len
        vocab_size = None
        start_time = time.time() if max_time is not None else None

        # Keep top-k tensors from the last step (needed by some HF versions)
        last_topk_tokens: Optional[torch.Tensor] = None
        last_topk_indices: Optional[torch.Tensor] = None

        scores_trace_beam = [] if return_scores else None
        relevance_trace_beam = [] if return_relevance else None
        io_data_trace_beam = [] if return_layerwise_output else None
        disk_streaming_beam = (cache_policy == "disk" and cache_dir_path is not None)

        stopped_by = None
        for _ in range(max_new_tokens if max_new_tokens is not None else 10_000_000):
            # time guard
            if start_time is not None and (time.time() - start_time) >= max_time:
                if debug:
                    print("[beam] max_time reached, stopping.")
                stopped_by = "max_time"
                break

            # ---- Query DLB per beam ----
            next_logits_list: List[torch.Tensor] = []
            io_data_step: List[dict] = [] if return_layerwise_output else None 

            for b in range(beams):
                io_b = self.dlb.predict(
                    generated[b:b+1], 
                    attn[b:b+1], 
                    debug=False, 
                    temperature=1.0
                )
                if return_layerwise_output:
                    io_data_step.append(io_b)

                logits_b = self._extract_last_logits(io_b)          # [1, T_cur, V]
                if logits_b.device != device:                        # <— normalize device
                    logits_b = logits_b.to(device)
                nl_b = self._as_float(logits_b[:, -1, :])           # [1, V] float on device
                next_logits_list.append(nl_b)
                if vocab_size is None:
                    vocab_size = nl_b.size(-1)

            beam_step_idx = cur_len - start_len
            if return_layerwise_output:
                if disk_streaming_beam:
                    path = self._save_to_disk(
                        io_data_step,
                        cache_dir=cache_dir_path,
                        filename=f"beam_step_{beam_step_idx:05d}_io.pt",
                        use_compression=relevance_use_compression,
                        compression_method=relevance_compression_method,
                        pickle_protocol=relevance_pickle_protocol,
                    )
                    io_data_trace_beam.append({"path": path})
                    del io_data_step
                    gc.collect()
                else:
                    io_data_trace_beam.append(io_data_step)              # per-step, per-beam

            next_logits = torch.cat(next_logits_list, dim=0)         # [beams, V] on device

            # processors (no warpers in deterministic beam)
            scores = logits_processor(self._as_long(generated), next_logits)
            if scores.device != device:
                scores = scores.to(device)

            # mask EOS until min_new_tokens is satisfied
            if min_new_tokens is not None and (cur_len - start_len) < min_new_tokens and eos_list:
                scores[:, eos_list] = -1e9

            # convert to log-prob and add previous beam scores
            next_token_scores = F.log_softmax(scores, dim=-1)        # [beams, V]
            next_token_scores = next_token_scores + beam_scores[:, None]  # stays on device

            # Select top 2*beams across all (beam, vocab) pairs
            flat = next_token_scores.view(1, beams * vocab_size)     # [1, beams*V]
            topk = min(2 * beams, beams * vocab_size)
            next_scores, next_tokens = torch.topk(flat, k=topk, dim=1)
            # all on device already, but be explicit for safety:
            next_scores = next_scores.to(device)
            next_tokens = next_tokens.to(device)

            next_indices = (next_tokens // vocab_size).to(device)
            next_tokens  = (next_tokens %  vocab_size).to(device)

            # keep the *last* top-k tensors for finalize() (shape [1, topk])
            last_topk_tokens = self._as_long(next_tokens).to(device)
            last_topk_indices = self._as_long(next_indices).to(device)

            # Let HF scorer decide which beams continue / finish
            beam_outputs = beam_scorer.process(
                input_ids=self._as_long(generated),
                next_scores=self._as_float(next_scores).to(device),
                next_tokens=self._as_long(next_tokens).to(device),
                next_indices=self._as_long(next_indices).to(device),
                pad_token_id=PAD,
                eos_token_id=eos_list if eos_list else None,
                beam_indices=None,
            )

            if return_scores:
                scores_cpu = next_scores.detach().to("cpu")
                if disk_streaming_beam:
                    path = self._save_to_disk(
                        scores_cpu,
                        cache_dir=cache_dir_path,
                        filename=f"beam_step_{beam_step_idx:05d}_scores.pt",
                        use_compression=relevance_use_compression,
                        compression_method=relevance_compression_method,
                        pickle_protocol=relevance_pickle_protocol,
                    )
                    scores_trace_beam.append({"path": path})
                    del scores_cpu
                else:
                    scores_trace_beam.append(scores_cpu)

            # Rebuild the new beam batch using public keys
            next_beam_scores = beam_outputs["next_beam_scores"].to(device)   # [beams]
            next_beam_tokens = beam_outputs["next_beam_tokens"].to(device)   # [beams]
            next_beam_indices = beam_outputs["next_beam_indices"].to(device) # [beams]

            # Save old beam state for relevance computation (before adding new tokens)
            if return_relevance:
                old_generated = generated[self._as_long(next_beam_indices), :].clone()
                old_attn = attn[self._as_long(next_beam_indices), :].clone()

            generated = torch.cat(
                [generated[self._as_long(next_beam_indices), :],
                self._as_long(next_beam_tokens).unsqueeze(-1).to(device)],
                dim=1,
            )
            attn = torch.cat(
                [attn[self._as_long(next_beam_indices), :],
                torch.ones((beams, 1), dtype=attn.dtype, device=attn.device)],
                dim=1,
            )
            beam_scores = self._as_float(next_beam_scores).to(device)
            cur_len += 1

            if return_relevance:
                step_offset = len(relevance_trace_beam)
                if _should_run_dlb(step_offset):
                    step_rel_scores = []
                    for b in range(beams):
                        self.dlb.predict(
                            old_generated[b:b+1],
                            old_attn[b:b+1],
                            debug=False,
                            temperature=1.0,
                        )

                        chosen_tok_b = next_beam_tokens[b:b+1]
                        rel_dict_b = self._compute_relevance(
                            target_token_ids=chosen_tok_b,
                            mode="default",
                            multiplier=100.0,
                            scaler=1.0,
                            thresholding=0.5,
                            task="generation",
                            debug=False,
                        )
                        entry = self._store_relevance_entry(
                            rel_dict_b,
                            policy=cache_policy,
                            step_idx=step_offset * beams + b,
                            cache_dir=cache_dir_path,
                            target_dtype=cache_dtype,
                            move_to_cpu=relevance_move_to_cpu,
                            use_compression=relevance_use_compression,
                            compression_method=relevance_compression_method,
                            pickle_protocol=relevance_pickle_protocol,
                        )
                        step_rel_scores.append(entry)
                        self._clear_dlb_memory()

                    relevance_trace_beam.append(step_rel_scores)

            # Stopping: HF early stopping OR stopping criteria
            if beam_scorer.is_done:
                stopped_by = "beam_scorer_done"
                break
            crit = stopping_criteria(self._as_long(generated[:1, :]), None)  # slice to [1, T]
            if self._criteria_true(crit):
                stopped_by = "stopping_criteria"
                break
        else:
            stopped_by = "loop_exhausted"

        if debug:
            print(f"[beam] stopped_by={stopped_by}")

        # Safety: if loop never ran, guard shapes
        if last_topk_tokens is None or last_topk_indices is None:
            last_topk_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
            last_topk_indices = torch.zeros((1, 1), dtype=torch.long, device=device)

        # Finalize across HF versions:
        try:
            final = beam_scorer.finalize(
                input_ids=self._as_long(generated),
                final_beam_scores=self._as_float(beam_scores).to(device),
                final_beam_tokens=last_topk_tokens.to(device),
                final_beam_indices=last_topk_indices.to(device),
                pad_token_id=PAD,
                eos_token_id=eos_list if eos_list else None,
                max_length=max_len_for_beam,
            )
        except TypeError:
            final = beam_scorer.finalize(
                input_ids=self._as_long(generated),
                final_beam_scores=self._as_float(beam_scores).to(device),
                pad_token_id=PAD,
                eos_token_id=eos_list if eos_list else None,
                max_length=max_len_for_beam,
            )

        sequences = final["sequences"]
        out_top1 = sequences[:1, :]

        want_extras = return_scores or return_relevance or return_layerwise_output
        if not want_extras:
            return out_top1

        info_beam = {}
        if return_scores:
            info_beam["scores_trace"] = scores_trace_beam
        if return_relevance:
            flat_relevance = [
                step_rels[0] if isinstance(step_rels, (list, tuple)) and len(step_rels) > 0 else {}
                for step_rels in relevance_trace_beam
            ]
            info_beam["relevance_trace"] = flat_relevance
            info_beam["relevance_cache_policy"] = cache_policy
            if cache_dir_path is not None:
                info_beam["relevance_cache_dir"] = str(cache_dir_path)
        if return_layerwise_output:
            flat_io_trace = [
                step_ios[0] if isinstance(step_ios, (list, tuple)) and len(step_ios) > 0 else {}
                for step_ios in io_data_trace_beam
            ]
            info_beam["layerwise_output_trace"] = flat_io_trace
        if disk_streaming_beam and cache_dir_path is not None:
            info_beam["cache_dir"] = str(cache_dir_path)
        return out_top1, info_beam
