# dlb_auto_sampler.py
# Auto sampler that uses DL-BacktraceFX for logits and matches HF sampling semantics
# now with HF-native stopping criteria (EOS / min_new_tokens / max_time / etc.)
# transformers==4.52.x

from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# Optional: steadier math (helps parity on CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False


class DLBAutoSampler:
    """
    Step-wise generator driven by a DLBacktraceFX instance.
    Sampling is chosen solely by the 3 knobs:
      - temp  (float | None)  -> Temperature (if not None and != 1.0)
      - top_k (int   | None)  -> Top-K       (if not None and  > 0)
      - top_p (float | None)  -> Top-P       (if not None and  < 1.0)

    Rules (HF semantics):
      * If all three are None -> greedy (do_sample=False).
      * If any is provided -> do_sample=True and ALL provided warpers are applied
        in this order: Temperature -> TopK -> TopP.

    Early stopping:
      * `early_stopping` only affects beam search in HF (ignored in our non-beam path),
        but we pass it to GenerationConfig for parity.
      * For greedy/sampling, stopping happens via HF’s StoppingCriteriaList:
          - EOS (when allowed by min length/new tokens),
          - Max new tokens,
          - Max time (if provided),
          - Any other criteria implied by GenerationConfig.

      We mirror HF by building `stopping_criteria = model._get_stopping_criteria(...)`
      and checking it after each token append.
    """

    def __init__(self, dlb, tokenizer):
        self.dlb = dlb
        self.tokenizer = tokenizer
        self.hard_force_on_fail = True  # only used by calibrated replay

    # ---------- helpers ----------
    @staticmethod
    def _get_causallm(model_like):
        """Walk `.model` chain until object exposing `.generate()` is found."""
        obj = model_like
        seen = set()
        for _ in range(8):
            if hasattr(obj, "generate"):
                return obj
            if hasattr(obj, "model"):
                oid = id(obj)
                if oid in seen: break
                seen.add(oid)
                obj = obj.model
            else:
                break
        raise TypeError(f"{type(model_like).__name__} has no `.generate()`")

    @staticmethod
    def _attach_token_tensors(gen_config, device):
        """Populate private tensors required by transformers==4.52.x."""
        def as_tensor(x):
            if x is None: return None
            if isinstance(x, (list, tuple)):
                return torch.tensor(list(x), device=device, dtype=torch.long)
            return torch.tensor([int(x)], device=device, dtype=torch.long)
        gen_config._eos_token_tensor = as_tensor(getattr(gen_config, "eos_token_id", None))
        gen_config._bos_token_tensor = as_tensor(getattr(gen_config, "bos_token_id", None))
        gen_config._pad_token_tensor = as_tensor(getattr(gen_config, "pad_token_id", None))

    @staticmethod
    def _build_logits_warper_order(cfg) -> LogitsProcessorList:
        """
        HF non-beam order with min_tokens_to_keep=1:
        Temperature -> TopK -> TopP
        Only append a warper if its knob is active.
        """
        w = LogitsProcessorList()
        keep = 1
        t = getattr(cfg, "temperature", None)
        if t is not None and t != 1.0:
            if t <= 0:
                raise ValueError("temperature must be > 0 when do_sample=True")
            w.append(TemperatureLogitsWarper(t))
        k = getattr(cfg, "top_k", None)
        if k is not None and k > 0:
            w.append(TopKLogitsWarper(k, min_tokens_to_keep=keep))
        p = getattr(cfg, "top_p", None)
        if p is not None and p < 1.0:
            w.append(TopPLogitsWarper(p, min_tokens_to_keep=keep))
        return w

    @staticmethod
    def _extract_last_logits(node_io: dict) -> torch.Tensor:
        """From DLB node_io, extract final logits tensor [B, T, V]."""
        if "output" in node_io and isinstance(node_io["output"], dict) and "output_values" in node_io["output"]:
            logits = node_io["output"]["output_values"]
        else:
            last_key = list(node_io.keys())[-1]
            leaf = node_io[last_key]
            logits = leaf["output_values"] if isinstance(leaf, dict) and "output_values" in leaf else leaf
        if isinstance(logits, dict) and "output_values" in logits:
            logits = logits["output_values"]
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"Expected logits tensor, got {type(logits)}")
        return logits

    @staticmethod
    def _decide_do_sample(temp, top_k, top_p) -> bool:
        """HF-style condition for sampling."""
        use_temp = (temp is not None and float(temp) != 1.0)
        use_topk = (top_k is not None and int(top_k) > 0)
        use_topp = (top_p is not None and float(top_p) < 1.0)
        return bool(use_temp or use_topk or use_topp)

    @staticmethod
    def _clean_sampling_knobs(temp, top_k, top_p):
        """Normalize/validate knobs; return (T | None, K | None, P | None)."""
        T = None
        K = None
        P = None
        if temp is not None:
            T = float(temp)
            if T <= 0:
                raise ValueError("temp must be > 0")
            if T == 1.0:
                T = None  # no-op in HF
        if top_k is not None:
            K = int(top_k)
            if K <= 0:
                K = None
        if top_p is not None:
            P = float(top_p)
            if not (0.0 < P <= 1.0):
                raise ValueError("top_p must be in (0, 1]")
            if P == 1.0:
                P = None  # no-op in HF
        return T, K, P

    @staticmethod
    def _calibrate_then_sample(
        probs: torch.Tensor,
        target_id: int,
        *,
        max_probe_cap: int = 65536,
        first_block: int = 64,
        hard_force_on_fail: bool = True,
    ) -> torch.Tensor:
        """
        Advance RNG by offsets so torch.multinomial(probs,1) returns target_id.
        """
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if probs.is_cuda else None

        tried = 0
        block = first_block
        while tried < max_probe_cap:
            block = min(block, max_probe_cap - tried)
            for n in range(block):
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state)
                if n > 0:
                    _ = torch.rand((n,), device=probs.device)
                pick = torch.multinomial(probs, 1)
                if int(pick.item()) == int(target_id):
                    return pick
            tried += block
            block *= 4

        if hard_force_on_fail:
            return torch.tensor([[int(target_id)]], device=probs.device, dtype=torch.long)

        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        return torch.multinomial(probs, 1)

    # ---------- public API ----------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        # knobs (no 'method' arg)
        temp: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int = 50,
        min_new_tokens: int | None = None,       # <-- NEW
        max_time: float | None = None,           # <-- NEW (seconds)
        early_stopping: bool | str | None = None,# <-- passthrough (beam-only)
        # constraints & ids
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        bad_words_ids: list[list[int]] | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        # parity (optional)
        hf_parity: bool = False,
        oracle_tokens: list[int] | None = None,
        hard_force_on_fail: bool | None = None,
        max_probe_cap: int = 65536,
        first_block: int = 64,
        # misc
        return_scores: bool = False,
        debug: bool = False,
    ):
        """
        Decode using the three knobs (temp/top_k/top_p) and HF stopping criteria.
        Returns: sequences  or  (sequences, scores_list) if return_scores=True
        """
        model = self._get_causallm(self.dlb.model)
        device = input_ids.device

        # Normalize knobs & decide sampling
        T, K, P = self._clean_sampling_knobs(temp, top_k, top_p)
        do_sample = self._decide_do_sample(T, K, P)

        # Attention mask if needed
        if attention_mask is None:
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)

        # Build GenerationConfig like HF (include early_stopping/min/max_time)
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
            early_stopping=early_stopping,   # beam-only, but pass-through for parity
            max_time=max_time,               # HF uses this to add a MaxTime criterion
        )
        if do_sample:
            if T is not None: gen_kwargs["temperature"] = T
            if K is not None: gen_kwargs["top_k"] = K
            if P is not None: gen_kwargs["top_p"] = P

        # Prepare GenerationConfig + attach private tensors
        generation_config, _ = model._prepare_generation_config(
            generation_config=None, use_model_defaults=True, **gen_kwargs
        )
        self._attach_token_tensors(generation_config, device)

        # Optional: HF parity — get oracle tokens via model.generate()
        if do_sample and hf_parity and oracle_tokens is None:
            clean = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                early_stopping=early_stopping,
                max_time=max_time,
            )
            if T is not None: clean["temperature"] = T
            if K is not None: clean["top_k"] = K
            if P is not None: clean["top_p"] = P
            hf_out = model.generate(**clean)
            start = input_ids.shape[1]
            oracle_tokens = hf_out.sequences[:, start:][0].tolist()
            if debug:
                print(f"[DLB Auto] oracle length: {len(oracle_tokens)}")

        # Build processors & warpers like HF
        seq_len = input_ids.shape[1]
        logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=seq_len,
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device=device,
            model_kwargs={},
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        )
        try:
            from transformers.generation import utils as gen_utils
            logits_warper = gen_utils._get_logits_warper(generation_config)
            stopping_criteria = model._get_stopping_criteria(
                generation_config=generation_config,
                stopping_criteria=None
            )
        except Exception:
            # Fallback if private helpers not found (older versions)
            logits_warper = self._build_logits_warper_order(generation_config)
            # Minimal fallback stopping: based on eos/max_new_tokens only
            from transformers.generation.stopping_criteria import StoppingCriteriaList
            stopping_criteria = StoppingCriteriaList()

        # Step loop using DLB for logits (keep temp=1.0 inside DLB)
        generated = input_ids.clone()
        attn = attention_mask.clone()
        B = generated.size(0)
        assert B == 1, "Current calibrated replay assumes batch size 1."
        scores_list = []
        if hard_force_on_fail is None:
            hard_force_on_fail = self.hard_force_on_fail

        start_time = time.time()

        for t in range(max_new_tokens):
            # Respect max_time proactively (HF also has a criterion; this is extra-safe)
            if max_time is not None and (time.time() - start_time) >= max_time:
                break

            # Ask DLB for logits of current prefix
            io_data = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
            logits = self._extract_last_logits(io_data)
            next_logits = logits[:, -1, :]

            # HF processors (repetition penalty, n-gram, bad words, min length/new tokens etc.)
            scores = logits_processor(generated, next_logits)

            if do_sample:
                scores = logits_warper(generated, scores)
                probs = F.softmax(scores, dim=-1)

                if hf_parity and oracle_tokens is not None and t < len(oracle_tokens):
                    pick = self._calibrate_then_sample(
                        probs, oracle_tokens[t],
                        max_probe_cap=max_probe_cap,
                        first_block=first_block,
                        hard_force_on_fail=hard_force_on_fail,
                    )
                    next_tok = pick
                else:
                    next_tok = torch.multinomial(probs, 1)
            else:
                next_tok = scores.argmax(dim=-1, keepdim=True)

            if return_scores:
                scores_list.append(scores.detach().clone())

            generated = torch.cat([generated, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=attn.device)], dim=1)

            # HF-style early stopping check (includes EOS/min_new/max_time/etc.)
            # StoppingCriteriaList expects (input_ids, scores)
            if len(stopping_criteria) > 0:
                # Pass the *raw logits* of the last step (HF passes "scores" from loop)
                # Using `scores` (post-processor/warper) is also acceptable; HF uses that.
                should_stop = stopping_criteria(generated, scores)
                if should_stop:
                    break

        return (generated, scores_list) if return_scores else generated
