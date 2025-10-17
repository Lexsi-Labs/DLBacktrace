# dlb_auto_sampler.py
# Auto sampler that uses DL-BacktraceFX for logits and matches HF sampling/beam semantics
# transformers >= 4.30 (tested on 4.52.x); supports older HF via MaxNewTokensCriteria fallback

from __future__ import annotations

import time
from typing import Optional, List, Tuple, cast

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.beam_search import BeamSearchScorer

# ---- Stopping criteria (with fallback for older Transformers) ----
try:
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        MaxTimeCriteria,
        MaxNewTokensCriteria,  # newer HF
    )
except ImportError:  # older HF
    from transformers.generation.stopping_criteria import (
        StoppingCriteriaList,
        MaxTimeCriteria,
        StoppingCriteria,
    )

    class MaxNewTokensCriteria(StoppingCriteria):
        def __init__(self, start_length: int, max_new_tokens: int):
            self.start_length = int(start_length)
            self.max_new_tokens = int(max_new_tokens)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            cur = input_ids.shape[1]
            return (cur - self.start_length) >= self.max_new_tokens


# Optional: steadier math (helps parity on CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False


class DLBAutoSampler:
    """
    DLB-native text generation (single-prompt => B=1) supporting:
      • Greedy (temp/top_k/top_p all None)
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

    # ---------- small dtype helpers ----------

    @staticmethod
    def _as_long(x: torch.Tensor) -> torch.LongTensor:
        return cast(torch.LongTensor, x.long())

    @staticmethod
    def _as_float(x: torch.Tensor) -> torch.FloatTensor:
        return cast(torch.FloatTensor, x.float())

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
    def _build_stopping(start_len: int, max_new_tokens: Optional[int], max_time: Optional[float]) -> StoppingCriteriaList:
        sc = StoppingCriteriaList()
        if max_new_tokens is not None:
            # Works for both native and fallback MaxNewTokensCriteria
            try:
                sc.append(MaxNewTokensCriteria(max_new_tokens=int(max_new_tokens)))
            except TypeError:
                sc.append(MaxNewTokensCriteria(start_length=int(start_len), max_new_tokens=int(max_new_tokens)))
        if max_time is not None:
            sc.append(MaxTimeCriteria(max_time=float(max_time)))
        return sc

    # ---------- public API ----------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        # sampling knobs
        temp: Optional[float] = None,
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
        num_return_sequences: Optional[int] = None,  # ignored on return; always top-1
        length_penalty: float = 1.0,
        # misc
        return_scores: bool = False,
        debug: bool = False,
    ):
        """
        Always returns:
            - [1, T_total] (top-1 sequence)
            - Or (sequence, scores_trace) for sampling when return_scores=True
        """
        model = self._get_causallm(self.dlb.model)
        device = input_ids.device
        B = input_ids.size(0)
        assert B == 1, "Current implementation assumes batch size = 1."

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
        T, K, P = self._clean_sampling_knobs(temp, top_k, top_p)
        do_sample = self._decide_do_sample(T, K, P)

        start_len = input_ids.shape[1]
        stopping_criteria = self._build_stopping(start_len, max_new_tokens=max_new_tokens, max_time=max_time)

        # If we're in BEAM mode, **hard-disable** sampling knobs to avoid warnings
        if num_beams > 1:
            do_sample = False
            T = K = P = None  # silence "ignored flag" warnings

        # Early stopping for beams (use bool for compatibility across HF versions)
        if num_beams > 1:
            if isinstance(early_stopping, bool):
                do_early_stopping = early_stopping
            elif early_stopping == "always":
                do_early_stopping = True
            else:  # "never" or None or anything else
                do_early_stopping = False
        else:
            do_early_stopping = False

        # Prepare GenerationConfig for processors
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,  # sampling only when num_beams == 1 and any knob is set
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
            generated = self._as_long(input_ids.clone())
            attn = self._as_long(attention_mask.clone())
            scores_trace = [] if return_scores else None

            stopped_by = None
            for _ in range(max_new_tokens if max_new_tokens is not None else 10_000_000):
                # Ask DLB for logits (B=1)
                io_data = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
                logits = self._extract_last_logits(io_data)        # [1, T_cur, V]
                next_logits = self._as_float(logits[:, -1, :])     # Float for processors

                # processors
                scores = logits_processor(self._as_long(generated), next_logits)

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

                if return_scores:
                    scores_trace.append(scores.detach().to("cpu"))

                generated = torch.cat([generated, self._as_long(next_tokens)], dim=1)
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
                if stopping_criteria(self._as_long(generated), None):
                    stopped_by = "stopping_criteria"
                    break
            else:
                stopped_by = "loop_exhausted"

            if debug:
                print(f"[greedy/sampling] stopped_by={stopped_by}")

            return (generated, scores_trace) if return_scores else generated  # [1, T]

        # ======================
        # Beam search path (deterministic, no sampling) — always return top-1
        # ======================
        beams = int(num_beams)
        assert beams >= 2, "num_beams must be >= 2 for beam search."

        # Some HF versions require max_length when early stopping is configured (especially when it was a string).
        # Define a conservative cap: prompt length + max_new_tokens (or +1024 fallback).
        max_len_for_beam = int(start_len + (max_new_tokens if max_new_tokens is not None else 1024))

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=do_early_stopping,  # bool for cross-version compatibility
            num_beam_hyps_to_keep=1,              # keep only the best hypothesis per item
            max_length=max_len_for_beam,          # fixes older HF requirement
        )

        # Expand seeds for bookkeeping (we will still call DLB per-beam)
        generated = self._as_long(input_ids.expand(beams, -1).contiguous())  # [beams, T]
        attn = self._as_long(attention_mask.expand(beams, -1).contiguous())

        # Local beam scores (public, no private attrs)
        beam_scores = torch.zeros((1, beams), dtype=torch.float32, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)                                    # [beams]

        cur_len = start_len
        vocab_size = None
        start_time = time.time() if max_time is not None else None

        stopped_by = None
        for _ in range(max_new_tokens if max_new_tokens is not None else 10_000_000):
            # time guard
            if start_time is not None and (time.time() - start_time) >= max_time:
                if debug:
                    print("[beam] max_time reached, stopping.")
                stopped_by = "max_time"
                break

            # ---- Query DLB per beam (keep B=1 for the traced graph) ----
            next_logits_list: List[torch.Tensor] = []
            for b in range(beams):
                io_b = self.dlb.predict(generated[b:b+1], attn[b:b+1], debug=False, temperature=1.0)
                logits_b = self._extract_last_logits(io_b)         # [1, T_cur, V]
                nl_b = self._as_float(logits_b[:, -1, :])          # [1, V] Float for processors
                next_logits_list.append(nl_b)
                if vocab_size is None:
                    vocab_size = nl_b.size(-1)

            next_logits = torch.cat(next_logits_list, dim=0)       # [beams, V] FloatTensor

            # processors (no warpers in deterministic beam)
            scores = logits_processor(self._as_long(generated), next_logits)

            # mask EOS until min_new_tokens is satisfied (hard guard)
            if min_new_tokens is not None and (cur_len - start_len) < min_new_tokens and eos_list:
                scores[:, eos_list] = -1e9

            # convert to log-prob and add previous beam scores
            next_token_scores = F.log_softmax(scores, dim=-1)      # [beams, V]
            next_token_scores = next_token_scores + beam_scores[:, None]

            # Select top 2*beams across all (beam, vocab) pairs
            flat = next_token_scores.view(1, beams * vocab_size)   # [1, beams*V]
            topk = min(2 * beams, beams * vocab_size)
            next_scores, next_tokens = torch.topk(flat, k=topk, dim=1)
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # Let HF scorer decide which beams continue / finish
            beam_outputs = beam_scorer.process(
                input_ids=self._as_long(generated),
                next_scores=self._as_float(next_scores),
                next_tokens=self._as_long(next_tokens),
                next_indices=self._as_long(next_indices),
                pad_token_id=PAD,
                eos_token_id=eos_list if eos_list else None,
                beam_indices=None,
            )

            # Rebuild the new beam batch using public keys
            next_beam_scores = beam_outputs["next_beam_scores"]      # [beams]
            next_beam_tokens = beam_outputs["next_beam_tokens"]      # [beams]
            next_beam_indices = beam_outputs["next_beam_indices"]    # [beams]

            generated = torch.cat(
                [generated[self._as_long(next_beam_indices), :], self._as_long(next_beam_tokens).unsqueeze(-1)],
                dim=1,
            )
            attn = torch.cat(
                [attn[self._as_long(next_beam_indices), :],
                 torch.ones((beams, 1), dtype=attn.dtype, device=attn.device)],
                dim=1,
            )
            beam_scores = self._as_float(next_beam_scores)
            cur_len += 1

            # Stopping: HF early stopping OR stopping criteria
            if beam_scorer.is_done:
                stopped_by = "beam_scorer_done"
                break
            if stopping_criteria(self._as_long(generated), None):
                stopped_by = "stopping_criteria"
                break
        else:
            stopped_by = "loop_exhausted"

        if debug:
            print(f"[beam] stopped_by={stopped_by}")

        # Finalize: HF pads as needed, returns top-N; we take top-1 and keep it 2D: [1, T]
        final = beam_scorer.finalize(
            input_ids=self._as_long(generated),
            final_beam_scores=self._as_float(beam_scores),
            best_indices=None,
            pad_token_id=PAD,
            eos_token_id=eos_list if eos_list else None,
            max_length=max_len_for_beam,
        )
        sequences = final["sequences"]  # [1, T_total] because num_beam_hyps_to_keep=1
        out_top1 = sequences[:1, :]     # ensure [1, T_total]
        return out_top1
