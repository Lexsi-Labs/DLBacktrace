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


# dl_backtrace/pytorch_backtrace/dlbacktrace/core/dlb_auto_sampler.py

from __future__ import annotations
import time
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

class DLBAutoSampler:
    """
    DLB-native text generation (single-batch) supporting:
      • Greedy (temp/top_k/top_p all None)
      • Sampling with any combo of temperature / top-k / top-p when num_beams == 1
      • Deterministic beam search when num_beams > 1 (no sampling)

    All logits come from DLB:
        io = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
        logits = io["output"]["output_values"]  # shape [B_or_beams, T, V]
        step_scores = processors(warpers(..., logits[:, -1, :]))
    """

    def __init__(self, dlb, tokenizer):
        """
        Args:
            dlb: an instance of DLBacktraceFX (your tracing engine)
            tokenizer: HF tokenizer (used only for ids and optional chat templates elsewhere)
        """
        self.dlb = dlb
        self.tokenizer = tokenizer

    # ---------- helpers ----------

    def _get_causallm(self, model_like):
        """Walk `.model` chain until we find a GenerationMixin-style causal LM."""
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
        """Populate private token tensors expected by HF internals in 4.52.x."""
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
            # Fallback: try last dict with 'output_values'
            for k in reversed(list(io_data.keys())):
                v = io_data[k]
                if isinstance(v, dict) and isinstance(v.get("output_values", None), torch.Tensor):
                    return v["output_values"]
        raise KeyError("DLB io_data does not contain 'output' -> 'output_values' tensor.")

    @staticmethod
    def _clean_sampling_knobs(temp: Optional[float], top_k: Optional[int], top_p: Optional[float]) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        T = float(temp) if (temp is not None and temp != 1.0) else None
        K = int(top_k) if (top_k is not None and top_k > 0) else None
        P = float(top_p) if (top_p is not None and top_p < 1.0) else None
        return T, K, P

    @staticmethod
    def _decide_do_sample(T: Optional[float], K: Optional[int], P: Optional[float]) -> bool:
        # HF rule of thumb: any knob triggers sampling
        return (T is not None) or (K is not None) or (P is not None)

    @staticmethod
    def _build_warper(temperature: Optional[float], top_k: Optional[int], top_p: Optional[float]) -> LogitsProcessorList:
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
        early_stopping: Optional[bool | str] = None,   # used only for beam search; default "never"
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
        debug: bool = False,
    ):
        """
        DLB-native generation:
          - num_beams == 1  -> greedy / sampling (temp/top_k/top_p)
          - num_beams  > 1  -> deterministic beam search (no sampling)

        Returns:
            Tensor of shape:
              - [1, T_total] for greedy/sampling
              - [num_return_sequences, T_total] for beams
            If return_scores=True (sampling path only), returns (sequences, scores_list)
        """
        model = self._get_causallm(self.dlb.model)
        device = input_ids.device
        B = input_ids.size(0)
        assert B == 1, "Current implementation assumes batch size = 1."

        # Build / infer attention mask
        if attention_mask is None:
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Normalize knobs
        T, K, P = self._clean_sampling_knobs(temp, top_k, top_p)
        do_sample = self._decide_do_sample(T, K, P)
        num_return_sequences = int(num_return_sequences or 1)

        # Prepare GenerationConfig for processors
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample if num_beams == 1 else False,  # no sampling in beam path here
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
        )
        if num_beams > 1:
            gen_kwargs["early_stopping"] = early_stopping if early_stopping is not None else "never"

        generation_config, _ = model._prepare_generation_config(
            generation_config=None, use_model_defaults=True, **gen_kwargs
        )
        self._attach_token_tensors(generation_config, device)

        # EOS list
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
        if do_sample and num_beams == 1:
            logits_warper = self._build_warper(T, K, P)
        else:
            logits_warper = None

        start_len = input_ids.shape[1]
        start_time = time.time() if max_time is not None else None

        # ======================
        # Non-beam path (B=1)
        # ======================
        if num_beams < 2:
            generated = input_ids.clone()
            attn = attention_mask.clone()
            scores_trace = [] if return_scores else None

            for _ in range(max_new_tokens):
                # Ask DLB for logits
                io_data = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
                logits = self._extract_last_logits(io_data)        # [B, T_cur, V]
                next_logits = logits[:, -1, :]                     # [B, V]

                # processors
                scores = logits_processor(generated, next_logits)

                # enforce min_new_tokens by masking EOS until allowed
                if min_new_tokens is not None and (generated.shape[1] - start_len) < min_new_tokens and eos_list:
                    scores[:, eos_list] = -1e9

                # sampling vs greedy
                if do_sample and logits_warper is not None:
                    scores = logits_warper(generated, scores)
                    probs = F.softmax(scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = scores.argmax(dim=-1, keepdim=True)

                if return_scores:
                    scores_trace.append(scores.detach().to("cpu"))

                # append & update mask
                generated = torch.cat([generated, next_tokens], dim=1)
                attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=attn.device)], dim=1)

                # early stop if everyone hits EOS
                if eos_list:
                    if torch.all(torch.isin(next_tokens.squeeze(-1), torch.tensor(eos_list, device=device))):
                        break

                # time guard
                if start_time is not None and (time.time() - start_time) >= max_time:
                    if debug:
                        print("[dlb] max_time reached, stopping.")
                    break

            return (generated, scores_trace) if return_scores else generated

        # ======================
        # Beam search path (deterministic, no sampling)
        # ======================
        beams = int(num_beams)
        assert beams >= 2, "num_beams must be >= 2 for beam search."

        # Expand beams
        generated = input_ids.repeat_interleave(beams, dim=0)     # [beams, T]
        attn = attention_mask.repeat_interleave(beams, dim=0)
        beam_scores = torch.zeros((1, beams), dtype=torch.float32, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)                        # [beams]

        finished = []  # list of (normed_score, seq)
        cur_len = start_len
        estop = early_stopping if early_stopping is not None else "never"

        for _ in range(max_new_tokens):
            if start_time is not None and (time.time() - start_time) >= max_time:
                if debug:
                    print("[beam] max_time reached, stopping.")
                break

            # DLB logits for all beams
            io_data = self.dlb.predict(generated, attn, debug=False, temperature=1.0)
            logits = self._extract_last_logits(io_data)           # [beams, T_cur, V]
            next_logits = logits[:, -1, :]                        # [beams, V]

            # processors (no warpers in deterministic beam)
            scores = logits_processor(generated, next_logits)

            # mask EOS until min_new_tokens is satisfied
            if min_new_tokens is not None and (cur_len - start_len) < min_new_tokens and eos_list:
                scores[:, eos_list] = -1e9

            # convert to log-prob and add previous beam scores
            logprobs = F.log_softmax(scores, dim=-1)              # [beams, V]
            logprobs = logprobs + beam_scores.unsqueeze(-1)

            V = logprobs.size(-1)
            flat = logprobs.view(1, beams * V)                    # [1, beams*V]
            topk_scores, topk_indices = torch.topk(flat, k=min(2 * beams, beams * V), dim=-1)

            next_generated = []
            next_attn = []
            next_beam_scores = []

            cand_alive = 0
            for rank in range(topk_scores.size(1)):
                score = topk_scores[0, rank]
                idx = topk_indices[0, rank].item()
                beam_id = idx // V
                token_id = idx % V

                src_seq = generated[beam_id:beam_id + 1]
                src_attn = attn[beam_id:beam_id + 1]

                # EOS -> finalize
                if token_id in eos_set:
                    seq = torch.cat([src_seq, torch.tensor([[token_id]], device=device)], dim=1)
                    if length_penalty == 1.0:
                        norm = float(seq.size(1))
                        normed = score / norm
                    else:
                        lp = ((5.0 + seq.size(1)) ** length_penalty) / (6.0 ** length_penalty)
                        normed = score / lp
                    finished.append((float(normed.item()), seq))
                    continue

                # keep alive beam
                new_seq = torch.cat([src_seq, torch.tensor([[token_id]], device=device)], dim=1)
                new_attn = torch.cat([src_attn, torch.ones_like(src_attn[:, :1])], dim=1)
                next_generated.append(new_seq)
                next_attn.append(new_attn)
                next_beam_scores.append(score)
                cand_alive += 1
                if cand_alive == beams:
                    break

            # no alive beams left
            if cand_alive == 0:
                if estop != "never":
                    if debug:
                        print("[beam] all beams finished, stopping.")
                    break
                else:
                    break

            generated = torch.cat(next_generated, dim=0)          # [beams, T+1]
            attn = torch.cat(next_attn, dim=0)
            beam_scores = torch.stack(next_beam_scores).to(device)
            cur_len += 1

            # early stopping True: stop when enough finished hyps
            if estop is True and len(finished) >= num_return_sequences:
                if debug:
                    print("[beam] early_stopping=True and enough finished hyps — stopping.")
                break

            if start_time is not None and (time.time() - start_time) >= max_time:
                if debug:
                    print("[beam] max_time reached, stopping.")
                break

        # Finalize: if not enough finished, take best alive beams (length-penalized)
        if len(finished) < num_return_sequences:
            alive = []
            for i in range(generated.size(0)):
                seq = generated[i:i + 1]
                s = beam_scores[i]
                if length_penalty == 1.0:
                    norm = float(seq.size(1))
                    normed = s / norm
                else:
                    lp = ((5.0 + seq.size(1)) ** length_penalty) / (6.0 ** length_penalty)
                    normed = s / lp
                alive.append((float(normed.item()), seq))
            alive.sort(key=lambda x: x[0], reverse=True)
            need = num_return_sequences - len(finished)
            finished.extend(alive[:need])

        # Select top N sequences
        finished.sort(key=lambda x: x[0], reverse=True)
        selected = [seq for _, seq in finished[:num_return_sequences]]
        out = torch.cat(selected, dim=0)  # [num_return_sequences, T_total]

        # Pad to same length if needed
        max_len = out.size(1)
        if any(seq.size(1) != max_len for _, seq in finished[:num_return_sequences]):
            padded = []
            for _, seq in finished[:num_return_sequences]:
                if seq.size(1) < max_len:
                    pad = torch.full((1, max_len - seq.size(1)), PAD, dtype=seq.dtype, device=seq.device)
                    seq = torch.cat([seq, pad], dim=1)
                padded.append(seq)
            out = torch.cat(padded, dim=0)

        return out
