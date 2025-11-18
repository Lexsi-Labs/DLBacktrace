# moe_auto_sampler.py
# Auto sampler for MoE models that uses MoE Backtrace for generation
# Adapted from DLBAutoSampler for MoE-specific workflow

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


class MoEAutoSampler:
    """
    MoE-native text generation (single-prompt => B=1) supporting:
      • Greedy (temperature/top_k/top_p all None)
      • Sampling: temperature / top-k / top-p (when num_beams == 1)
      • Deterministic beam search via HF BeamSearchScorer (when num_beams > 1)

    All logits come from MoE Backtrace:
        all_out, all_in, _ = self.moe_bt.compute_outputs(...)
        logits = all_out[str(step)]['lm_head']  # [batch, T, V]

    Returns:
      Always a single sequence with shape [1, T_total]
      (If return_scores=True on the sampling path, returns (sequence, info_dict))
    """

    def __init__(self, moe_bt, tokenizer):
        self.moe_bt = moe_bt
        self.tokenizer = tokenizer

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
            "Pass AutoModelForCausalLM (not base model)."
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
    def _extract_logits_from_output(all_out_dict: dict, step_key: str) -> torch.Tensor:
        """
        Extract logits from MoE model output dictionary.
        Handles different MoE architectures with different key names.
        
        Args:
            all_out_dict: Output dictionary from compute_outputs
            step_key: Step index as string
            
        Returns:
            torch.Tensor: Logits tensor [1, T, V]
        """
        step_output = all_out_dict[step_key]
        
        # Try common LM head key names for different MoE architectures
        possible_keys = [
            'decoder_lm_head',  # OLMoE
            'lm_head',          # Generic
            'output',           # Some models
        ]
        
        for key in possible_keys:
            if key in step_output:
                return step_output[key]
        
        # Fallback: search for any key containing 'lm' or 'head'
        lm_keys = [k for k in step_output.keys() if 'lm' in k.lower() or 'head' in k.lower()]
        if lm_keys:
            return step_output[lm_keys[0]]
        
        # If still not found, raise informative error
        raise KeyError(
            f"Could not find LM head output in step {step_key}. "
            f"Available keys: {list(step_output.keys())}"
        )
    
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

    def _compute_relevance_for_step(
        self,
        all_in,
        all_out,
        step_idx,
        target_token_id,
        *,
        mode="default",
        multiplier=100.0,
        scaler=1.0,
        thresholding=0.5,
        debug=False,
    ):
        """
        Compute relevance for a specific generation step using MoE Backtrace.
        
        Args:
            all_in: Input activations dict from compute_outputs
            all_out: Output activations dict from compute_outputs
            step_idx: Generation step index (0-based)
            target_token_id: The token ID that was generated at this step
            mode: Relevance propagation mode
            multiplier: Starting relevance value
            scaler: Relevance scaling factor
            thresholding: Relevance threshold
            debug: Enable debug logging
        
        Returns:
            tuple: (all_wt_dict, expert_relevance_dict)
                - all_wt_dict: Token relevance scores for all nodes
                - expert_relevance_dict: Expert-specific relevance scores
        """
        step_key = str(step_idx)
        
        # Get the activations for this specific step
        step_all_in = all_in.get(step_key, {})
        step_all_out = all_out.get(step_key, {})
        
        # Compute relevance using MoE Backtrace proportional_eval
        # This populates both self.moe_bt.all_wt and self.moe_bt.all_layer_expert_relevance
        all_wt = self.moe_bt.proportional_eval(
            all_in=step_all_in,
            all_out=step_all_out,
            start_wt=[],
            multiplier=multiplier,
            scaler=scaler,
            max_unit=0,
            predicted_token=target_token_id,
            thresholding=thresholding,
            task="generation",
        )
        
        # Return both token relevance and expert relevance separately
        import copy
        expert_relevance = copy.deepcopy(self.moe_bt.all_layer_expert_relevance)
        
        return all_wt, expert_relevance

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
        num_return_sequences: Optional[int] = None,  # ignored on return; always top-1
        length_penalty: float = 1.0,
        # misc
        return_scores: bool = False,
        return_layerwise_output: bool = False,
        return_relevance: bool = False,
        debug: bool = False,
    ):
        """
        Generate text using MoE Backtrace.
        
        Always returns:
            - [1, T_total] (top-1 sequence)
            - Or (sequence, info_dict) when return_scores/return_relevance/return_layerwise_output=True
        """
        model = self._get_causallm(self.moe_bt.model)
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
        T, K, P = self._clean_sampling_knobs(temperature, top_k, top_p)
        do_sample = self._decide_do_sample(T, K, P)

        start_len = input_ids.shape[1]
        stopping_criteria = self._build_stopping(start_len, max_new_tokens=max_new_tokens, max_time=max_time)

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
            # Decode the initial prompt to text
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            
            # Initialize traces
            scores_trace = [] if return_scores else None
            relevance_trace = [] if return_relevance else None
            io_data_trace = [] if return_layerwise_output else None
            
            # Track generated tokens
            generated_tokens = []
            
            stopped_by = None
            for step_idx in range(max_new_tokens if max_new_tokens is not None else 10_000_000):
                # Compute outputs using MoE Backtrace
                # For each step, we need to run compute_outputs with the current sequence
                current_text = prompt_text + self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                if debug:
                    print(f"Step {step_idx}: Computing outputs for text length {len(current_text)}")
                
                # Run MoE Backtrace compute_outputs for single step
                all_out, all_in, _ = self.moe_bt.compute_outputs(
                    input_text=current_text,
                    tokenizer=self.tokenizer,
                    max_length=1  # Generate one token at a time
                )
                
                # Extract logits from the last step
                last_step_key = str(len(all_out) - 1)
                logits = self._extract_logits_from_output(all_out, last_step_key)
                
                if torch.is_tensor(logits):
                    logits = logits.to(device)
                else:
                    logits = torch.tensor(logits, device=device, dtype=torch.float32)
                
                next_logits = self._as_float(logits[:, -1, :])  # [1, V]
                
                # Build current input_ids tensor for processors
                current_ids = torch.cat([
                    input_ids,
                    torch.tensor([generated_tokens], device=device, dtype=torch.long) if generated_tokens else torch.empty((1, 0), device=device, dtype=torch.long)
                ], dim=1)
                
                # Apply processors
                scores = logits_processor(self._as_long(current_ids), next_logits)
                if scores.device != device:
                    scores = scores.to(device)
                
                # Enforce min_new_tokens by masking EOS until allowed
                if min_new_tokens is not None and step_idx < min_new_tokens and eos_list:
                    scores[:, eos_list] = -1e9
                
                # Sampling vs greedy
                if do_sample and logits_warper is not None:
                    scores = logits_warper(self._as_long(current_ids), scores)
                    probs = F.softmax(scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)  # LongTensor
                else:
                    next_tokens = scores.argmax(dim=-1, keepdim=True)      # LongTensor
                
                # Force to same device
                next_tokens = self._as_long(next_tokens).to(device)
                next_token_id = int(next_tokens.view(-1)[0].item())
                
                if return_scores:
                    scores_trace.append(scores.detach().to("cpu"))
                
                if return_layerwise_output:
                    io_data_trace.append({'all_in': all_in, 'all_out': all_out})
                
                if return_relevance:
                    # Compute relevance for this step
                    # Returns both token relevance (all_wt) and expert relevance separately
                    all_wt, expert_rel = self._compute_relevance_for_step(
                        all_in=all_in,
                        all_out=all_out,
                        step_idx=int(last_step_key),
                        target_token_id=next_token_id,
                        mode="default",
                        multiplier=100.0,
                        scaler=1.0,
                        thresholding=0.5,
                        debug=False,
                    )
                    # Store both separately in the trace
                    relevance_trace.append({
                        'all_wt': all_wt,
                        'expert_relevance': expert_rel
                    })
                
                # Add token to generated sequence
                generated_tokens.append(next_token_id)
                
                # Early stop if EOS produced
                if eos_list and next_token_id in eos_set:
                    stopped_by = "eos"
                    break
                
                # HF-native stopping criteria (max_new_tokens / max_time)
                final_ids = torch.cat([
                    input_ids,
                    torch.tensor([generated_tokens], device=device, dtype=torch.long)
                ], dim=1)
                crit = stopping_criteria(self._as_long(final_ids[:1, :]), None)
                if self._criteria_true(crit):
                    stopped_by = "stopping_criteria"
                    break
            else:
                stopped_by = "loop_exhausted"
            
            if debug:
                print(f"[greedy/sampling] stopped_by={stopped_by}")
            
            # Build final output tensor
            generated = torch.cat([
                input_ids,
                torch.tensor([generated_tokens], device=device, dtype=torch.long)
            ], dim=1)
            
            want_extras = return_scores or return_relevance or return_layerwise_output
            if not want_extras:
                return generated  # [1, T]
            
            info = {}
            if return_scores:
                info["scores_trace"] = scores_trace
            if return_relevance:
                info["relevance_trace"] = relevance_trace
            if return_layerwise_output:
                info["layerwise_output_trace"] = io_data_trace
            return generated, info  # ([1, T], dict)

        # ======================
        # Beam search path (deterministic, no sampling) — always return top-1
        # ======================
        else:
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
                num_beam_hyps_to_keep=1,  # keep only the best hypothesis
                max_length=max_len_for_beam,
            )

            # Decode the initial prompt to text
            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            
            # Initialize traces
            scores_trace_beam = [] if return_scores else None
            relevance_trace_beam = [] if return_relevance else None
            io_data_trace_beam = [] if return_layerwise_output else None

            # Track generated tokens for each beam
            beam_generated_tokens = [[] for _ in range(beams)]
            
            # Local beam scores
            beam_scores = torch.zeros((1, beams), dtype=torch.float32, device=device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view(-1).to(device)  # [beams] on device

            cur_len = start_len
            vocab_size = None
            start_time = time.time() if max_time is not None else None

            # Keep top-k tensors from the last step (needed by some HF versions)
            last_topk_tokens: Optional[torch.Tensor] = None
            last_topk_indices: Optional[torch.Tensor] = None

            stopped_by = None
            for step_idx in range(max_new_tokens if max_new_tokens is not None else 10_000_000):
                # time guard
                if start_time is not None and (time.time() - start_time) >= max_time:
                    if debug:
                        print("[beam] max_time reached, stopping.")
                    stopped_by = "max_time"
                    break

                # ---- Query MoE Backtrace per beam ----
                next_logits_list: List[torch.Tensor] = []
                io_data_step: List[dict] = [] if return_layerwise_output else None

                for b in range(beams):
                    # Build current text for this beam
                    current_text = prompt_text + self.tokenizer.decode(
                        beam_generated_tokens[b], skip_special_tokens=False
                    )
                    
                    if debug and step_idx == 0:
                        print(f"Beam {b}: Computing outputs for text length {len(current_text)}")
                    
                    # Run MoE Backtrace compute_outputs for single step
                    all_out_b, all_in_b, _ = self.moe_bt.compute_outputs(
                        input_text=current_text,
                        tokenizer=self.tokenizer,
                        max_length=1  # Generate one token at a time
                    )
                    
                    if return_layerwise_output:
                        io_data_step.append({'all_in': all_in_b, 'all_out': all_out_b})

                    # Extract logits from the last step
                    last_step_key = str(len(all_out_b) - 1)
                    logits_b = self._extract_logits_from_output(all_out_b, last_step_key)
                    
                    if torch.is_tensor(logits_b):
                        logits_b = logits_b.to(device)
                    else:
                        logits_b = torch.tensor(logits_b, device=device, dtype=torch.float32)
                    
                    nl_b = self._as_float(logits_b[:, -1, :])  # [1, V] float on device
                    next_logits_list.append(nl_b)
                    if vocab_size is None:
                        vocab_size = nl_b.size(-1)

                if return_layerwise_output:
                    io_data_trace_beam.append(io_data_step)  # per-step, per-beam

                next_logits = torch.cat(next_logits_list, dim=0)  # [beams, V] on device

                # Build current input_ids tensor for processors
                # For beam search, we need to track the full sequence for each beam
                current_ids_list = []
                for b in range(beams):
                    beam_ids = torch.cat([
                        input_ids,
                        torch.tensor([beam_generated_tokens[b]], device=device, dtype=torch.long) if beam_generated_tokens[b] else torch.empty((1, 0), device=device, dtype=torch.long)
                    ], dim=1)
                    current_ids_list.append(beam_ids)
                current_ids = torch.cat(current_ids_list, dim=0)  # [beams, T]

                # processors (no warpers in deterministic beam)
                scores = logits_processor(self._as_long(current_ids), next_logits)
                if scores.device != device:
                    scores = scores.to(device)

                # mask EOS until min_new_tokens is satisfied
                if min_new_tokens is not None and step_idx < min_new_tokens and eos_list:
                    scores[:, eos_list] = -1e9

                # convert to log-prob and add previous beam scores
                next_token_scores = F.log_softmax(scores, dim=-1)  # [beams, V]
                next_token_scores = next_token_scores + beam_scores[:, None]  # stays on device

                # Select top 2*beams across all (beam, vocab) pairs
                flat = next_token_scores.view(1, beams * vocab_size)  # [1, beams*V]
                topk = min(2 * beams, beams * vocab_size)
                next_scores, next_tokens = torch.topk(flat, k=topk, dim=1)
                # all on device already, but be explicit for safety:
                next_scores = next_scores.to(device)
                next_tokens = next_tokens.to(device)

                next_indices = (next_tokens // vocab_size).to(device)
                next_tokens = (next_tokens % vocab_size).to(device)

                # keep the *last* top-k tensors for finalize() (shape [1, topk])
                last_topk_tokens = self._as_long(next_tokens).to(device)
                last_topk_indices = self._as_long(next_indices).to(device)

                # Let HF scorer decide which beams continue / finish
                beam_outputs = beam_scorer.process(
                    input_ids=self._as_long(current_ids),
                    next_scores=self._as_float(next_scores).to(device),
                    next_tokens=self._as_long(next_tokens).to(device),
                    next_indices=self._as_long(next_indices).to(device),
                    pad_token_id=PAD,
                    eos_token_id=eos_list if eos_list else None,
                    beam_indices=None,
                )

                if return_scores:
                    scores_trace_beam.append(next_scores.detach().to("cpu"))

                # Rebuild the new beam batch using public keys
                next_beam_scores = beam_outputs["next_beam_scores"].to(device)  # [beams]
                next_beam_tokens = beam_outputs["next_beam_tokens"].to(device)  # [beams]
                next_beam_indices = beam_outputs["next_beam_indices"].to(device)  # [beams]

                # Update beam_generated_tokens based on beam reordering
                new_beam_generated_tokens = []
                for b in range(beams):
                    src_beam_idx = int(next_beam_indices[b].item())
                    new_tokens = beam_generated_tokens[src_beam_idx].copy()
                    new_tokens.append(int(next_beam_tokens[b].item()))
                    new_beam_generated_tokens.append(new_tokens)
                beam_generated_tokens = new_beam_generated_tokens

                beam_scores = self._as_float(next_beam_scores).to(device)
                cur_len += 1

                if return_relevance:
                    step_rel_scores = []
                    for b in range(beams):
                        # Build current text for this beam
                        current_text_b = prompt_text + self.tokenizer.decode(
                            beam_generated_tokens[b], skip_special_tokens=False
                        )
                        
                        # Compute outputs for relevance
                        all_out_rel, all_in_rel, _ = self.moe_bt.compute_outputs(
                            input_text=current_text_b,
                            tokenizer=self.tokenizer,
                            max_length=1
                        )
                        
                        last_step_key_rel = str(len(all_out_rel) - 1)
                        chosen_tok_b = next_beam_tokens[b:b+1]  # tensor([token_id]) on device
                        
                        # Returns both token relevance and expert relevance separately
                        all_wt_b, expert_rel_b = self._compute_relevance_for_step(
                            all_in=all_in_rel,
                            all_out=all_out_rel,
                            step_idx=int(last_step_key_rel),
                            target_token_id=int(chosen_tok_b.item()),
                            mode="default",
                            multiplier=100.0,
                            scaler=1.0,
                            thresholding=0.5,
                            debug=False,
                        )
                        step_rel_scores.append({
                            'all_wt': all_wt_b,
                            'expert_relevance': expert_rel_b
                        })

                    relevance_trace_beam.append(step_rel_scores)

                # Stopping: HF early stopping OR stopping criteria
                if beam_scorer.is_done:
                    stopped_by = "beam_scorer_done"
                    break
                
                # Check stopping criteria on the first beam
                first_beam_ids = torch.cat([
                    input_ids,
                    torch.tensor([beam_generated_tokens[0]], device=device, dtype=torch.long)
                ], dim=1)
                crit = stopping_criteria(self._as_long(first_beam_ids[:1, :]), None)
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

            # Build final input_ids for all beams for finalize
            final_beam_ids_list = []
            for b in range(beams):
                beam_ids = torch.cat([
                    input_ids,
                    torch.tensor([beam_generated_tokens[b]], device=device, dtype=torch.long)
                ], dim=1)
                final_beam_ids_list.append(beam_ids)
            final_beam_ids = torch.cat(final_beam_ids_list, dim=0)  # [beams, T]

            # Finalize across HF versions:
            try:
                final = beam_scorer.finalize(
                    input_ids=self._as_long(final_beam_ids),
                    final_beam_scores=self._as_float(beam_scores).to(device),
                    final_beam_tokens=last_topk_tokens.to(device),
                    final_beam_indices=last_topk_indices.to(device),
                    pad_token_id=PAD,
                    eos_token_id=eos_list if eos_list else None,
                    max_length=max_len_for_beam,
                )
            except TypeError:
                final = beam_scorer.finalize(
                    input_ids=self._as_long(final_beam_ids),
                    final_beam_scores=self._as_float(beam_scores).to(device),
                    pad_token_id=PAD,
                    eos_token_id=eos_list if eos_list else None,
                    max_length=max_len_for_beam,
                )

            sequences = final["sequences"]  # [1, T_total] because num_beam_hyps_to_keep=1
            out_top1 = sequences[:1, :]  # ensure [1, T_total]

            want_extras = return_scores or return_relevance or return_layerwise_output
            if not want_extras:
                return out_top1

            info_beam = {}
            if return_scores:
                info_beam["scores_trace"] = scores_trace_beam
            if return_relevance:
                # collapse to top-1 beam (final winner)
                flat_relevance = [
                    step_rels[0] if isinstance(step_rels, (list, tuple)) and len(step_rels) > 0 else {}
                    for step_rels in relevance_trace_beam
                ]
                info_beam["relevance_trace"] = flat_relevance
            if return_layerwise_output:
                # collapse to top-1 beam (final winner)
                flat_io_trace = [
                    step_ios[0] if isinstance(step_ios, (list, tuple)) and len(step_ios) > 0 else {}
                    for step_ios in io_data_trace_beam
                ]
                info_beam["layerwise_output_trace"] = flat_io_trace
            return out_top1, info_beam