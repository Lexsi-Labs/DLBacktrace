"""
_compat.py — Transformers version-compatibility shims.

Provides:
  • Logit-warper/processor classes (renamed in transformers ≥ 5.x)
  • BeamHypotheses + BeamSearchScorer (removed from transformers ≥ 4.51),
    adapted from the original HuggingFace implementation.
  • Stopping-criteria classes (API unchanged, but import paths shifted)
"""
from __future__ import annotations

from collections import UserDict
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logits warpers / processors  (renamed in transformers 5.x)
# ---------------------------------------------------------------------------
from transformers.generation.logits_process import LogitsProcessorList  # stable

try:
    from transformers.generation.logits_process import TemperatureLogitsWarper
except ImportError:
    from transformers.generation.logits_process import TemperatureLogitsProcessor as TemperatureLogitsWarper  # type: ignore[no-redef]

try:
    from transformers.generation.logits_process import TopKLogitsWarper
except ImportError:
    from transformers.generation.logits_process import TopKLogitsProcessor as TopKLogitsWarper  # type: ignore[no-redef]

try:
    from transformers.generation.logits_process import TopPLogitsWarper
except ImportError:
    from transformers.generation.logits_process import TopPLogitsProcessor as TopPLogitsWarper  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------
from transformers.generation.stopping_criteria import StoppingCriteriaList  # stable

try:
    from transformers.generation.stopping_criteria import MaxTimeCriteria
except ImportError:
    import time as _time

    class MaxTimeCriteria:  # type: ignore[no-redef]
        def __init__(self, max_time: float):
            self._max_time = max_time
            self._start = _time.time()

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return (_time.time() - self._start) >= self._max_time

try:
    from transformers.generation.stopping_criteria import MaxNewTokensCriteria
except ImportError:
    from transformers.generation.stopping_criteria import StoppingCriteria  # type: ignore[assignment]

    class MaxNewTokensCriteria(StoppingCriteria):  # type: ignore[no-redef]
        def __init__(self, start_length: int, max_new_tokens: int):
            self.start_length = int(start_length)
            self.max_new_tokens = int(max_new_tokens)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return (input_ids.shape[1] - self.start_length) >= self.max_new_tokens

try:
    from transformers.generation.stopping_criteria import EosTokenCriteria
    HAS_EOS_CRITERIA = True
except ImportError:
    HAS_EOS_CRITERIA = False

    class EosTokenCriteria:  # type: ignore[no-redef]
        """Stub for older transformers that lack EosTokenCriteria."""
        def __init__(self, eos_token_id):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            self._eos = set(eos_token_id)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return bool(input_ids[0, -1].item() in self._eos)

# ---------------------------------------------------------------------------
# BeamHypotheses + BeamSearchScorer
# Adapted from the original HuggingFace implementation (transformers ≤ 4.48).
# Used as a fallback when transformers >= 4.51 removed these classes.
#
# Adaptations vs. the original:
#   • Removed @add_start_docstrings (internal HF decorator, not available here)
#   • process()  — eos_token_id accepts int OR list[int]
#   • finalize() — accepts an optional max_length kwarg; returns dict{"sequences"}
#                  instead of a bare tensor (matches our generation-loop callers)
# ---------------------------------------------------------------------------
try:
    from transformers.generation.beam_search import BeamSearchScorer  # old HF ≤ 4.48
except ImportError:

    class BeamHypotheses:
        def __init__(
            self,
            num_beams: int,
            max_length: int,
            length_penalty: float,
            early_stopping: bool,
        ):
            self.max_length = max_length - 1  # ignoring bos_token
            self.length_penalty = length_penalty
            self.early_stopping = early_stopping
            self.num_beams = num_beams
            self.beams = []
            self.worst_score = 1e9

        def __len__(self):
            return len(self.beams)

        def add(self, hyp: torch.LongTensor, sum_logprobs: float):
            score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
            if len(self) < self.num_beams or score > self.worst_score:
                self.beams.append((score, hyp))
                if len(self) > self.num_beams:
                    sorted_next_scores = sorted(
                        [(s, idx) for idx, (s, _) in enumerate(self.beams)]
                    )
                    del self.beams[sorted_next_scores[0][1]]
                    self.worst_score = sorted_next_scores[1][0]
                else:
                    self.worst_score = min(score, self.worst_score)

        def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
            if len(self) < self.num_beams:
                return False
            elif self.early_stopping:
                return True
            else:
                cur_score = best_sum_logprobs / cur_len ** self.length_penalty
                return self.worst_score >= cur_score

    class BeamSearchScorer:  # type: ignore[no-redef]
        """
        Standard beam search scorer.
        Drop-in replacement for transformers.generation.beam_search.BeamSearchScorer.
        """

        def __init__(
            self,
            batch_size: int,
            num_beams: int,
            device: Union[str, torch.device],
            length_penalty: float = 1.0,
            do_early_stopping: bool = False,
            num_beam_hyps_to_keep: int = 1,
            max_length: Optional[int] = None,
        ):
            if not isinstance(num_beams, int) or num_beams <= 1:
                raise ValueError(
                    f"`num_beams` must be an integer > 1, got {num_beams}."
                )
            self.num_beams = num_beams
            self.device = device
            self.length_penalty = length_penalty
            self.do_early_stopping = do_early_stopping
            self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
            self.max_length = max_length or 2048  # used if not overridden in finalize

            self._beam_hyps = [
                BeamHypotheses(
                    num_beams=self.num_beams,
                    max_length=self.max_length,
                    length_penalty=self.length_penalty,
                    early_stopping=self.do_early_stopping,
                )
                for _ in range(batch_size)
            ]
            self._done = torch.tensor(
                [False] * batch_size, dtype=torch.bool, device=self.device
            )

        @property
        def is_done(self) -> bool:
            return bool(self._done.all())

        def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            beam_indices=None,
        ) -> UserDict:
            cur_len = input_ids.shape[-1]
            batch_size = len(self._beam_hyps)
            assert batch_size == (input_ids.shape[0] // self.num_beams)

            # Normalise eos_token_id to a set for O(1) lookup
            if eos_token_id is None:
                eos_set: set = set()
            elif isinstance(eos_token_id, int):
                eos_set = {eos_token_id}
            else:
                eos_set = set(eos_token_id)
            # Use single int for legacy path (None if empty)
            _eos_single = next(iter(eos_set)) if len(eos_set) == 1 else None

            device = input_ids.device
            next_beam_scores = torch.zeros(
                (batch_size, self.num_beams), dtype=next_scores.dtype, device=device
            )
            next_beam_tokens = torch.zeros(
                (batch_size, self.num_beams), dtype=next_tokens.dtype, device=device
            )
            next_beam_indices = torch.zeros(
                (batch_size, self.num_beams), dtype=next_indices.dtype, device=device
            )

            for batch_idx, beam_hyp in enumerate(self._beam_hyps):
                if self._done[batch_idx]:
                    assert len(beam_hyp) >= self.num_beams
                    assert eos_token_id is not None and pad_token_id is not None
                    next_beam_scores[batch_idx, :] = 0
                    next_beam_tokens[batch_idx, :] = pad_token_id
                    next_beam_indices[batch_idx, :] = 0
                    continue

                beam_idx = 0
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(
                        next_tokens[batch_idx],
                        next_scores[batch_idx],
                        next_indices[batch_idx],
                    )
                ):
                    batch_beam_idx = batch_idx * self.num_beams + next_index
                    token_id = next_token.item()
                    if token_id in eos_set:
                        # don't add hypotheses ranked below num_beams
                        if beam_token_rank >= self.num_beams:
                            continue
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            next_score.item(),
                        )
                    else:
                        next_beam_scores[batch_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1

                    if beam_idx == self.num_beams:
                        break

                if beam_idx < self.num_beams:
                    raise ValueError(
                        f"At most {self.num_beams} tokens in {next_tokens[batch_idx]} "
                        f"can equal eos_token_id={eos_token_id}."
                    )

                self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

            return UserDict(
                {
                    "next_beam_scores": next_beam_scores.view(-1),
                    "next_beam_tokens": next_beam_tokens.view(-1),
                    "next_beam_indices": next_beam_indices.view(-1),
                }
            )

        def finalize(
            self,
            input_ids: torch.LongTensor,
            final_beam_scores: torch.FloatTensor,
            final_beam_tokens: Optional[torch.LongTensor] = None,
            final_beam_indices: Optional[torch.LongTensor] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            max_length: Optional[int] = None,
        ) -> dict:
            batch_size = len(self._beam_hyps)
            max_len = max_length or self.max_length

            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]

            # Add all open (non-completed) beams to hypotheses
            for batch_idx, beam_hyp in enumerate(self._beam_hyps):
                if self._done[batch_idx]:
                    continue
                for beam_id in range(self.num_beams):
                    batch_beam_idx = batch_idx * self.num_beams + beam_id
                    final_score = final_beam_scores[batch_beam_idx].item()
                    final_tokens = input_ids[batch_beam_idx]
                    beam_hyp.add(final_tokens, final_score)

            # Select best hypotheses
            sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
            best = []
            for i, beam_hyp in enumerate(self._beam_hyps):
                sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
                for j in range(self.num_beam_hyps_to_keep):
                    best_hyp = sorted_hyps.pop()[1]
                    sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                    best.append(best_hyp)

            # Build output tensor (padded)
            sent_max_len = min(int(sent_lengths.max().item()) + 1, max_len)
            decoded: torch.LongTensor = input_ids.new(
                batch_size * self.num_beam_hyps_to_keep, sent_max_len
            )
            if sent_lengths.min().item() != sent_lengths.max().item():
                assert pad_token_id is not None, "`pad_token_id` must be defined"
                decoded.fill_(pad_token_id)

            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_len and eos_token_id:
                    decoded[i, sent_lengths[i]] = eos_token_id[0]

            # Return dict so callers can do final["sequences"]
            return {"sequences": decoded}
