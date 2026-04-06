#!/usr/bin/env python3
"""
DLBacktrace — FastAPI Server

Loads a model on startup and exposes REST endpoints for running
backtrace relevance computation.

Usage:
  # MoE backend (auto-detected)
  python server.py --model allenai/OLMoE-1B-7B-0125-Instruct --device cuda

  # PyTorch backend
  python server.py --model meta-llama/Llama-3.2-1B --backend pytorch --device cuda

  # Custom port
  python server.py --model allenai/OLMoE-1B-7B-0125-Instruct --port 9000
"""

import argparse
import gc
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Library imports ─────────────────────────────────────────────────────────

try:
    from dl_backtrace.version import __version__
except ImportError:
    __version__ = "0.1.1"

logger = logging.getLogger("dl_backtrace.server")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Banner
# ═══════════════════════════════════════════════════════════════════════════

BANNER = r"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                            ║
║                                                                                            ║
║  ██████╗ ██╗     ██████╗  █████╗  ██████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗███████╗ ║
║  ██╔══██╗██║     ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝ ║
║  ██║  ██║██║     ██████╔╝███████║██║     █████╔╝    ██║   ██████╔╝███████║██║     █████╗   ║
║  ██║  ██║██║     ██╔══██╗██╔══██║██║     ██╔═██╗    ██║   ██╔══██╗██╔══██║██║     ██╔══╝   ║
║  ██████╔╝███████╗██████╔╝██║  ██║╚██████╗██║  ██╗   ██║   ██║  ██║██║  ██║╚██████╗███████╗ ║
║  ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝ ║
║                                                                                            ║
║                                                                                            ║
║                  Deep Learning Explainability & Attribution Engine                         ║
║                              API Server v{version}                                         ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
""".replace("{version}", __version__)


# ═══════════════════════════════════════════════════════════════════════════
#  Pydantic Schemas
# ═══════════════════════════════════════════════════════════════════════════

class RelevanceFormat(str, Enum):
    full = "full"
    summary = "summary"
    top_k = "top_k"


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    prompt: str = Field(..., description="Input text prompt", min_length=1)
    max_new_tokens: int = Field(10, ge=1, le=512, description="Max tokens to generate")
    temperature: float = Field(1.0, gt=0.0, le=10.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, description="Top-k sampling (0 = disabled)")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="Nucleus sampling threshold")
    return_relevance: bool = Field(True, description="Return per-step relevance traces")
    return_scores: bool = Field(False, description="Return per-step score traces")
    relevance_format: RelevanceFormat = Field(
        RelevanceFormat.summary,
        description="How to serialize relevance: full (arrays), summary (stats), top_k (top nodes)"
    )
    relevance_top_k: int = Field(10, ge=1, description="Number of top nodes when relevance_format=top_k")
    multiplier: float = Field(100.0, description="Starting relevance multiplier")
    scaler: float = Field(1.0, description="Relevance scaling factor")
    thresholding: float = Field(0.5, description="Relevance threshold")
    debug: bool = Field(False, description="Enable debug logging")
    explain_tokens: Union[str, int, List[int]] = Field(
        "all",
        description=(
            'Which tokens to compute DLB relevance for. '
            '"all" (default) — every generated token, '
            '"none" — skip relevance, '
            'an integer N — first N tokens only, '
            'or a list of specific token indices e.g. [0, 4, 9]'
        ),
    )
    relevance_cache_policy: str = Field(
        "disk",
        description=(
            'Cache policy for per-step relevance data. '
            '"disk" — stream to disk (low RAM, engine.py default), '
            '"memory" — keep in RAM (faster but higher RAM usage)'
        ),
    )
    cache_dir: str = Field(
        "cache",
        description="Directory for disk-streamed relevance/scores/IO data (used when relevance_cache_policy='disk')",
    )


class NodeRelevanceSummary(BaseModel):
    """Summary stats for a single node's relevance."""
    sum: float
    mean: float
    max: float
    min: float
    shape: List[int]


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""
    generated_text: str
    generated_token_ids: List[int]
    input_token_ids: List[int]
    num_tokens_generated: int
    time_seconds: float
    model_id: str
    backend: str
    relevance_trace: Optional[List[Dict[str, Any]]] = None
    scores_trace: Optional[List[Any]] = None


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str = "ok"
    version: str
    model_id: str
    backend: str
    device: str
    gpu_name: Optional[str] = None
    gpu_vram_total_mb: Optional[float] = None


class SupportedModel(BaseModel):
    """Info about a supported MoE model."""
    model_type: str
    description: str
    class_names: List[str]
    example_ids: List[str]


# ═══════════════════════════════════════════════════════════════════════════
#  Model Wrappers (reused from engine.py)
# ═══════════════════════════════════════════════════════════════════════════

class ModelWrapper(nn.Module):
    """Wrapper for standard PyTorch DLBacktrace backend (float32)."""
    def __init__(self, model_id: str, token: str):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, token=token
        ).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


class MoEModelWrapper(nn.Module):
    """Wrapper for MoE Backtrace backend (bfloat16)."""
    def __init__(self, model_id: str, token: str):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, token=token
        ).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits


# ═══════════════════════════════════════════════════════════════════════════
#  Relevance Serialization Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_python(obj):
    """Recursively convert numpy/torch objects to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _summarize_node(arr) -> Dict[str, Any]:
    """Compute summary statistics for a relevance array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray):
        return {
            "sum": float(np.sum(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "shape": list(arr.shape),
        }
    return {"value": _to_python(arr)}


def serialize_relevance_trace(
    trace: list,
    fmt: RelevanceFormat,
    top_k: int = 10,
) -> list:
    """
    Serialize a relevance_trace list according to the requested format.

    Each entry in `trace` is a dict like:
      - MoE: {'all_wt': {node: arr, ...}, 'expert_relevance': {node: arr, ...}}
      - PyTorch: {node: arr, ...}
    """
    result = []
    for step_data in trace:
        if not isinstance(step_data, dict):
            result.append(_to_python(step_data))
            continue

        step_out = {}
        for section_key, section_val in step_data.items():
            if not isinstance(section_val, dict):
                # Scalar or non-dict value (e.g. metadata)
                step_out[section_key] = _to_python(section_val)
                continue

            if fmt == RelevanceFormat.full:
                step_out[section_key] = {
                    str(k): _to_python(v) for k, v in section_val.items()
                }

            elif fmt == RelevanceFormat.summary:
                step_out[section_key] = {
                    str(k): _summarize_node(v) for k, v in section_val.items()
                }

            elif fmt == RelevanceFormat.top_k:
                # Rank nodes by abs sum of relevance, keep top-k
                scored = []
                for k, v in section_val.items():
                    if isinstance(v, np.ndarray):
                        score = float(np.sum(np.abs(v)))
                    elif isinstance(v, torch.Tensor):
                        score = float(v.abs().sum().item())
                    else:
                        score = 0.0
                    scored.append((str(k), score, v))
                scored.sort(key=lambda x: x[1], reverse=True)
                step_out[section_key] = {
                    name: {
                        "relevance_score": score,
                        "values": _to_python(arr),
                    }
                    for name, score, arr in scored[:top_k]
                }

        result.append(step_out)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Server State
# ═══════════════════════════════════════════════════════════════════════════

class ServerState:
    """Holds the loaded model, tokenizer, and backtrace engine."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.backtrace_engine = None  # DLBacktrace or MoE Backtrace
        self.model_id: str = ""
        self.backend: str = ""  # "pytorch" or "moe"
        self.device: str = "cpu"
        self.ready: bool = False

    def load(self, model_id: str, backend: str, device: str):
        """Load model, tokenizer, and initialize backtrace engine."""
        from transformers import AutoTokenizer

        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.model_id = model_id
        self.backend = backend
        self.device = device

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if backend == "moe":
            logger.info("Loading MoE model: %s (bfloat16)", model_id)
            self.model = MoEModelWrapper(model_id, hf_token)

            from dl_backtrace.moe_pytorch_backtrace import Backtrace
            logger.info("Initializing MoE Backtrace engine...")
            self.backtrace_engine = Backtrace(
                model=self.model,
                model_type=None,  # auto-detect
                device=device,
            )

        elif backend == "pytorch":
            logger.info("Loading PyTorch model: %s (float32)", model_id)
            self.model = ModelWrapper(model_id, hf_token)

            # We don't initialize DLBacktrace here because it needs
            # sample inputs for graph tracing. It will be initialized
            # on first request with the actual input shapes.
            self.backtrace_engine = None  # lazy init

        self.ready = True
        logger.info("✅ Server ready — model: %s, backend: %s, device: %s",
                     model_id, backend, device)

    def run_moe_generate(self, req: GenerateRequest) -> Dict[str, Any]:
        """Run generation using the MoE Backtrace backend."""
        tokens = self.tokenizer(req.prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        os.makedirs(req.cache_dir, exist_ok=True)
        t0 = time.perf_counter()
        results = self.backtrace_engine.run_task(
            task="generation",
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            tokenizer=self.tokenizer,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            return_relevance=req.return_relevance,
            return_scores=req.return_scores,
            multiplier=req.multiplier,
            scaler=req.scaler,
            thresholding=req.thresholding,
            debug=req.debug,
            top_k=req.top_k if req.top_k > 0 else None,
            top_p=req.top_p,
            explain_tokens=req.explain_tokens,
            relevance_cache_policy=req.relevance_cache_policy,
            relevance_cache_dir=req.cache_dir,
        )
        elapsed = time.perf_counter() - t0

        # Decode generated text
        gen_ids = results["generated_ids"]
        input_len = input_ids.shape[1]
        new_ids = gen_ids[0, input_len:].tolist() if gen_ids.dim() > 1 else gen_ids[0].tolist()[input_len:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Serialize relevance
        relevance_trace = None
        if req.return_relevance and "relevance_trace" in results:
            relevance_trace = serialize_relevance_trace(
                results["relevance_trace"],
                fmt=req.relevance_format,
                top_k=req.relevance_top_k,
            )

        scores_trace = None
        if req.return_scores and "scores_trace" in results:
            scores_trace = _to_python(results["scores_trace"])

        return {
            "generated_text": generated_text,
            "generated_token_ids": new_ids,
            "input_token_ids": input_ids[0].tolist(),
            "num_tokens_generated": len(new_ids),
            "time_seconds": round(elapsed, 4),
            "model_id": self.model_id,
            "backend": self.backend,
            "relevance_trace": relevance_trace,
            "scores_trace": scores_trace,
        }

    def run_pytorch_generate(self, req: GenerateRequest) -> Dict[str, Any]:
        """Run generation using the PyTorch DLBacktrace backend."""
        from dl_backtrace.pytorch_backtrace import DLBacktrace
        from torch.export import Dim

        tokens = self.tokenizer(
            [req.prompt], return_tensors="pt", padding=True, truncation=True,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        input_len = input_ids.shape[1]

        # Dynamic shapes for graph tracing
        seq_dim = Dim("seq", min=1, max=input_len)
        dynamic_shapes = {
            "input_ids":      {0: 1, 1: seq_dim},
            "attention_mask": {0: 1, 1: seq_dim},
        }

        # Initialize DLBacktrace (re-traced per request to handle varying shapes)
        ir = DLBacktrace(
            self.model,
            (input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes,
            device=self.device,
            verbose=False,
        )

        os.makedirs(req.cache_dir, exist_ok=True)
        t0 = time.perf_counter()
        results = ir.run_task(
            task="generation",
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            tokenizer=self.tokenizer,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            return_relevance=req.return_relevance,
            return_scores=req.return_scores,
            multiplier=req.multiplier,
            scaler=req.scaler,
            thresholding=req.thresholding,
            debug=req.debug,
            top_k=req.top_k if req.top_k > 0 else None,
            top_p=req.top_p,
            explain_tokens=req.explain_tokens,
            relevance_cache_policy=req.relevance_cache_policy,
            relevance_cache_dir=req.cache_dir,
        )
        elapsed = time.perf_counter() - t0

        # Decode generated text
        gen_ids = results.get("generated_ids")
        if gen_ids is not None:
            new_ids = gen_ids[0, input_len:].tolist() if gen_ids.dim() > 1 else []
            generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            new_ids = []
            generated_text = ""

        # Serialize relevance
        relevance_trace = None
        if req.return_relevance and "relevance_trace" in results:
            relevance_trace = serialize_relevance_trace(
                results["relevance_trace"],
                fmt=req.relevance_format,
                top_k=req.relevance_top_k,
            )

        scores_trace = None
        if req.return_scores and "scores_trace" in results:
            scores_trace = _to_python(results["scores_trace"])

        # Cleanup
        del ir
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "generated_text": generated_text,
            "generated_token_ids": new_ids,
            "input_token_ids": input_ids[0].tolist(),
            "num_tokens_generated": len(new_ids),
            "time_seconds": round(elapsed, 4),
            "model_id": self.model_id,
            "backend": self.backend,
            "relevance_trace": relevance_trace,
            "scores_trace": scores_trace,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

# Global state — populated by CLI args before uvicorn.run()
_server_state = ServerState()
_cli_args = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup."""
    global _cli_args
    if _cli_args is not None:
        _server_state.load(
            model_id=_cli_args.model,
            backend=_cli_args.backend,
            device=_cli_args.device,
        )
    yield
    # Shutdown cleanup
    logger.info("Shutting down server...")
    if _server_state.model is not None:
        del _server_state.model
        del _server_state.backtrace_engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="DLBacktrace API",
    description="REST API for Deep Learning Explainability & Attribution",
    version=__version__,
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns server status and loaded model info."""
    if not _server_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    info = HealthResponse(
        status="ok",
        version=__version__,
        model_id=_server_state.model_id,
        backend=_server_state.backend,
        device=_server_state.device,
    )
    if torch.cuda.is_available():
        info.gpu_name = torch.cuda.get_device_name()
        info.gpu_vram_total_mb = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1
        )
    return info


@app.get("/models", response_model=List[SupportedModel], tags=["System"])
async def list_models():
    """List all supported MoE model architectures."""
    from dl_backtrace.moe_pytorch_backtrace.backtrace.supported_models import SUPPORTED_MOE_MODELS

    return [
        SupportedModel(
            model_type=mtype,
            description=info["description"],
            class_names=info["class_names"],
            example_ids=info["example_ids"],
        )
        for mtype, info in SUPPORTED_MOE_MODELS.items()
    ]


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(req: GenerateRequest):
    """
    Run backtrace generation.

    Send a prompt and hyperparameters, receive generated text and
    per-token relevance attribution data.
    """
    if not _server_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        if _server_state.backend == "moe":
            result = _server_state.run_moe_generate(req)
        elif _server_state.backend == "pytorch":
            result = _server_state.run_pytorch_generate(req)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unknown backend: {_server_state.backend}"
            )
        return GenerateResponse(**result)

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Try a shorter prompt or fewer max_new_tokens."
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        prog="dlbacktrace-server",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python server.py --model allenai/OLMoE-1B-7B-0125-Instruct --device cuda\n"
            "  python server.py --model meta-llama/Llama-3.2-1B --backend pytorch --device cuda\n"
            "  python server.py --model jetmoe/jetmoe-8b --backend moe --port 9000\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID (e.g. allenai/OLMoE-1B-7B-0125-Instruct)",
    )
    parser.add_argument(
        "--backend", type=str, default="moe", choices=["pytorch", "moe"],
        help="Backend engine: pytorch (DLBacktrace) or moe (MoE Backtrace) (default: moe)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of uvicorn workers (default: 1, use 1 for GPU)",
    )
    parser.add_argument(
        "--log-level", type=str, default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cache",
        help="Server-wide default directory for disk-streamed relevance/scores/IO data (default: cache)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for saving JSON benchmark/result reports (default: results)",
    )
    parser.add_argument(
        "--explain-tokens", nargs="+", default=["all"],
        help=(
            'Server-wide default for which tokens to compute DLB relevance for. '
            '"all" (default) — every generated token, '
            '"none" — skip relevance, '
            'an int N — first N tokens only, '
            'or specific indices like "0 4 9". '
            'Can be overridden per-request via the explain_tokens field.'
        ),
    )
    parser.add_argument(
        "--view-output", action="store_true", default=False,
        help="Load and print saved .dlbr relevance output after each generation (default: off)",
    )
    return parser.parse_args()


def main():
    global _cli_args

    args = parse_args()
    _cli_args = args

    # Resolve --explain-tokens into the right type (mirrors engine.py logic)
    et = args.explain_tokens
    if len(et) == 1 and et[0].lower() in ("all", "none"):
        args.explain_tokens_resolved = et[0].lower()
    elif len(et) == 1 and et[0].isdigit():
        args.explain_tokens_resolved = int(et[0])   # first-N
    else:
        args.explain_tokens_resolved = [int(x) for x in et]  # specific indices

    # Device fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    print(BANNER)
    print(f"  Model          : {args.model}")
    print(f"  Backend        : {args.backend}")
    print(f"  Device         : {args.device}")
    print(f"  Address        : http://{args.host}:{args.port}")
    print(f"  Docs           : http://{args.host}:{args.port}/docs")
    print(f"  Cache dir      : {args.cache_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Explain tokens : {args.explain_tokens_resolved}")
    print(f"  View output    : {args.view_output}")
    print("═" * 70)

    import uvicorn
    uvicorn.run(
        "dl_backtrace.cli.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
