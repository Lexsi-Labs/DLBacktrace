import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import torch
import torch.nn as nn
from torch.export import Dim
from transformers import AutoTokenizer, AutoModelForCausalLM
from dl_backtrace.pytorch_backtrace import DLBacktrace
import warnings
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import re

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_LOGS"] = "+dynamic"

num_cores = os.cpu_count()
if num_cores:
    torch.set_num_threads(num_cores)
    print(f"PyTorch num_threads set to: {torch.get_num_threads()}")
else:
    print("Could not determine number of CPU cores for PyTorch.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "meta-llama/Llama-3.2-1B"

class LlamaWrapper(nn.Module):
    def __init__(self, model_id, token):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            token=token
        ).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

model = LlamaWrapper(model_name, os.getenv("HUGGING_FACE_HUB_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

def extract_feature_importance(
    relevance_trace,
    input_ids,
    tokenizer,
    token_index=0,
    input_key="input_ids"
):
    if not relevance_trace or token_index >= len(relevance_trace):
        return {
            "feature_importance": {},
            "raw_feature_importance": {}
        }

    if isinstance(input_ids, torch.Tensor):
        prompt_tokens = input_ids.detach().cpu().long().view(-1).tolist()
    else:
        prompt_tokens = list(input_ids)

    tokens = tokenizer.convert_ids_to_tokens(prompt_tokens)

    rel_dict = relevance_trace[token_index]

    key = None
    if input_key in rel_dict:
        key = input_key
    else:
        for k in rel_dict.keys():
            if isinstance(k, str) and any(x in k.lower() for x in ["input", "embed", "token"]):
                key = k
                break
        if key is None:
            key = next(iter(rel_dict.keys()))

    relevance_tensor = rel_dict[key]

    if isinstance(relevance_tensor, torch.Tensor):
        relevance_np = relevance_tensor.detach().cpu().numpy()
    else:
        relevance_np = np.array(relevance_tensor)

    prompt_len = len(prompt_tokens)

    if relevance_np.ndim == 1 and relevance_np.shape[0] == prompt_len:
        per_token_relevance = relevance_np
    elif prompt_len in relevance_np.shape:
        axis = list(relevance_np.shape).index(prompt_len)
        relevance_np = np.moveaxis(relevance_np, axis, 0)
        if relevance_np.ndim > 1:
            per_token_relevance = np.sum(relevance_np, axis=tuple(range(1, relevance_np.ndim)))
        else:
            per_token_relevance = relevance_np
    else:
        flat = relevance_np.ravel()
        per_token_relevance = np.zeros(prompt_len)
        per_token_relevance[:min(len(flat), prompt_len)] = flat[:prompt_len]

    per_token_relevance = per_token_relevance.astype(np.float64)

    raw_feature_importance = dict(zip(tokens, per_token_relevance.tolist()))

    min_val = per_token_relevance.min()
    max_val = per_token_relevance.max()

    if max_val > min_val:
        normalized = (per_token_relevance - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(per_token_relevance)

    normalized = np.round(normalized, 1)

    feature_importance = dict(zip(tokens, normalized.tolist()))
    feature_importance = {
        re.sub(r'[^a-zA-Z0-9_]', '', str(k)): v
        for k, v in feature_importance.items()
    }

    return {
        "feature_importance": feature_importance,
        "raw_feature_importance": raw_feature_importance
    }

@app.get("/health")
def healthcheck():
    return {
        "success": True, 
        "time": datetime.utcnow(), 
        "details": "API working fine"
    }

@app.post("/inference")
def inference(payload: dict):
    try:
        start_time = time.perf_counter()

        sentences = [payload.get("prompt")]
        tokens = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Dynamic shapes
        if len(sentences) > 1:
            batch_dim = Dim("batch", min=1, max=len(sentences))
        else:
            batch_dim = 1  # Static dimension

        seq_dim = Dim("seq", min=1, max=input_ids.shape[1])
        dynamic_shapes = {
            "input_ids": {0: batch_dim, 1: seq_dim},
            "attention_mask": {0: batch_dim, 1: seq_dim},
        }

        print(f"Input prompt: {sentences[0]}")
        print(f"Input IDs shape: {input_ids.shape}")

        ir = DLBacktrace(
            model,
            (input_ids, attention_mask),
            dynamic_shapes=dynamic_shapes,
            device='cuda',  # or 'cpu'
            verbose=False
        )

        results = ir.run_task(
            task="generation",
            inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
            tokenizer=tokenizer,
            min_new_tokens=payload.get("min_tokens", 1),
            max_new_tokens=payload.get("max_tokens", 20),
            dlb_tokens_count=payload.get("dlb_tokens_count", 1),
            temperature=payload.get("temperature"),
            top_k=payload.get("top_k"),
            top_p=payload.get("top_p"),
            num_beams=payload.get("num_beams",1) or 1,
            num_return_sequences=payload.get("num_return_sequences") or 1,
            length_penalty=payload.get("length_penalty"),
            early_stopping=True,
            return_scores=False,
            return_relevance=True,
            return_layerwise_output=False,
            debug=False
        )

        print(f"\n✅ Task completed: {results['task']}")
        print(f"Generated IDs shape: {results['generated_ids'].shape}")

        generated_text = tokenizer.batch_decode(results['generated_ids'], skip_special_tokens=True)
        print(f"\nInput prompt: {sentences[0]}")
        print(f"Generated text: {generated_text[0]}")

        prompt_tokens = len(input_ids[0])
        completion_tokens = len(results['generated_ids'][0])

        if 'relevance_trace' in results:
            print(f"\n📊 Relevance trace: {len(results['relevance_trace'])} steps")
            print(f"   First step nodes: {len(results['relevance_trace'][0])}")

        if 'scores_trace' in results:
            print(f"\n📈 Scores trace: {len(results['scores_trace'])} steps")
            print(f"   First step shape: {results['scores_trace'][0].shape}")

        if 'layerwise_output_trace' in results:
            print(f"\n🔬 Layerwise output trace: {len(results['layerwise_output_trace'])} steps")
            print(f"   First step nodes: {len(results['layerwise_output_trace'][0])}")

        ir.visualize_dlbacktrace(output_path=payload.get("trace_id"), engine_auto_threshold=2500)
        with open(f"backtrace_collapsed_fast.svg", "rb") as image_file:
            graph_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        ir.visualize_input_heatmap_for_token(
            results['relevance_trace'],
            n=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
            generated_ids=results['generated_ids']
        )

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)

        relevance = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        feature_importance = extract_feature_importance(
            relevance_trace=results['relevance_trace'],
            input_ids=input_ids,
            tokenizer=tokenizer,
            token_index=0
        )
        
        explainability = {
            **feature_importance,
            "network_graph": graph_b64,
            "relevance": relevance
        }

        usage = {
            "prompt_tokens": prompt_tokens, 
            "completion_tokens": completion_tokens, 
            "total_tokens": prompt_tokens + completion_tokens
        }

        result = {
            "model_name": model_name,
            "prompt": payload.get("prompt"),
            "output": generated_text[0],
            "usage":  usage,
            "explainability": explainability,
            "inference_duration":  time.perf_counter() - start_time,
        }

        return {
            "success": True,
            "details": result
        }
    except Exception as e:
        print(f"Error during inference => {str(e)}")
        return {
            "success": False,
            "details": "Failed to run inference"
        }