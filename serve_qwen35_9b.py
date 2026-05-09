import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from test_qwen35_9b_xformers_fla import count_fla_calls, register_xformers_memory_efficient_backend


DEFAULT_MODEL_PATH = (
    "/mnt/data/huggingface-z/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", DEFAULT_MODEL_PATH)
ATTN_IMPLEMENTATION = "xformers_memory_efficient_aligned"

state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "full_attention_calls": None,
    "fla_calls": None,
    "loaded_s": None,
}
generate_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1)
    stop: list[str] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen3.5-9B-4bit-xformers-fla"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def _reset_rope_deltas(model) -> None:
    for module in model.modules():
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None


def _build_chat_prompt(tokenizer, messages: list[ChatMessage]) -> str:
    payload = [message.model_dump() for message in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return "\n".join(f"{message.role}: {message.content}" for message in messages) + "\nassistant:"


def _apply_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    cut = len(text)
    for marker in stop:
        if marker:
            pos = text.find(marker)
            if pos != -1:
                cut = min(cut, pos)
    return text[:cut]


def _load_model() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    state["full_attention_calls"] = register_xformers_memory_efficient_backend()
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).eval()
    state["loaded_s"] = time.perf_counter() - start
    state["tokenizer"] = tokenizer
    state["model"] = model
    state["fla_calls"] = count_fla_calls(model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Qwen3.5-9B 4bit xFormers+FLA Inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = state["model"] is not None
    cuda_info = None
    if torch.cuda.is_available():
        cuda_info = {
            "device": torch.cuda.get_device_name(0),
            "capability": f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
            "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        }
    return {
        "ok": loaded,
        "model_path": MODEL_PATH,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "loaded_s": state["loaded_s"],
        "cuda": cuda_info,
    }


def _generate_text(request: GenerateRequest) -> dict[str, Any]:
    model = state["model"]
    tokenizer = state["tokenizer"]
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    with generate_lock:
        _reset_rope_deltas(model)
        full_before = dict(state["full_attention_calls"])
        fla_before = dict(state["fla_calls"])
        torch.cuda.reset_peak_memory_stats()
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        do_sample = request.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "repetition_penalty": request.repetition_penalty,
        }
        if do_sample:
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_p"] = request.top_p

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

        new_tokens = generated[0, prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = _apply_stop(text, request.stop)
        completion_tokens = int(new_tokens.shape[-1])
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_s": elapsed_s,
            "full_attention_calls": _counter_delta(full_before, state["full_attention_calls"]),
            "fla_calls": _counter_delta(fla_before, state["fla_calls"]),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    return _generate_text(request)


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not implemented")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    tokenizer = state["tokenizer"]
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    prompt = _build_chat_prompt(tokenizer, request.messages)
    result = _generate_text(
        GenerateRequest(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    )
    created = int(time.time())
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
        "backend": {
            "elapsed_s": result["elapsed_s"],
            "full_attention_calls": result["full_attention_calls"],
            "fla_calls": result["fla_calls"],
            "peak_alloc_gib": result["peak_alloc_gib"],
        },
    }
