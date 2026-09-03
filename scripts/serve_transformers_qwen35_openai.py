#!/usr/bin/env python3
"""Minimal OpenAI chat-completions server for Qwen3.5 via Transformers.

vLLM 0.12 cannot execute Qwen3_5ForConditionalGeneration. This server keeps
the experimental clients unchanged while using the official implementation.
Requests are serialized because one model replica serves one selected GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=1024, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0)
    stream: bool = False


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                pieces.append(str(part.get("text", "")))
        return "\n".join(pieces)
    return str(content)


def normalized_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        normalized.append(
            {"role": role, "content": [{"type": "text", "text": message_text(message)}]}
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    generation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        del model
        torch.cuda.empty_cache()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": args.served_model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        async with generation_lock:
            inputs = processor.apply_chat_template(
                normalized_messages(request.messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            max_new_tokens = min(request.max_tokens or 1024, args.max_new_tokens)
            do_sample = bool(request.temperature and request.temperature > 0.0)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": processor.tokenizer.pad_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = request.temperature
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            completion_ids = generated[:, inputs.input_ids.shape[1] :]
            content = processor.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        prompt_tokens = int(inputs.input_ids.shape[1])
        completion_tokens = int(completion_ids.shape[1])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
