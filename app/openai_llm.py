"""Optional OpenAI-compatible BaseLlm adapter for running Atlas locally.

Activated ONLY when DEMO_OPENAI_BASE_URL is set (for example Ollama at
http://127.0.0.1:11434/v1 or sixi-asi at http://127.0.0.1:8080/v1). When unset,
app/agent.py uses Google ADK's default Gemini(Vertex AI) backend unchanged.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def _schema_to_json(schema: Any) -> dict[str, Any]:
    """Convert a google.genai Schema (or dict) into standard JSON Schema."""
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        raw = schema
    elif hasattr(schema, "model_dump"):
        raw = schema.model_dump(exclude_none=True)
    else:
        return {"type": "object", "properties": {}}
    return _normalize_schema(raw)


def _normalize_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.lower()
        else:
            out[k] = _normalize_schema(v)
    return out


def _extract_system_instruction(llm_request: LlmRequest) -> str:
    cfg = llm_request.config
    if cfg is None or cfg.system_instruction is None:
        return ""
    si = cfg.system_instruction
    if isinstance(si, str):
        return si
    if isinstance(si, types.Content):
        return "\n".join(p.text for p in (si.parts or []) if p.text)
    if isinstance(si, list):
        parts: list[str] = []
        for item in si:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text") and item.text:
                parts.append(item.text)
        return "\n".join(parts)
    return str(si)


def _extract_openai_tools(llm_request: LlmRequest) -> list[dict[str, Any]]:
    cfg = llm_request.config
    if cfg is None or not cfg.tools:
        return []
    out: list[dict[str, Any]] = []
    for tool in cfg.tools:
        decls = getattr(tool, "function_declarations", None) or []
        for fn in decls:
            params = _schema_to_json(
                getattr(fn, "parameters_json_schema", None)
                or getattr(fn, "parameters", None)
            )
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn.name,
                        "description": fn.description or "",
                        "parameters": params,
                    },
                }
            )
    return out


def _contents_to_messages(
    system_instruction: str, contents: list[types.Content] | None
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    for content in contents or []:
        role = content.role or "user"
        parts = content.parts or []

        text_chunks: list[str] = []
        fn_calls: list[dict[str, Any]] = []
        fn_responses: list[dict[str, Any]] = []

        for part in parts:
            if getattr(part, "text", None):
                text_chunks.append(part.text)
            if getattr(part, "function_call", None):
                fc = part.function_call
                call_id = fc.id or f"call_{uuid.uuid4().hex[:8]}"
                fn_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args or {}),
                        },
                    }
                )
            if getattr(part, "function_response", None):
                fr = part.function_response
                call_id = fr.id or f"call_{fr.name}"
                fn_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": fr.name,
                        "content": json.dumps(fr.response or {}),
                    }
                )

        if fn_responses:
            messages.extend(fn_responses)
            if text_chunks:
                messages.append({"role": "user", "content": "\n".join(text_chunks)})
            continue

        if role == "model":
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_chunks) if text_chunks else "",
            }
            if fn_calls:
                msg["tool_calls"] = fn_calls
            messages.append(msg)
        else:
            messages.append({"role": "user", "content": "\n".join(text_chunks)})

    return messages


class OpenAICompatLlm(BaseLlm):
    """ADK BaseLlm backed by any OpenAI-compatible /v1/chat/completions server."""

    base_url: str
    api_key: str = "local"
    timeout_s: float = 120.0
    temperature: float | None = 0.0
    max_tokens: int = 1024
    reasoning_effort: str | None = None

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r".*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        sys_inst = _extract_system_instruction(llm_request)
        messages = _contents_to_messages(sys_inst, llm_request.contents)
        tools = _extract_openai_tools(llm_request)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = tools

        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="")]),
                partial=False,
            )
            return

        message = choices[0].get("message") or {}
        parts: list[types.Part] = []
        text = message.get("content") or ""
        if text:
            parts.append(types.Part(text=text))

        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    parsed_args = {"raw": raw_args}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}
            if not isinstance(parsed_args, dict):
                parsed_args = {"value": parsed_args}
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        name=name,
                        args=parsed_args,
                    )
                )
            )

        if not parts:
            parts.append(types.Part(text=""))

        usage_raw = data.get("usage") or {}
        usage = None
        if usage_raw:
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=int(usage_raw.get("prompt_tokens") or 0),
                candidates_token_count=int(usage_raw.get("completion_tokens") or 0),
                total_token_count=int(usage_raw.get("total_tokens") or 0),
            )

        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=usage,
            partial=False,
        )
