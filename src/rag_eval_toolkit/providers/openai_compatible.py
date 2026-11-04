"""Opt-in Chat Completions adapter using the standard library, with injectable transport."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from email.message import Message
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import Field, JsonValue, SecretStr, field_validator

from rag_eval_toolkit.core.errors import ConfigurationError, JudgeError
from rag_eval_toolkit.core.models import EvaluationSample, Model
from rag_eval_toolkit.core.serialization import canonical_json
from rag_eval_toolkit.judges.base import JudgeMetric, JudgeVerdict
from rag_eval_toolkit.judges.prompts import PROMPT_VERSION, build_messages


class JudgeOptions(Model):
    base_url: str
    model: str = Field(min_length=1)
    allow_network: bool = False
    timeout: float = Field(default=30.0, gt=0, le=300)
    max_input_chars: int = Field(default=100_000, ge=1)
    max_response_bytes: int = Field(default=1_000_000, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    seed: int | None = 0
    json_mode: bool = True

    @field_validator("model")
    @classmethod
    def model_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            _ = parsed.port
            if (
                parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or any(char.isspace() for char in value)
            ):
                raise ValueError("invalid endpoint")
            if parsed.scheme == "http":
                local = parsed.hostname.lower() == "localhost"
                if not local:
                    try:
                        local = ipaddress.ip_address(parsed.hostname).is_loopback
                    except ValueError:
                        local = False
                if not local:
                    raise ValueError("HTTP is permitted only for loopback inference servers")
        except ValueError:
            raise ValueError("Use HTTPS or loopback HTTP, without credentials/query/fragment") from None
        return value.rstrip("/")


class ChatTransport(Protocol):
    def send(
        self, url: str, payload: bytes, headers: Mapping[str, str], *,
        timeout: float, max_response_bytes: int,
    ) -> bytes: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Message, newurl: str,
    ) -> None:
        return None


class UrllibTransport:
    """Bounded response reads, verified TLS, no redirects and no automatic retries."""

    def send(
        self, url: str, payload: bytes, headers: Mapping[str, str], *,
        timeout: float, max_response_bytes: int,
    ) -> bytes:
        request = Request(url, data=payload, headers=dict(headers), method="POST")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
                body = response.read(max_response_bytes + 1)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise JudgeError(f"Judge HTTP request failed (status {code})") from None
        except (URLError, OSError, ValueError):
            raise JudgeError("Judge transport failed or timed out; check endpoint/connectivity") from None
        if len(body) > max_response_bytes:
            raise JudgeError("Judge response exceeds configured size limit")
        return bytes(body)


class OpenAICompatibleJudge:
    """Explicitly configured OpenAI-compatible remote API or loopback server."""

    def __init__(
        self, options: JudgeOptions, *, api_key: str | None = None,
        transport: ChatTransport | None = None,
    ) -> None:
        if not options.allow_network:
            raise ConfigurationError("Judge networking requires allow_network=True")
        self.options = options
        self._api_key = SecretStr(api_key) if api_key else None
        self._transport = transport if transport is not None else UrllibTransport()

    def provenance(self) -> dict[str, JsonValue]:
        return {
            "provider": "openai_compatible", "base_url": self.options.base_url,
            "model": self.options.model, "temperature": self.options.temperature,
            "seed": self.options.seed, "max_tokens": self.options.max_tokens,
            "prompt_version": PROMPT_VERSION, "json_mode": self.options.json_mode,
            "max_input_chars": self.options.max_input_chars,
            "max_response_bytes": self.options.max_response_bytes,
            "timeout": self.options.timeout,
        }

    def evaluate(self, sample: EvaluationSample, metric: JudgeMetric) -> JudgeVerdict:
        messages = build_messages(sample, metric)
        if sum(len(message["content"]) for message in messages) > self.options.max_input_chars:
            raise JudgeError("Judge input exceeds configured limit; no data was sent")
        payload: dict[str, object] = {
            "model": self.options.model, "messages": messages, "max_tokens": self.options.max_tokens,
        }
        if self.options.temperature is not None:
            payload["temperature"] = self.options.temperature
        if self.options.seed is not None:
            payload["seed"] = self.options.seed
        if self.options.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        try:
            body = self._transport.send(
                self.options.base_url + "/chat/completions", canonical_json(payload).encode("utf-8"),
                headers, timeout=self.options.timeout, max_response_bytes=self.options.max_response_bytes,
            )
        except JudgeError:
            raise
        except Exception:
            raise JudgeError("Configured judge transport failed") from None
        if len(body) > self.options.max_response_bytes:
            raise JudgeError("Judge response exceeds configured size limit")
        try:
            envelope = json.loads(body)
            if not isinstance(envelope, dict) or not isinstance(envelope.get("choices"), list):
                raise ValueError("invalid completion envelope")
            choice = envelope["choices"][0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise ValueError("invalid completion choice")
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal"):
                raise ValueError("incomplete/refused completion")
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("content must be JSON text")
            return JudgeVerdict.model_validate_json(content)
        except (ValueError, TypeError, KeyError, IndexError, RecursionError):
            raise JudgeError("Judge returned malformed, incomplete, or invalid structured output") from None
