"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import typing
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import EmptyResponseError, OutputTruncatedError, RateLimitError, split_retry_enabled

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'

#: Per-call cost trace. Set GRAPHITI_LLM_TRACE to a file path and every LLM call
#: appends one JSON line with its prompt name and the upstream token accounting.
#: Used to attribute cost to pipeline stages instead of guessing from prompt size.
_TRACE_NAME: 'contextvars.ContextVar[str]' = contextvars.ContextVar('llm_prompt_name', default='')


def _trace_usage(prompt_name: str, response: Any, request_chars: int) -> None:
    path = os.environ.get('GRAPHITI_LLM_TRACE')
    if not path:
        return
    usage = getattr(response, 'usage', None)
    row = {
        'prompt': prompt_name,
        'prompt_tokens': getattr(usage, 'prompt_tokens', 0) if usage else 0,
        'completion_tokens': getattr(usage, 'completion_tokens', 0) if usage else 0,
        'cached_tokens': (
            getattr(usage, 'prompt_cache_hit_tokens', 0)
            or getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0)
        ) if usage else 0,
        'request_chars': request_chars,
    }
    try:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception:  # noqa: BLE001 - tracing must never break a call
        pass

StructuredOutputMode = Literal['json_schema', 'json_object']


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    This client targets any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
    vLLM, llama.cpp, Ollama, DeepSeek, Together, etc.). It defaults to native
    ``json_schema`` structured output (constrained decoding) and can fall back to
    ``json_object`` for the minority of providers that do not support ``json_schema``.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        structured_output_mode (StructuredOutputMode): How structured output is requested.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
        structured_output_mode: StructuredOutputMode = 'json_schema',
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.
            structured_output_mode (StructuredOutputMode): Whether to request structured
                output via native ``json_schema`` (the default, uses constrained decoding)
                or to fall back to ``json_object``. Set to ``'json_object'`` for providers
                that do not support the ``json_schema`` response format (e.g. DeepSeek); in
                that mode the schema is injected into the prompt instead of being enforced
                by the API.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens
        self.structured_output_mode: StructuredOutputMode = structured_output_mode
        self._request_semaphore = asyncio.Semaphore(
            max(1, int(os.getenv('LLM_MAX_CONCURRENCY', '6')))
        )

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _build_response_format(self, response_model: type[BaseModel] | None) -> dict[str, Any]:
        """Build the ``response_format`` payload for the chat completion request.

        Uses native ``json_schema`` when a response model is provided and the client is in
        ``json_schema`` mode; otherwise falls back to ``json_object``. In ``json_object``
        mode the schema is not enforced by the API — ``generate_response`` injects it into
        the prompt instead.
        """
        if response_model is None or self.structured_output_mode == 'json_object':
            return {'type': 'json_object'}

        # Native json_schema. We intentionally omit "strict": true — strict mode requires
        # the schema to meet OpenAI's strict subset (additionalProperties: false, every
        # field required), which raw model_json_schema() routinely violates (that's why the
        # dedicated OpenAIClient uses responses.parse() instead). So adherence is best-effort
        # on OpenAI-proper; constrained-decoding servers (vLLM, llama.cpp) still enforce it.
        return {
            'type': 'json_schema',
            'json_schema': {
                'name': getattr(response_model, '__name__', 'structured_response'),
                'schema': response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip a wrapping markdown code fence from a JSON payload.

        OpenAI-compatible models served via Ollama/llama.cpp etc. frequently wrap their
        output in a ```json … ``` fence even when a json_schema/json_object response_format
        is requested, which breaks a bare ``json.loads``. No-op when there is no fence.
        """
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```[a-zA-Z0-9_-]*[ \t]*\r?\n?', '', stripped)
            stripped = re.sub(r'\r?\n?```[ \t]*$', '', stripped)
        return stripped.strip()

    @staticmethod
    def _salvage_json(raw: str) -> dict[str, Any] | None:
        """Best-effort recovery of a JSON object from malformed/truncated LLM output.

        In ``json_object`` mode (DeepSeek and other non-``json_schema`` providers) the
        API does NOT enforce the schema, so models routinely emit invalid JSON: unescaped
        quotes inside string values, bare newlines, trailing commas, or a truncated tail
        when ``max_tokens`` cuts off mid-object. This routine tries a sequence of cheap
        repairs and returns the first parseable dict, or ``None`` if none work.

        Repairs tried (in order of increasing intrusiveness):
            1. Strip code fences (```json … ```).
            2. Truncate to the last balanced ``}`` (recover from a truncated tail).
            3. Remove trailing commas before ``}`` / ``]``.
            4. Escape raw newlines / tabs inside what look like string values.
            5. A final greedy ``{ ... }`` extraction.

        Returns ``None`` so the original decode error can reach caller-level
        input bisection.
        """
        if not raw or not raw.strip():
            return None

        candidates: list[str] = []

        stripped = OpenAIGenericClient._strip_code_fences(raw)
        candidates.append(stripped)

        # 2. Truncate to the last closing brace — fixes truncation mid-object.
        last_brace = stripped.rfind('}')
        if last_brace > 0:
            candidates.append(stripped[: last_brace + 1])

        # 3. Drop trailing commas (common: `"x": 1,}` or `[..., ]`).
        no_trailing = re.sub(r',(\s*[}\]])', r'\1', stripped)
        if no_trailing != stripped:
            candidates.append(no_trailing)

        # 3b. Insert a missing comma between adjacent tokens. The classic
        #     "Expecting ',' delimiter" failure: `"a" "b"` or `} {` or `] [` with
        #     nothing but whitespace between two structural tokens that belong
        #     in a comma-separated list. Insert a comma only at these junctions;
        #     it can't make valid JSON *worse* because the input already failed.
        def _insert_missing_commas(text: str) -> str:
            # Match end-of-token (string close, }, ], number/true/false/null)
            # followed by whitespace then a start-of-token (string open, {, [).
            # We conservatively handle the common string→string and }/]/value
            # boundaries. Re-run repeatedly isn't needed; one pass covers the
            # typical single missing comma.
            return re.sub(
                r'(["}\]\d])(\s+)(["{\[])',
                r'\1,\2\3',
                text,
            )

        with_commas = _insert_missing_commas(stripped)
        if with_commas != stripped:
            candidates.append(with_commas)
            # combine with trailing-comma cleanup too
            candidates.append(re.sub(r',(\s*[}\]])', r'\1', with_commas))

        # 4. Escape literal control chars that appear *inside* double-quoted
        #    strings. We only touch newlines/tabs between quotes that are not
        #    already escaped — a conservative regex pass.
        def _escape_control_in_strings(text: str) -> str:
            out: list[str] = []
            in_str = False
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == '\\' and in_str and i + 1 < len(text):
                    out.append(ch)
                    out.append(text[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    in_str = not in_str
                    out.append(ch)
                    i += 1
                    continue
                if in_str and ch in '\n\r\t':
                    out.append('\\n' if ch == '\n' else ('\\r' if ch == '\r' else '\\t'))
                    i += 1
                    continue
                out.append(ch)
                i += 1
            return ''.join(out)

        escaped = _escape_control_in_strings(stripped)
        if escaped != stripped:
            candidates.append(escaped)
            # also try escaped + no-trailing-comma combined
            candidates.append(re.sub(r',(\s*[}\]])', r'\1', escaped))

        # 5. Greedy: pull the first '{' … last '}' span and try it.
        first = stripped.find('{')
        last = stripped.rfind('}')
        if first != -1 and last > first:
            candidates.append(stripped[first: last + 1])

        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                # Unwrap the {"properties": {...}} echo some models produce.
                if (
                    'properties' in parsed
                    and isinstance(parsed['properties'], dict)
                    and len(parsed) == 1
                ):
                    parsed = parsed['properties']
                return parsed
        return None

    @staticmethod
    def _format_json_error(raw: str, exc: Exception) -> str:
        """Build a compact, location-aware hint to feed back to the model.

        ``json.JSONDecodeError`` carries ``lineno``/``colno``/``pos``; we surface
        the offending line and a few chars of context so the model can fix the
        exact spot rather than re-generating the whole payload blind.
        """
        pos = getattr(exc, 'pos', None)
        msg = getattr(exc, 'msg', str(exc))
        if pos is not None and isinstance(pos, int):
            start = max(0, pos - 60)
            snippet = raw[start: pos + 60]
            return (
                f'Your previous JSON output was invalid: {msg} '
                f'at character offset {pos}. '
                f'Context around the error: ...{snippet!r}... '
                f'Common causes: unescaped double quotes inside a string value, '
                f'bare newlines inside strings, trailing commas, or truncated output. '
                f'Return ONLY a corrected JSON object with the same schema.'
            )
        return (
            f'Your previous JSON output was invalid: {msg}. '
            f'Return ONLY a corrected JSON object with the same schema.'
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        extra_body = {}
        if os.getenv('GRAPHITI_DISABLE_REASONING') == '1':
            # A reasoning model spends the completion budget on hidden reasoning
            # first: long LoCoMo sessions then end with finish_reason=length and
            # an empty body, which no retry can fix. The benchmark harness turns
            # reasoning off for extraction (documented in the comparison report).
            extra_body['reasoning_effort'] = 'none'
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),  # type: ignore[arg-type]
                **extra_body,
            )
            _trace_usage(_TRACE_NAME.get(), response,
                         sum(len(str(m.get('content') or '')) for m in openai_messages))
            result = response.choices[0].message.content or ''
            finish_reason = response.choices[0].finish_reason
            # A length-stopped JSON response is incomplete by definition. Do
            # not salvage or re-prompt it: callers discard the whole response
            # and retry smaller input halves.
            if finish_reason == 'length':
                logger.error(
                    'LLM output truncated (model=%s, chars=%d); discarding the complete response',
                    self.model, len(result),
                )
                raise OutputTruncatedError(
                    f'LLM output reached its token limit; discarded {len(result)} response characters'
                )
            # An empty body (refusal, length finish_reason, or a flaky endpoint) would make
            # json.loads raise a cryptic JSONDecodeError; surface a clear error instead.
            if not result:
                refusal = getattr(response.choices[0].message, 'refusal', None) or ''
                logger.warning(
                    f'Empty LLM response: model={self.model}, '
                    f'finish_reason={finish_reason}, refusal={refusal!r}'
                )
                raise EmptyResponseError(
                    f'LLM returned an empty response '
                    f'(finish_reason={finish_reason}, refusal={refusal!r})'
                )
            # Many OpenAI-compatible/local models wrap JSON in a ```json fence even under a
            # structured response_format; strip it before parsing.
            stripped = self._strip_code_fences(result)
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as je:
                # The API didn't enforce the schema (json_object mode) and the
                # model produced malformed JSON. Try to salvage before giving
                # up — truncation, trailing commas, unescaped quotes/newlines.
                salvaged = self._salvage_json(result)
                if salvaged is not None:
                    logger.warning(
                        f'Salvaged malformed JSON from LLM output '
                        f'(model={self.model}, error={je.msg} @ pos {je.pos}); '
                        f'raw head: {result[:200]!r}'
                    )
                    parsed = salvaged
                else:
                    # Salvage failed — surface a rich log and let the caller
                    # (tenacity / error-feedback retry) handle re-generation.
                    logger.error(
                        f'LLM JSON parse failed (model={self.model}, '
                        f'finish_reason={finish_reason}): {je.msg} @ line {je.lineno} col {je.colno}'
                    )
                    logger.error(self._get_failed_generation_log(messages, result))
                    # Attach the raw output so the error-feedback loop in
                    # generate_response can build a precise correction hint.
                    je.raw_output = result  # type: ignore[attr-defined]
                    raise
            # Some models echo the JSON schema's "properties" wrapper instead of returning
            # a flat object (e.g. {"properties": {"field": val}} instead of {"field": val}).
            # Unwrap when "properties" is the only key and its value is a dict.
            if (
                isinstance(parsed, dict)
                and 'properties' in parsed
                and isinstance(parsed['properties'], dict)
                and len(parsed) == 1
            ):
                parsed = parsed['properties']
            return parsed
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, typing.Any]:
        _TRACE_NAME.set(str(prompt_name or ""))
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        if max_tokens is None:
            max_tokens = self.max_tokens

        # In json_object fallback mode the API does not enforce the schema, so embed it in
        # the prompt to guide the model. In json_schema mode the schema is enforced via
        # response_format, so no prompt injection is needed.
        if response_model is not None and self.structured_output_mode == 'json_object':
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            if split_retry_enabled():
                try:
                    async with self._request_semaphore:
                        return await self._generate_response_with_retry(
                            messages, response_model, max_tokens=max_tokens, model_size=model_size
                        )
                except Exception as e:
                    span.set_status('error', str(e))
                    span.record_exception(e)
                    raise

            # Non-extraction callers retain the existing JSON correction loop.
            max_json_reprompt = 2
            original_last_content = messages[-1].content
            last_exc: Exception | None = None
            for attempt in range(max_json_reprompt + 1):
                try:
                    async with self._request_semaphore:
                        return await self._generate_response_with_retry(
                            messages, response_model, max_tokens=max_tokens, model_size=model_size
                        )
                except json.JSONDecodeError as json_error:
                    last_exc = json_error
                    if attempt >= max_json_reprompt:
                        break
                    raw = getattr(json_error, 'raw_output', '') or ''
                    messages[-1].content = (
                        original_last_content + f'\n\n{self._format_json_error(raw, json_error)}'
                    )
                    logger.warning(
                        'JSON parse failed (prompt=%s, model=%s, attempt %d/%d); '
                        're-prompting with error hint.',
                        prompt_name,
                        self.model,
                        attempt + 1,
                        max_json_reprompt,
                    )
                except Exception as e:
                    span.set_status('error', str(e))
                    span.record_exception(e)
                    messages[-1].content = original_last_content
                    raise

            span.set_status('error', str(last_exc))
            if last_exc is not None:
                span.record_exception(last_exc)
            messages[-1].content = original_last_content
            assert last_exc is not None
            raise last_exc
