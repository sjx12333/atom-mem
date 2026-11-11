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

from contextvars import ContextVar
from functools import wraps


_split_retry_enabled: ContextVar[bool] = ContextVar('split_retry_enabled', default=False)


def split_retry_enabled() -> bool:
    return _split_retry_enabled.get()


def with_split_retry_context(func):
    """Mark LLM calls in an async extraction operation for caller bisection."""
    @wraps(func)
    async def wrapped(*args, **kwargs):
        token = _split_retry_enabled.set(True)
        try:
            return await func(*args, **kwargs)
        finally:
            _split_retry_enabled.reset(token)

    return wrapped


class RateLimitError(Exception):
    """Exception raised when the rate limit is exceeded."""

    def __init__(self, message='Rate limit exceeded. Please try again later.'):
        self.message = message
        super().__init__(self.message)


class RefusalError(Exception):
    """Exception raised when the LLM refuses to generate a response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """Exception raised when the LLM returns an empty response."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class OutputTruncatedError(Exception):
    """The provider stopped because the output token limit was reached."""

    code = 'LLM_OUTPUT_TRUNCATED'

    def __init__(self, message='LLM output was truncated by the token limit.'):
        self.message = message
        super().__init__(self.message)
