import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
import concurrent.futures
import urllib3
urllib3.connectionpool.HTTPConnectionPool.default_maxsize = 100
urllib3.connectionpool.HTTPSConnectionPool.default_maxsize = 100

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr


@register_model("openai_api")
class OpenAIAPIModel(TemplateLM):
    """
    Minimal adapter for OpenAI-like HTTP APIs.

    Supports chat-style (`/chat/completions`) and completion-style (`/completions`) endpoints.
    The class follows the shape used by other API adapters in this repo so it can be
    plugged into the existing evaluation flow.
    """

    def __init__(
        self,
        model: str = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stream: bool = False,
        tokenized_requests: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.tokenized_requests = tokenized_requests
        # number of concurrent requests to send (can be provided via model_args)
        self.num_concurrent = int(kwargs.get("num_concurrent", 1))
        self.logger = logging.getLogger("OpenAIAPIModel")

        if not self.api_key:
            self.logger.warning("OPENAI_API_KEY not set — requests will likely fail")

        # Keep a dict of last model args for compatibility/debugging
        self.model_args = {
            "model": model,
            "base_url": self.base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
        })
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _call_api(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    self.logger.warning("Rate limited; backing off %s seconds", backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                if code and 500 <= code < 600:
                    self.logger.warning("Server error %s, retrying in %s", code, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            except requests.RequestException as e:
                self.logger.warning("Request failed (attempt %s/%s): %s", attempt, self.max_retries, e)
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Exceeded max retries for API request")

    def _create_payload(
        self,
        messages: Union[List[str], List[dict], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        eos=None,
        **kwargs,
    ) -> Dict[str, Any]:
        gen_kwargs = gen_kwargs or {}
        # prefer per-request overrides
        temperature = gen_kwargs.get("temperature", self.temperature)
        max_tokens = gen_kwargs.get("max_tokens", gen_kwargs.get("max_new_tokens", self.max_tokens))

        # Determine if chat-style messages (list of dicts) or prompt strings
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            # Chat completion payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return {"endpoint": "chat/completions", "payload": payload}

        # Otherwise treat as text completion (single prompt string expected)
        prompt = messages if isinstance(messages, str) else (messages[0] if isinstance(messages, list) else "")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return {"endpoint": "completions", "payload": payload}

    def create_message(self, messages: Union[List[str], List[dict], List[JsonChatStr], JsonChatStr], generate=False):
        # Convert JsonChatStr wrappers into plain list-of-dicts expected by the API
        if isinstance(messages, list) and all(isinstance(m, JsonChatStr) for m in messages):
            return [json.loads(m.prompt) for m in messages]
        if isinstance(messages, JsonChatStr):
            return json.loads(messages.prompt)
        return messages

    @staticmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        results = []
        for out in outputs:
            # If the output is the raw response JSON from OpenAI-like API
            if not out:
                results.append("")
                continue
            if "choices" in out and len(out["choices"]) > 0:
                choice = out["choices"][0]
                # Chat-style
                if "message" in choice:
                    results.append(choice["message"].get("content", ""))
                else:
                    results.append(choice.get("text", ""))
            else:
                # fallback: try top-level text or response field
                results.append(out.get("text", out.get("response", "")) if isinstance(out, dict) else str(out))
        return results

    def model_call(self, messages: Union[List[str], List[dict], str], **kwargs) -> Optional[dict]:
        info = self._create_payload(self.create_message(messages), generate=True, gen_kwargs=kwargs.get("gen_kwargs", {}))
        # print(f"Original Messages:/n{messages}\n\n Create Messages: {self.create_message(messages)}\n\n Create Playload: {info}")
        # input('check')
        endpoint = info["endpoint"]
        payload = info["payload"]
        resp = self._call_api(endpoint, payload)
        # print(resp)
        # input('check results')
        return resp

    def generate_until(self, requests: List[Any], disable_tqdm: bool = False) -> List[str]:
        if self.tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for openai_api")

        contexts = [req.args[0] for req in requests] # prompts
        gen_kwargs_list = [req.args[1] for req in requests]

        outputs = []
        # If concurrency requested, use a thread pool to issue concurrent API calls
        if self.num_concurrent and self.num_concurrent > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_concurrent) as exc:
                futures = [
                    exc.submit(self.model_call, ctx, gen_kwargs=gkw)
                    for ctx, gkw in zip(contexts, gen_kwargs_list)
                ]
                for fut in futures:
                    try:
                        resp = fut.result()
                        outputs.append(resp)
                    except Exception as e:
                        self.logger.error("Generation failed for a request (concurrent): %s", e)
                        outputs.append({})
        else:
            for ctx, gkw in zip(contexts, gen_kwargs_list):
                try:
                    response = self.model_call(ctx, gen_kwargs=gkw)
                    # Wrap response json to align with parse_generations input
                    outputs.append(response)
                except Exception as e:
                    self.logger.error("Generation failed for a request: %s", e)
                    outputs.append({})

        return self.parse_generations(outputs)

    def apply_chat_template(self, chat_history: List[Dict[str, str]]):
        # Return a JsonChatStr so other adapters can detect it's a chat
        return JsonChatStr(json.dumps(chat_history))

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        raise NotImplementedError("Token encoding not implemented for openai_api")

    @property
    def eot_token_id(self):
        """End-of-text token id used as a prefix/sentinel for some LM APIs.

        This adapter does not use token ids directly, but TemplateLM requires
        this property. Return 0 as a sensible default placeholder.
        """
        return 0

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[tuple]:
        """Stub implementation to satisfy abstract interface.

        The OpenAI-style HTTP API does not provide token-level loglikelihoods
        in this adapter. For now raise NotImplementedError to indicate that
        exact token-level scoring isn't supported by this class.
        """
        raise NotImplementedError("_loglikelihood_tokens not implemented for openai_api")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        """Stub for rolling loglikelihood (perplexity) required by TemplateLM.

        Currently unsupported for the OpenAI HTTP adapter.
        """
        raise NotImplementedError("loglikelihood_rolling not implemented for openai_api")