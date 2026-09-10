"""Groq LLM wrapper: multi-key rotation, token logging, call counting, mock mode."""

import os
import time

import config


class AllKeysExhausted(Exception):
    """Raised when every GROQ_API_KEY_N has been rate-limited/quota-exhausted."""


def _load_keys():
    keys = []
    i = 1
    while True:
        key = os.environ.get(f"GROQ_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    return keys


class LLMClient:
    """Real Groq client. Same interface as MockLLMClient."""

    DEFAULT_SYSTEM = "You are an expert in combinatorial optimization. Be concise and precise."

    def __init__(self, model=config.MODEL, sleep_between_calls=config.SLEEP_BETWEEN_CALLS):
        from dotenv import load_dotenv
        from groq import Groq

        load_dotenv()
        self._Groq = Groq
        self.keys = _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No GROQ_API_KEY_1..N found in environment. Copy .env.example to .env and fill in your keys."
            )
        self.key_idx = 0
        self.client = self._Groq(api_key=self.keys[self.key_idx])
        self.model = model
        self.sleep_between_calls = sleep_between_calls
        self._calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset(self):
        self._calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def count(self):
        return self._calls

    def _rotate_key(self):
        self.key_idx += 1
        if self.key_idx >= len(self.keys):
            raise AllKeysExhausted("All Groq API keys are rate-limited/exhausted.")
        self.client = self._Groq(api_key=self.keys[self.key_idx])

    def predict(self, prompt, system=None, temperature=config.TEMPERATURE):
        system = system or self.DEFAULT_SYSTEM
        backoff_attempts = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                self._calls += 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.prompt_tokens += usage.prompt_tokens or 0
                    self.completion_tokens += usage.completion_tokens or 0
                time.sleep(self.sleep_between_calls)
                return response.choices[0].message.content or ""
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower() or "quota" in msg.lower()
                if not is_rate_limit:
                    raise
                backoff_attempts += 1
                if backoff_attempts <= 3:
                    time.sleep(min(60, 5 * (2 ** backoff_attempts)))
                    continue
                self._rotate_key()
                backoff_attempts = 0


class MockLLMClient:
    """Zero-token stand-in with the same interface. Powers --dry-run and offline tests.

    The canned response is deliberately dual-purpose: its first lines parse as
    valid method names (for Phase-1 exploration), and its trailing bracketed
    list parses as a valid route/item-selection (for Resolve/Refine) - one
    fixed string satisfies every phase's parser without knowing which phase
    called it.
    """

    CANNED_RESPONSE = (
        "Nearest Neighbor\n"
        "2-Opt\n"
        "Greedy Algorithm\n"
        "Simulated Annealing\n"
        "Step 1: build an initial solution\n"
        "Step 2: improve it\n"
        "Step 3: finalize\n"
        "Final answer: [0, 1, 2, 3, 4, 0]"
    )

    def __init__(self, *args, **kwargs):
        self._calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset(self):
        self._calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def count(self):
        return self._calls

    def predict(self, prompt, system=None, temperature=None):
        self._calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 10
        return self.CANNED_RESPONSE
