import os
import time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


class LLMClient:
    """
    Clean LLM wrapper. Handles retries and rate limits automatically.
    """

    def __init__(self, model_name="llama-3.3-70b-versatile",
                 sleep_between_calls=10):
        self.model_name = model_name
        self.sleep_between_calls = sleep_between_calls
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.total_calls = 0

    def predict(self, prompt, system="You are an expert in combinatorial optimization. Be concise and precise."):
        """
        Make one LLM call. Retries once on rate limit.
        Returns response string.
        """
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                self.total_calls += 1
                time.sleep(self.sleep_between_calls)
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM call failed (attempt {attempt+1}): {e}")
                time.sleep(30)
        return ""

    def reset_counter(self):
        self.total_calls = 0

    def get_call_count(self):
        return self.total_calls
