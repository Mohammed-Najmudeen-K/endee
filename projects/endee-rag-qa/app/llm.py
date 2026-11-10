import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError

load_dotenv()
_client = None

def generate(prompt, system):
    global _client
    if _client is None:
        _client = genai.Client()
    models = [os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
              os.getenv("GEMINI_FALLBACK", "gemini-2.5-flash-lite")]
    last = None
    for model in models:
        for attempt in range(3):
            try:
                r = _client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system, temperature=0.2))
                return r.text
            except ServerError as e:
                last = e
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise last