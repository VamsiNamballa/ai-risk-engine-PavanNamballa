import json
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"


SYSTEM_PROMPT = (
    "You are an expert financial analyst. "
    "You answer any question about a company or document — investment advice, risk, performance, outlook, strategy, or comparisons. "
    "Always respond with valid JSON only — no markdown, no explanation outside the JSON."
)

PROMPT_TEMPLATE = """Read the financial document excerpts below and answer the question thoroughly.

Document Content:
{context}

Question: {query}

Reply ONLY with this JSON (no extra text, no markdown):
{{
  "executive_summary": "3-4 sentences directly answering the question with specific numbers and facts from the document.",
  "key_points": ["specific finding 1", "specific finding 2", "specific finding 3"],
  "recommendation": "A clear, direct recommendation based on the question asked. If investment-related say whether to invest or not. If risk-related state the risk. Be specific.",
  "verdict": "Positive",
  "score": 50,
  "confidence": "Medium"
}}

Rules:
- verdict must be exactly one of: Positive, Neutral, Negative
  * Positive = good news, good to invest, low risk, strong performance, healthy financials
  * Neutral = mixed signals, moderate risk, uncertain outlook
  * Negative = bad news, avoid investing, high risk, poor performance, financial distress
- score is an integer 0-100 representing overall health/attractiveness (100 = excellent, 0 = very poor)
  * For investment questions: 100 = strong buy, 0 = strong sell
  * For risk questions: invert it — high risk = low score
  * For performance questions: 100 = excellent, 0 = very poor
- confidence must be exactly one of: High, Medium, Low
- Use real numbers and facts from the document in your answers
- Do NOT use placeholder text"""


def _empty_response():
    return {
        "executive_summary": "No relevant information was found in the uploaded documents to answer this question.",
        "key_points": [],
        "recommendation": "Please upload a document containing relevant information and try again.",
        "verdict": "Neutral",
        "score": None,
        "confidence": "Low",
    }


def _generate_groq(prompt):
    for attempt in range(3):
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=15,
            )
            if not response.ok:
                raise RuntimeError(f"Groq {response.status_code}: {response.text}")
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Groq generation failed after 3 attempts: {e}")
            time.sleep(2 ** attempt)


def _generate_gemini(prompt):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
                contents=prompt,
            )
            return json.loads(response.text)
        except Exception:
            if attempt == 2:
                raise
            # retry on rate limit or transient errors
            time.sleep(2 ** attempt)



def active_model():
    if GROQ_API_KEY:
        return f"Groq · {GROQ_MODEL}"
    if GOOGLE_API_KEY:
        return f"Gemini · {GEMINI_MODEL}"
    return "No API key configured"


def generate_response(query, chunks):
    if not chunks:
        return _empty_response()

    context = "\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    if GROQ_API_KEY:
        return _generate_groq(prompt)
    if GOOGLE_API_KEY:
        return _generate_gemini(prompt)
    raise RuntimeError("No API key configured. Set GOOGLE_API_KEY or GROQ_API_KEY as an environment variable.")
