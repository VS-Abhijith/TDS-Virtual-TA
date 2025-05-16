import base64
import io
import json
import os
from typing import Optional

import faiss
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
# Load FAISS index and metadata at startup
index = faiss.read_index("faiss.index")
with open("chunk_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN environment variable not set.")

EMBED_URL = "https://aipipe.org/openai/v1/embeddings"
LLM_URL = "https://aipipe.org/openai/v1/chat/completions"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
TOP_K = 5


class Query(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image string


def ocr_from_base64(base64_str: str) -> str:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {e}")


def get_embedding(text: str) -> np.ndarray:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
    }
    payload = {"model": EMBED_MODEL, "input": text}
    response = requests.post(EMBED_URL, headers=headers, json=payload, timeout=15)
    if response.status_code == 200:
        return np.array(response.json()["data"][0]["embedding"], dtype="float32")
    else:
        raise RuntimeError(f"Embedding failed: {response.text}")


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a helpful virtual TA for the Tools in Data Science course.

Here is the student question:
{question}

Use the following course/forum content to answer:
{context}

Answer clearly and concisely. Provide links or file names if available.
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for a course Q&A system."},
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(LLM_URL, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"LLM generation failed: {response.text}")


@app.post("/api/")
async def answer_query(query: Query):
    # Combine question + OCR text (if image provided)
    ocr_text = ""
    if query.image:
        try:
            ocr_text = ocr_from_base64(query.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    combined_text = query.question
    if ocr_text.strip():
        combined_text += "\n\n" + ocr_text

    # Embed combined text and query FAISS
    query_vec = get_embedding(combined_text).reshape(1, -1)
    distances, indices = index.search(query_vec, TOP_K)
    retrieved_chunks = [metadata[i]["text"] for i in indices[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Generate answer
    answer = generate_answer(query.question, context)

    # Prepare source links for response
    source_links = []
    for i in indices[0]:
        md = metadata[i]
        url = md.get("source") or ""
        if url and not url.startswith("http"):
            url = ""  # sanitize if not a full URL
        if url:
            source_links.append({"url": url, "text": md.get("title", "") or "Source"})

    return {"answer": answer, "links": source_links}
