from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import time

app = FastAPI(title="AI SaaS API", version="1.0.0")

OLLAMA_URL = "http://ollama.ollama.svc.cluster.local:11434"
DEFAULT_MODEL = "qwen2:1.5b"

class ChatRequest(BaseModel):
    message: str
    model: str = DEFAULT_MODEL

class SummarizeRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL

class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "Korean"
    model: str = DEFAULT_MODEL

class APIResponse(BaseModel):
    result: str
    model: str
    elapsed_ms: int

async def ask_ollama(model: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        resp.raise_for_status()
        return resp.json()["response"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=APIResponse)
async def chat(req: ChatRequest):
    start = time.time()
    result = await ask_ollama(req.model, req.message)
    elapsed = int((time.time() - start) * 1000)
    return APIResponse(result=result, model=req.model, elapsed_ms=elapsed)

@app.post("/api/summarize", response_model=APIResponse)
async def summarize(req: SummarizeRequest):
    start = time.time()
    prompt = f"Summarize the following text concisely:\n\n{req.text}"
    result = await ask_ollama(req.model, prompt)
    elapsed = int((time.time() - start) * 1000)
    return APIResponse(result=result, model=req.model, elapsed_ms=elapsed)

@app.post("/api/translate", response_model=APIResponse)
async def translate(req: TranslateRequest):
    start = time.time()
    prompt = f"Translate the following text to {req.target_lang}. Only output the translation, nothing else:\n\n{req.text}"
    result = await ask_ollama(req.model, prompt)
    elapsed = int((time.time() - start) * 1000)
    return APIResponse(result=result, model=req.model, elapsed_ms=elapsed)
