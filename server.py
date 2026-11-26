import re

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import generate
from train import load_model

model, tokenizer = load_model()
app = FastAPI()


class Request(BaseModel):
    prompt: str
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9


@app.get("/api/info")
def info():
    layers = [{"name": name, "shape": " × ".join(map(str, p.shape)), "params": p.numel()}
              for name, p in model.named_parameters()]
    return {"layers": layers, "total": sum(layer["params"] for layer in layers)}


@app.post("/api/complete")
def complete(req: Request):
    words = re.findall(r"[a-z]+|[^\sa-z]", req.prompt.lower())
    unknown = [w for w in words if w not in tokenizer.stoi]
    if not words:
        return JSONResponse({"error": "Type the start of a sentence."}, status_code=400)
    if unknown:
        return JSONResponse({"error": "Unknown word: " + ", ".join(unknown)}, status_code=400)
    if len(words) >= model.context_len:
        return JSONResponse({"error": f"Use at most {model.context_len - 1} words."}, status_code=400)
    sentence = generate(model, tokenizer, " ".join(words), req.temperature, req.top_k, req.top_p)
    return {"sentence": sentence}


app.mount("/", StaticFiles(directory="frontend/dist", html=True))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
