import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import generate
from train import load_model

model, bpe = load_model()
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
    prompt = req.prompt.strip()
    if not prompt:
        return JSONResponse({"error": "Type the start of a story."}, status_code=400)
    if len(prompt) > 500:
        return JSONResponse({"error": "Use at most 500 characters."}, status_code=400)
    return {"text": generate(model, bpe, prompt, req.temperature, req.top_k, req.top_p)}


app.mount("/", StaticFiles(directory="frontend/dist", html=True))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
