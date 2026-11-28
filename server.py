import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import stream
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
    return {"parameters": sum(p.numel() for p in model.parameters())}


@app.post("/api/complete")
def complete(req: Request):
    prompt = req.prompt.strip()
    if not prompt:
        return JSONResponse({"error": "Type the start of a story."}, status_code=400)
    if len(prompt) > 500:
        return JSONResponse({"error": "Use at most 500 characters."}, status_code=400)
    pieces = stream(model, bpe, prompt, req.temperature, req.top_k, req.top_p)
    events = (f"data: {json.dumps({'text': piece})}\n\n" for piece in pieces)
    return StreamingResponse(events, media_type="text/event-stream")


app.mount("/", StaticFiles(directory="frontend/dist", html=True))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
