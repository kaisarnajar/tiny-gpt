"""Web server: type the start of a sentence in the browser, get the full sentence back.

    cd frontend && npm install && npm run build && cd ..
    python server.py            # -> http://localhost:8000

API:
    POST /api/complete  {"prompt": "The Dog"}
      200 {"sentence": "the dog barks loudly ."}
      400 {"error": "Unknown word: \"unicorn\""}
"""

import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import generate
from train import MODEL_PATH, load_model

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

if not Path(MODEL_PATH).exists():
    raise SystemExit(f"{MODEL_PATH} not found. Run `python train.py` first.")
model, tokenizer = load_model()  # loaded once, reused for every request

app = FastAPI(title="tiny-gpt")


class CompleteRequest(BaseModel):
    prompt: str


def error(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": message})


@app.post("/api/complete")
def complete(req: CompleteRequest):
    # "The Dog." -> ["the", "dog", "."]  (lowercase, punctuation as its own word)
    words = re.findall(r"[a-z]+|[^\sa-z]", req.prompt.lower())

    if not words:
        return error("Type the start of a sentence, e.g. \"The Dog\".")
    unknown = [w for w in words if w not in tokenizer.stoi]
    if unknown:
        label = "Unknown word" if len(unknown) == 1 else "Unknown words"
        return error(f"{label}: " + ", ".join(f'"{w}"' for w in unknown))
    if len(words) >= model.context_len:
        return error(f"Prompt is too long: use at most {model.context_len - 1} words.")

    return {"sentence": generate(model, tokenizer, " ".join(words), verbose=False)}


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def frontend_missing():
        return PlainTextResponse("Frontend not built. Run: cd frontend && npm install && npm run build")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
