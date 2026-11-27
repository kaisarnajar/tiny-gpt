import os
import urllib.request

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_BYTES = 500 * 1024 * 1024
EOS = "<|endoftext|>"


def download(url, path, max_bytes=None):
    request = urllib.request.Request(url)
    if max_bytes:
        request.add_header("Range", f"bytes=0-{max_bytes - 1}")
    data = urllib.request.urlopen(request).read().decode("utf-8", errors="ignore")
    data = data[:data.rfind(EOS) + len(EOS)]
    open(path, "w").write(data)
    print(f"{path}: {len(data):,} characters, {data.count(EOS):,} stories")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download(BASE + "TinyStoriesV2-GPT4-train.txt", "data/train.txt", TRAIN_BYTES)
    download(BASE + "TinyStoriesV2-GPT4-valid.txt", "data/val.txt")
