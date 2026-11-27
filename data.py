import os
import urllib.request

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(URL, "data/stories.txt")
    text = open("data/stories.txt").read()
    print(f"data/stories.txt: {len(text):,} characters, {text.count('<|endoftext|>'):,} stories")
