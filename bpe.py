import json
import os
import re
import sys
from collections import Counter

SPLIT = re.compile(r" ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z_0-9]+|\s+(?!\S)|\s+")


def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self, merges=()):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b in merges:
            self.add(a, b)

    def add(self, a, b):
        new_id = 256 + len(self.merges)
        self.merges[(a, b)] = new_id
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        return new_id

    def train(self, text, vocab_size):
        chunks = Counter(SPLIT.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunks]
        counts = list(chunks.values())
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for ids, n in zip(words, counts):
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += n
            best = max(pairs, key=pairs.get)
            new_id = self.add(*best)
            words = [merge(ids, best, new_id) for ids in words]

    def encode(self, text):
        out = []
        for chunk in SPLIT.findall(text):
            ids = list(chunk.encode("utf-8"))
            while len(ids) > 1:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            out += ids
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        json.dump(list(self.merges), open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


def python_code():
    folder = os.path.dirname(os.__file__)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".py"))
    return "\n".join(open(os.path.join(folder, f), encoding="utf-8", errors="replace").read() for f in files)


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    text = python_code()
    bpe = BPE()
    bpe.train(text, vocab_size)
    bpe.save("bpe.json")
    print(f"trained on {len(text):,} characters, vocab size {len(bpe.vocab)}")

    for sample in ["def hello_world():\n    print('hello')", "for i in range(10):", "the cow eats grass ."]:
        ids = bpe.encode(sample)
        pieces = [bpe.decode([i]) for i in ids]
        print(f"\n{sample!r}\n  {len(sample)} characters -> {len(ids)} tokens: {pieces}")
