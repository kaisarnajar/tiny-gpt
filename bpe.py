import json
import re
import sys
from collections import Counter

EOS = "<|endoftext|>"
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
        self.eos_id = len(self.vocab)
        self.cache = {}

    def add(self, a, b):
        new_id = 256 + len(self.merges)
        self.merges[(a, b)] = new_id
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def train(self, text, vocab_size):
        chunks = Counter(SPLIT.findall(text.replace(EOS, "")))
        words = [list(chunk.encode("utf-8")) for chunk in chunks]
        counts = list(chunks.values())
        while len(self.vocab) < vocab_size - 1:
            pairs = Counter()
            for ids, n in zip(words, counts):
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += n
            best = max(pairs, key=pairs.get)
            self.add(*best)
            words = [merge(ids, best, self.merges[best]) for ids in words]
        self.eos_id = len(self.vocab)

    def encode_chunk(self, chunk):
        if chunk not in self.cache:
            ids = list(chunk.encode("utf-8"))
            while len(ids) > 1:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            self.cache[chunk] = ids
        return self.cache[chunk]

    def encode(self, text):
        out = []
        for i, part in enumerate(text.split(EOS)):
            if i > 0:
                out.append(self.eos_id)
            for chunk in SPLIT.findall(part):
                out += self.encode_chunk(chunk)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids if i != self.eos_id).decode("utf-8", errors="replace")

    def save(self, path):
        json.dump(list(self.merges), open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


if __name__ == "__main__":
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    text = open("data/val.txt").read()
    bpe = BPE()
    bpe.train(text, vocab_size)
    bpe.save("bpe.json")
    print(f"trained on {len(text):,} characters, vocab size {bpe.vocab_size}")
    sample = "Once upon a time, there was a little girl named Lily."
    print(sample, "->", [bpe.decode([i]) for i in bpe.encode(sample)])
