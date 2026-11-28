import heapq
import json
import re
import sys
from collections import Counter, defaultdict

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
        pairs, where = Counter(), defaultdict(set)
        for i, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pairs[pair] += counts[i]
                where[pair].add(i)
        heap = [(-n, pair) for pair, n in pairs.items()]
        heapq.heapify(heap)
        while len(self.vocab) < vocab_size - 1 and heap:
            n, best = heapq.heappop(heap)
            if -n != pairs[best] or n == 0:
                continue
            self.add(*best)
            new_id = self.merges[best]
            for i in where.pop(best):
                old_ids = words[i]
                new_ids = merge(old_ids, best, new_id)
                if new_ids == old_ids:
                    continue
                for pair in zip(old_ids, old_ids[1:]):
                    pairs[pair] -= counts[i]
                for pair in zip(new_ids, new_ids[1:]):
                    pairs[pair] += counts[i]
                    where[pair].add(i)
                for pair in set(zip(old_ids, old_ids[1:])) | set(zip(new_ids, new_ids[1:])):
                    if pairs[pair] > 0:
                        heapq.heappush(heap, (-pairs[pair], pair))
                words[i] = new_ids
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
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    text = open("data/train.txt").read(50_000_000)
    bpe = BPE()
    bpe.train(text, vocab_size)
    bpe.save("bpe.json")
    print(f"trained on {len(text):,} characters, vocab size {bpe.vocab_size}")
    for sample in ["Once upon a time, there was a little girl named Lily.",
                   "The water cycle describes how water moves.", "def add(a, b):\n    return a + b"]:
        print(repr(sample), "->", [bpe.decode([i]) for i in bpe.encode(sample)])
