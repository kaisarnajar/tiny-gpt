"""Word-level tokenizer. Builds its vocabulary from the corpus. Has no learned weights.

Vocabulary layout:
    0: <pad>   filler so every sentence in a batch has the same length
    1: <eos>   end of sentence
    2...     : every word in the corpus, sorted alphabetically
"""

from pathlib import Path

PAD = "<pad>"
EOS = "<eos>"


class Tokenizer:
    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)  # ID -> text
        self.stoi = {w: i for i, w in enumerate(self.itos)}  # text -> ID
        self.pad_id = self.stoi[PAD]
        self.eos_id = self.stoi[EOS]

    @classmethod
    def from_corpus(cls, path: str | Path) -> "Tokenizer":
        words = set(Path(path).read_text().split())
        return cls([PAD, EOS] + sorted(words))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        unknown = [w for w in text.split() if w not in self.stoi]
        if unknown:
            raise ValueError(f"Words not in the vocabulary: {unknown}")
        ids = [self.stoi[w] for w in text.split()]
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special = {self.pad_id, self.eos_id}
        return " ".join(self.itos[i] for i in ids if not (skip_special and i in special))
