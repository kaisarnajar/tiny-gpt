"""Word-level tokenizer with a fixed vocabulary. Has no learned weights."""


class Tokenizer:
    def __init__(self):
        self.vocab = {"He": 0, "She": 1, "boy": 2, "girl": 3}  # Text -> ID
        self.id_to_word = {i: w for w, i in self.vocab.items()}  # ID -> Text

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab[word] for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_word[i] for i in ids)
