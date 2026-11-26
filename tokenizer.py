class Tokenizer:
    def __init__(self, words):
        self.itos = words
        self.stoi = {w: i for i, w in enumerate(words)}
        self.pad_id = 0
        self.eos_id = 1
        self.vocab_size = len(words)

    @classmethod
    def from_file(cls, path):
        words = sorted(set(open(path).read().split()))
        return cls(["<pad>", "<eos>"] + words)

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if i > 1)
