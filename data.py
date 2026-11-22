"""Generate a synthetic training corpus for Step 2 (embeddings).

Sentences are built from templates so that:
  * words of the same category (animals, foods, ...) appear in the same contexts,
    so their embeddings should end up close together;
  * the SAME token means different things at different POSITIONS:
      "the" at position 0 -> next word is an animal
      "the" at position 3 -> next word is a food / object
    Only a positional embedding can tell these apart.

All text is lowercase and words are separated by spaces ("." is its own word).
The longest sentence is 6 words (+ <eos> = 7 tokens), which fits context_len = 8.

Usage:
    python data.py            # 50,000 sentences -> data/corpus.txt
    python data.py 100000     # custom number of sentences
"""

import random
import sys
from collections import Counter
from pathlib import Path

SEED = 42
NUM_SENTENCES = 50_000
OUT_PATH = Path("data/corpus.txt")

ANIMALS = ["cat", "dog", "bird", "cow", "horse", "lion", "tiger", "mouse", "fox", "rabbit"]
FOODS = ["fish", "apple", "bone", "cake", "bread", "milk", "grass", "cheese"]
OBJECTS = ["ball", "toy", "stick", "box", "rope", "hat"]
EAT_VERBS = ["eats", "likes", "wants"]  # followed by a food
PLAY_VERBS = ["chases", "finds", "sees"]  # followed by an object
ADJECTIVES = ["big", "small", "happy", "hungry", "fast", "slow"]
MALE = ["boy", "man", "king", "prince"]
FEMALE = ["girl", "woman", "queen", "princess"]


def svo_sentence(rng: random.Random) -> str:
    # the {animal} {verb} the {food/object} .     e.g. "the cat eats the fish ."
    animal = rng.choice(ANIMALS)
    if rng.random() < 0.5:
        verb, thing = rng.choice(EAT_VERBS), rng.choice(FOODS)
    else:
        verb, thing = rng.choice(PLAY_VERBS), rng.choice(OBJECTS)
    return f"the {animal} {verb} the {thing} ."


def adjective_sentence(rng: random.Random) -> str:
    # the {animal} is {adjective} .               e.g. "the dog is happy ."
    return f"the {rng.choice(ANIMALS)} is {rng.choice(ADJECTIVES)} ."


def pronoun_sentence(rng: random.Random) -> str:
    # he is a {male noun} . / she is a {female noun} .   (continuity with Step 1)
    if rng.random() < 0.5:
        return f"he is a {rng.choice(MALE)} ."
    return f"she is a {rng.choice(FEMALE)} ."


# (generator, weight): how often each template is used
TEMPLATES = [(svo_sentence, 0.6), (adjective_sentence, 0.2), (pronoun_sentence, 0.2)]


def generate(n: int, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    funcs, weights = zip(*TEMPLATES)
    return [rng.choices(funcs, weights)[0](rng) for _ in range(n)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SENTENCES
    sentences = generate(n)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sentences) + "\n")

    words = [w for s in sentences for w in s.split()]
    counts = Counter(words)
    lengths = Counter(len(s.split()) for s in sentences)

    print(f"Wrote {len(sentences):,} sentences to {OUT_PATH}")
    print(f"Unique sentences : {len(set(sentences)):,}")
    print(f"Total words      : {len(words):,}")
    print(f"Vocabulary size  : {len(counts)} words (+ <pad>, <eos> added by the tokenizer)")
    print(f"Sentence lengths : {dict(sorted(lengths.items()))}  (max + <eos> = {max(lengths) + 1} tokens)")
    print(f"Most common      : {counts.most_common(8)}")
    print("\nSamples:")
    for s in sentences[:8]:
        print("  " + s)


if __name__ == "__main__":
    main()
