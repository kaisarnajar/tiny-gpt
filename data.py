import random
import sys
from pathlib import Path

SEED = 42
NUM_SENTENCES = 50_000
MAX_WORDS = 7
OUT_PATH = Path("data/corpus.txt")

ANIMALS = {
    "cat": {
        "sound": "meows",
        "eats": ["fish", "milk", "its food"],
        "chases": ["the mouse", "the bird", "a ball"],
        "is": ["lazy", "curious", "playful", "soft"],
        "lives": ["in the house"],
        "sleeps": ["on the sofa", "on the bed", "in the sun"],
        "does": ["climbs the tree", "drinks some milk"],
    },
    "dog": {
        "sound": "barks",
        "eats": ["meat", "a bone", "its food"],
        "chases": ["the cat", "the ball", "a stick"],
        "is": ["loyal", "friendly", "happy", "playful"],
        "lives": ["in the house", "in a kennel"],
        "sleeps": ["on the rug", "in its bed", "by the door"],
        "does": ["runs in the park", "guards the house", "wags its tail"],
    },
    "cow": {
        "sound": "moos",
        "eats": ["grass", "hay"],
        "is": ["calm", "big", "gentle"],
        "lives": ["on the farm"],
        "sleeps": ["in the barn"],
        "does": ["gives us milk", "stands in the field"],
    },
    "horse": {
        "sound": "neighs",
        "eats": ["hay", "grass", "an apple", "a carrot"],
        "is": ["fast", "strong", "tall"],
        "lives": ["on the farm", "in the stable"],
        "sleeps": ["in the stable"],
        "does": ["runs in the field", "pulls the cart"],
    },
    "sheep": {
        "sound": "bleats",
        "eats": ["grass"],
        "is": ["soft", "woolly", "gentle"],
        "lives": ["on the farm", "on the hill"],
        "sleeps": ["in the barn"],
        "does": ["follows the flock"],
    },
    "duck": {
        "sound": "quacks",
        "eats": ["bread", "seeds", "small fish"],
        "is": ["noisy", "happy"],
        "lives": ["by the pond", "on the farm"],
        "does": ["swims in the pond"],
    },
    "bird": {
        "sound": "sings",
        "eats": ["seeds", "worms", "berries"],
        "is": ["small", "colorful"],
        "lives": ["in the tree", "in a nest"],
        "sleeps": ["in its nest"],
        "does": ["flies over the trees", "builds a nest"],
    },
    "owl": {
        "sound": "hoots",
        "eats": ["mice", "insects"],
        "is": ["wise", "quiet"],
        "lives": ["in the tree", "in the forest"],
        "sleeps": ["during the day"],
        "does": ["hunts at night", "flies at night"],
    },
    "mouse": {
        "sound": "squeaks",
        "eats": ["cheese", "seeds", "bread"],
        "is": ["small", "tiny", "quick"],
        "lives": ["in a hole", "in the house"],
        "sleeps": ["in its hole"],
        "does": ["hides from the cat"],
    },
    "rabbit": {
        "eats": ["carrots", "grass", "lettuce"],
        "is": ["fast", "soft", "shy"],
        "lives": ["in a burrow", "in the garden"],
        "sleeps": ["in its burrow"],
        "does": ["hops in the garden", "digs a hole"],
    },
    "fox": {
        "eats": ["rabbits", "berries", "mice"],
        "chases": ["the rabbit", "the chickens"],
        "is": ["clever", "quick", "sly"],
        "lives": ["in the forest", "in a den"],
        "sleeps": ["in its den"],
        "does": ["hunts at night"],
    },
    "lion": {
        "sound": "roars",
        "eats": ["meat", "zebras"],
        "chases": ["the zebra", "the deer"],
        "is": ["strong", "brave", "fierce"],
        "lives": ["in africa", "in the wild"],
        "sleeps": ["in the shade", "all day"],
        "does": ["hunts in a group"],
    },
    "tiger": {
        "sound": "roars",
        "eats": ["meat", "deer"],
        "chases": ["the deer"],
        "is": ["strong", "fierce", "striped"],
        "lives": ["in the jungle", "in the forest"],
        "sleeps": ["in the shade"],
        "does": ["swims in the river", "hunts alone"],
    },
    "elephant": {
        "sound": "trumpets",
        "eats": ["leaves", "grass", "fruit"],
        "is": ["big", "heavy", "gentle", "clever"],
        "lives": ["in africa", "in the jungle"],
        "does": ["drinks from the river", "sprays water"],
    },
    "monkey": {
        "eats": ["bananas", "fruit", "nuts"],
        "is": ["playful", "clever", "noisy"],
        "lives": ["in the jungle", "in the trees"],
        "sleeps": ["in the trees"],
        "does": ["climbs the tree", "swings from the branches"],
    },
    "turtle": {
        "eats": ["leaves", "lettuce", "worms"],
        "is": ["slow", "old", "calm"],
        "lives": ["in the pond", "near the sea"],
        "sleeps": ["in its shell"],
        "does": ["walks very slowly", "swims in the sea"],
    },
}

SOUND_EXTRAS = ["", "loudly", "at night", "all day"]

PEOPLE_ACTIVITIES = [
    "reads a book", "drinks water", "drinks tea", "eats an apple", "eats bread",
    "walks the dog", "feeds the cat", "rides a bike", "writes a letter",
    "cooks dinner", "sings a song", "opens the door", "cleans the room",
    "plants a tree",
]
CHILD_ACTIVITIES = ["goes to school", "plays football", "does homework", "plays with the dog"]
ADULT_ACTIVITIES = ["goes to work", "drinks coffee", "drives the car", "reads the newspaper"]

PEOPLE = {
    "the boy": CHILD_ACTIVITIES,
    "the girl": CHILD_ACTIVITIES,
    "my brother": CHILD_ACTIVITIES,
    "my sister": CHILD_ACTIVITIES,
    "the man": ADULT_ACTIVITIES,
    "the woman": ADULT_ACTIVITIES,
    "my father": ADULT_ACTIVITIES,
    "my mother": ADULT_ACTIVITIES,
    "the farmer": ["feeds the cows", "milks the cow", "drives the tractor", "plants seeds"],
    "the teacher": ["teaches the class", "writes on the board", "reads a story"],
}

IDENTITIES = {
    "he": ["a boy", "a man", "my brother", "my father", "my friend"],
    "she": ["a girl", "a woman", "my sister", "my mother", "my friend"],
}
JOBS = ["a doctor", "a teacher", "a farmer", "a student", "a singer"]
PERSON_ADJECTIVES = ["kind", "tall", "happy", "tired", "hungry", "busy"]


def animal_sentence(rng):
    name = rng.choice(list(ANIMALS))
    facts = ANIMALS[name]
    relation = rng.choice(list(facts))

    if relation == "sound":
        extra = rng.choice(SOUND_EXTRAS)
        return f"the {name} {facts['sound']} {extra} .".replace("  ", " ")
    if relation == "is":
        very = "very " if rng.random() < 0.3 else ""
        return f"the {name} is {very}{rng.choice(facts['is'])} ."
    if relation == "does":
        return f"the {name} {rng.choice(facts['does'])} ."
    return f"the {name} {relation} {rng.choice(facts[relation])} ."


def person_sentence(rng):
    subject = rng.choice(list(PEOPLE))
    if subject in ("the farmer", "the teacher") or rng.random() < 0.5:
        activity = rng.choice(PEOPLE[subject])
    else:
        activity = rng.choice(PEOPLE_ACTIVITIES)
    return f"{subject} {activity} ."


def identity_sentence(rng):
    pronoun = rng.choice(list(IDENTITIES))
    r = rng.random()
    if r < 0.4:
        return f"{pronoun} is {rng.choice(IDENTITIES[pronoun])} ."
    if r < 0.7:
        return f"{pronoun} is {rng.choice(JOBS)} ."
    return f"{pronoun} is {rng.choice(PERSON_ADJECTIVES)} ."


TEMPLATES = [(animal_sentence, 0.65), (person_sentence, 0.25), (identity_sentence, 0.10)]


def generate(n, seed=SEED):
    rng = random.Random(seed)
    builders, weights = zip(*TEMPLATES)
    sentences = [rng.choices(builders, weights)[0](rng) for _ in range(n)]
    too_long = [s for s in sentences if len(s.split()) > MAX_WORDS]
    assert not too_long, f"Sentences longer than {MAX_WORDS} words: {too_long[:3]}"
    return sentences


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SENTENCES
    sentences = generate(n)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sentences) + "\n")
    print(f"Wrote {len(sentences):,} sentences to {OUT_PATH}")
