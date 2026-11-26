"""Look inside the trained model: embeddings (Step 2) and attention (Step 3).

    python train.py      # first, creates model.pt
    python explore.py    # prints 8 sections, saves embeddings.png and attention.png

1. Shapes            what the embedding tables look like
2. Softmax           raw scores -> probabilities (manual vs PyTorch)
3. Cross-entropy     -log(p_true), position by position (manual vs PyTorch)
4. Similarity        nearest words by cosine similarity
5. 2D picture        PCA of the word embeddings -> embeddings.png
6. Position matters  Step 2 model trained with vs without positional embeddings
7. Attention fixes   he/she and cow/monkey: Step 2 model vs Step 3 model
8. Attention map     who looks at whom -> attention.png
"""

import math

import matplotlib

matplotlib.use("Agg")  # write the PNG without opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import ANIMALS, PERSON_ADJECTIVES
from model import cross_entropy_manual, softmax_manual
from train import CONTEXT_LEN, load_model, train

EXPLORE_EPOCHS = 10  # for the with/without-position comparison (fast)
PLOT_PATH = "embeddings.png"
ATTN_PATH = "attention.png"

CATEGORIES = {
    "animals": list(ANIMALS),
    "people": ["boy", "girl", "man", "woman", "mother", "father", "brother", "sister",
               "farmer", "teacher", "doctor", "student", "singer", "friend"],
    "food & drink": ["fish", "milk", "meat", "bone", "grass", "hay", "apple", "carrot", "carrots", "seeds",
                     "bread", "worms", "berries", "insects", "cheese", "lettuce", "leaves", "fruit",
                     "bananas", "nuts", "coffee", "tea", "water"],
    "places": ["house", "kennel", "farm", "stable", "barn", "hill", "pond", "nest", "forest", "hole",
               "burrow", "garden", "den", "africa", "wild", "jungle", "trees", "sea", "river", "park",
               "field", "sofa", "bed", "rug", "shade", "school", "work"],
    "adjectives": sorted({a for f in ANIMALS.values() for a in f["is"]} | set(PERSON_ADJECTIVES)),
    "verbs": ["meows", "barks", "moos", "neighs", "bleats", "quacks", "sings", "hoots", "squeaks", "roars",
              "trumpets", "eats", "chases", "lives", "sleeps", "climbs", "runs", "swims", "flies", "hunts",
              "hops", "digs", "reads", "drinks", "walks", "feeds", "rides", "writes", "cooks", "plays",
              "goes", "drives", "teaches"],
}


def header(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def top_words(model, tokenizer, prompt: str, k: int = 3) -> str:
    """Top-k next words after the LAST word of the prompt."""
    with torch.no_grad():
        logits = model(torch.tensor([tokenizer.encode(prompt)]))[0, -1]
    p, i = torch.softmax(logits, -1).topk(k)
    return " | ".join(f"{tokenizer.itos[j]} {q:.2f}" for q, j in zip(p.tolist(), i.tolist()))


def main():
    model, tok = load_model()
    W = model.token_emb.weight.detach()  # (V, 64)
    P = model.pos_emb.weight.detach()  # (8, 64)

    # 1 ---------------------------------------------------------------------
    header("1. Shapes")
    print(f"token table    : {tuple(W.shape)}  one row of {W.shape[1]} numbers per word")
    print(f"position table : {tuple(P.shape)}    one row per position 0..{P.shape[0] - 1}")
    cat = W[tok.stoi["cat"]]
    print(f"'cat' (ID {tok.stoi['cat']}) first 8 of 64 numbers: {[round(v, 2) for v in cat[:8].tolist()]}")

    # 2 ---------------------------------------------------------------------
    header("2. Softmax: scores -> probabilities, for 'the' at position 0")
    with torch.no_grad():
        logits = model(torch.tensor([[tok.stoi["the"]]]))[0, 0]  # (V,)
    probs = softmax_manual(logits)
    print(f"logits: {len(logits)} raw scores, from {logits.min():.2f} to {logits.max():.2f}")
    print(f"{'word':>10} {'logit':>7} {'exp(logit-max)':>15} {'prob':>6}")
    for i in logits.topk(6).indices.tolist():
        print(f"{tok.itos[i]:>10} {logits[i]:7.2f} {math.exp(logits[i] - logits.max()):15.3f} {probs[i]:6.3f}")
    print(f"sum of all {len(probs)} probabilities = {probs.sum():.4f}")
    print(f"manual vs torch.softmax, max difference = {(probs - torch.softmax(logits, -1)).abs().max():.2e}")

    # 3 ---------------------------------------------------------------------
    sentence = "the cat eats fish ."
    header(f"3. Cross-entropy for '{sentence}'")
    ids = tok.encode(sentence, add_eos=True)
    x, y = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])
    with torch.no_grad():
        logits = model(x)  # (1, T, V)
    probs = softmax_manual(logits)[0]
    print(f"{'pos':>3} {'sees':>6} {'true next':>10} {'p(true)':>8} {'-log p':>7}")
    for t in range(x.shape[1]):
        p = probs[t, y[0, t]].item()
        print(f"{t:>3} {tok.itos[x[0, t]]:>6} {tok.itos[y[0, t]]:>10} {p:8.3f} {abs(math.log(p)):7.3f}")
    manual = cross_entropy_manual(logits, y)
    torch_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    print(f"loss = mean(-log p): manual {manual:.4f} | F.cross_entropy {torch_ce:.4f} "
          f"| uniform guess ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}")

    # 4 ---------------------------------------------------------------------
    header("4. Nearest words (cosine similarity of token embeddings)")
    Wn = F.normalize(W, dim=1)
    sims = Wn @ Wn.T  # (V, V)
    for word in ["cat", "lion", "grass", "forest", "mother", "barks", "happy"]:
        i = tok.stoi[word]
        s = sims[i].clone()
        s[[i, tok.pad_id, tok.eos_id]] = -1  # skip itself and special tokens
        best = s.topk(5)
        print(f"{word:>8}: " + ", ".join(f"{tok.itos[j]} {v:.2f}" for v, j in zip(best.values.tolist(), best.indices.tolist())))

    # 5 ---------------------------------------------------------------------
    header(f"5. 2D picture of the embeddings -> {PLOT_PATH}")
    words, labels = [], []
    for cat_name, members in CATEGORIES.items():
        for w in members:
            if w in tok.stoi and w not in words:
                words.append(w)
                labels.append(cat_name)
    E = W[[tok.stoi[w] for w in words]]
    E = E - E.mean(0)
    _, _, Vh = torch.linalg.svd(E, full_matrices=False)
    xy = (E @ Vh[:2].T).numpy()  # project 64 dims onto the 2 main directions (PCA)

    fig, ax = plt.subplots(figsize=(12, 9))
    for cat_name, color in zip(CATEGORIES, plt.cm.tab10.colors):
        pts = [k for k, lab in enumerate(labels) if lab == cat_name]
        ax.scatter(xy[pts, 0], xy[pts, 1], color=color, label=cat_name, s=30)
        for k in pts:
            ax.annotate(words[k], xy[k], fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_title("Token embeddings (64 dims) projected to 2D with PCA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Plotted {len(words)} words in {len(CATEGORIES)} categories. Open {PLOT_PATH}.")

    # 6 ---------------------------------------------------------------------
    header(f"6. Does position matter? Two Step 2 models (no attention), {EXPLORE_EPOCHS} epochs each")
    results = {}
    for use_pos in (False, True):
        m, _, info = train(use_pos=use_pos, use_attn=False, epochs=EXPLORE_EPOCHS, verbose=False, save_path=None)
        results[use_pos] = (m, info)
        print(f"use_pos={str(use_pos):5} | val loss {info['val_loss']:.4f} | best possible {info['best_possible']:.4f}")
    prompts = ["the", "the dog chases the", "the cat sleeps on the"]
    for use_pos in (False, True):
        m = results[use_pos][0]
        print(f"\nuse_pos={use_pos}: next word after 'the' ...")
        for prompt in prompts:
            print(f"  at position {len(prompt.split()) - 1} ({prompt!r:>25}) -> {top_words(m, tok, prompt)}")
    print("\nWithout position, every 'the' gets the SAME prediction. With position, the model knows "
          "which part of the sentence it is in.")

    # 7 ---------------------------------------------------------------------
    header("7. What attention fixes: Step 2 model (no attention) vs Step 3 model (this model.pt)")
    step2 = results[True][0]  # the Step 2 model with position from section 6
    for prompt in ["he is a", "she is a", "the cow lives on the", "the monkey lives in the"]:
        print(f"{prompt!r:>27}")
        print(f"{'Step 2':>27} -> {top_words(step2, tok, prompt, k=4)}")
        print(f"{'Step 3':>27} -> {top_words(model, tok, prompt, k=4)}")
    print("Step 2 sees only the LAST word and its position, so each pair gets identical predictions. "
          "Step 3 looks back at 'he'/'she' and 'cow'/'monkey' through attention.")

    # 8 ---------------------------------------------------------------------
    sentences = ["he is a doctor .", "the cow lives on the farm ."]
    header(f"8. Attention map: who looks at whom -> {ATTN_PATH}")
    fig, axes = plt.subplots(1, len(sentences), figsize=(6 * len(sentences), 5.2))
    for ax, sentence in zip(axes, sentences):
        words = sentence.split()
        with torch.no_grad():
            model(torch.tensor([tok.encode(sentence)]))
        w = model.attn.last_weights[0]  # (T, T): row = word doing the looking, column = word looked at
        print(f"\n'{sentence}' (each row sums to 1; only the words at or before it)")
        for i, word in enumerate(words):
            row = ", ".join(f"{words[j]} {w[i, j]:.2f}" for j in range(i + 1))
            print(f"  {word:>7} looks at: {row}")
        ax.imshow(w.numpy(), cmap="Purples", vmin=0, vmax=1)
        ax.set_xticks(range(len(words)), words, rotation=45, ha="right")
        ax.set_yticks(range(len(words)), words)
        for i in range(len(words)):
            for j in range(len(words)):
                if j <= i:
                    ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if w[i, j] > 0.55 else "black")
        ax.set_xlabel("key: word being looked at")
        ax.set_ylabel("query: word doing the looking")
        ax.set_title(f"'{sentence}'")
    fig.tight_layout()
    fig.savefig(ATTN_PATH, dpi=150)
    print(f"\nSaved {ATTN_PATH}. Look at the row of the word that has to predict the next word.")


if __name__ == "__main__":
    main()
