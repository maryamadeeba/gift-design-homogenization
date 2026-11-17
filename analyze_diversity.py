"""Compare the diversity of human-made vs AI-generated gift designs.

Folder layout:
    data/human/gift01.jpeg ... gift30.jpeg
    data/ai/ai01.png ... ai30.png
Only ids present in both folders are compared, so dropping a photo drops its pair.

Run (CPU is fine):
    pip install torch open_clip_torch umap-learn matplotlib pillow numpy
    python analyze_diversity.py

Outputs results.json and umap.png in the current folder.
For the background-removed images:
    python analyze_diversity.py --human data_nobg/human --ai data_nobg/ai --tag nobg
(writes results_nobg.json and umap_nobg.png, so the first results are kept)
"""
import argparse
import json
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value")
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

HUMAN_DIR = Path("data/human")
AI_DIR = Path("data/ai")
CLIP_ARCH, CLIP_WEIGHTS = "ViT-B-32-quickgelu", "openai"  # matches how the openai weights were trained
N_PERMUTATIONS = 5000
N_DRAWS = 1000          # subsample draws for the uncertainty range
SUBSAMPLE_FRACTION = 0.8
RNG = np.random.default_rng(42)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def numbered_files(folder):
    """Map the number in each file name to its path (human_07.jpg -> 7)."""
    out = {}
    for p in folder.iterdir():
        m = re.search(r"(\d+)", p.stem)
        if p.suffix.lower() in EXTS and m:
            out[int(m.group(1))] = p
    return out


def embed(paths, model, preprocess):
    vecs = []
    with torch.no_grad():
        for p in paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            v = model.encode_image(img)
            vecs.append((v / v.norm(dim=-1, keepdim=True)).squeeze(0).numpy())
    return np.stack(vecs)


def vendi_score(x):
    """Effective number of distinct items: exp(entropy of eigenvalues of K/n)."""
    k = x @ x.T
    w = np.linalg.eigvalsh(k / len(x))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def mean_pairwise_distance(x):
    sim = x @ x.T
    n = len(x)
    off = sim[~np.eye(n, dtype=bool)]
    return float(1 - off.mean())


def subsample_range(x, fn):
    """5th-95th percentile of fn over random subsets drawn without replacement."""
    m = max(3, int(round(len(x) * SUBSAMPLE_FRACTION)))
    vals = [fn(x[RNG.choice(len(x), m, replace=False)]) for _ in range(N_DRAWS)]
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def main():
    global HUMAN_DIR, AI_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=str(HUMAN_DIR))
    ap.add_argument("--ai", default=str(AI_DIR))
    ap.add_argument("--tag", default="", help="added to output names, e.g. nobg")
    args = ap.parse_args()
    HUMAN_DIR, AI_DIR = Path(args.human), Path(args.ai)
    suffix = f"_{args.tag}" if args.tag else ""
    human_all, ai_all = numbered_files(HUMAN_DIR), numbered_files(AI_DIR)
    ids = sorted(set(human_all) & set(ai_all))   # keep only complete human/AI pairs
    dropped = sorted(set(human_all) ^ set(ai_all))
    if dropped:
        print("skipping unpaired ids:", dropped)
    human_paths = [human_all[i] for i in ids]
    ai_paths = [ai_all[i] for i in ids]
    print(f"{len(ids)} matched pairs")

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_WEIGHTS)
    model.eval()
    h = embed(human_paths, model, preprocess)
    a = embed(ai_paths, model, preprocess)

    results = {"clip_model": f"{CLIP_ARCH} ({CLIP_WEIGHTS})", "pair_ids": ids, "n_pairs": len(ids)}
    for name, x in (("human", h), ("ai", a)):
        results[name] = {
            "vendi_score": vendi_score(x),
            "vendi_range_5_95": subsample_range(x, vendi_score),
            "mean_pairwise_distance": mean_pairwise_distance(x),
            "distance_range_5_95": subsample_range(x, mean_pairwise_distance),
        }

    # Permutation test: if "human" vs "ai" labels meant nothing, how often would
    # a random split of the same images show a gap this large?
    both = np.vstack([h, a])
    n = len(h)
    for name, fn in (("vendi", vendi_score), ("distance", mean_pairwise_distance)):
        observed = fn(h) - fn(a)
        hits = 0
        for _ in range(N_PERMUTATIONS):
            idx = RNG.permutation(len(both))
            if fn(both[idx[:n]]) - fn(both[idx[n:]]) >= observed:
                hits += 1
        results[f"{name}_gap_human_minus_ai"] = float(observed)
        results[f"{name}_gap_p_value"] = (hits + 1) / (N_PERMUTATIONS + 1)

    # How close each AI design lands to its own prompt twin and to any human design.
    if len(h) == len(a):
        twin = 1 - np.sum(h * a, axis=1)
        results["mean_distance_to_matched_twin"] = float(twin.mean())
    nearest_human = 1 - (a @ h.T).max(axis=1)
    results["mean_distance_ai_to_nearest_human"] = float(nearest_human.mean())

    Path(f"results{suffix}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # Picture only: claims come from the numbers above, not from cluster sizes in the plot.
    import umap
    import matplotlib.pyplot as plt
    labels = ["human"] * len(h) + ["ai"] * len(a)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, nn in zip(axes, (5, 10, 15)):
        nn = min(nn, len(both) - 1)
        xy = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="cosine", random_state=42).fit_transform(both)
        for lab, color in (("human", "#534AB7"), ("ai", "#D85A30")):
            idx = [i for i, l in enumerate(labels) if l == lab]
            ax.scatter(xy[idx, 0], xy[idx, 1], c=color, label=lab, s=40)
        ax.set_title(f"UMAP, n_neighbors={nn}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"umap{suffix}.png", dpi=150)
    print(f"saved umap{suffix}.png and results{suffix}.json")


if __name__ == "__main__":
    main()
