"""NOBODY-Bench evaluator: PAMA, PPMA and PJA of a vision-language model's captions.

usage
    python evaluate.py --captions my_model.csv [--bench nobody_bench.json] [--name my_model] [--out my_model.json]

input
    One caption per benchmark image, written with the benchmark prompt (`prompt` in nobody_bench.json; `caption.py`
    produces the file for any VLM). Either
      * a CSV with the columns `id,caption` (the column names `img_fn` / `image` and `description` also work), or
      * a JSON object {id: caption}, or a JSON list of {"id": ..., "caption": ...}.
    `id` is the `id` of a benchmark image; its file name or its `image` path are accepted too.

measures (all in [0, 1], reported in %, higher is better unless noted)
    PAMA  Person-Absent Mention Agreement. For a caption with person mention m in {0, 1} and an image whose human
          captions mention a person with rate h, the agreement m*h + (1-m)*(1-h) is the probability that a randomly
          drawn human caption of the image makes the same decision. PAMA is its mean over the 664 images without a
          visible person.
    PPMA  Person-Present Mention Agreement: the same mean over the 949 images with a visible person. Read the two
          together: a model that never mentions people reaches 87.5% PAMA but only 6.9% PPMA.
    PJA   Public-Judge Agreement. Human judges of four public preference datasets preferred a response without a
          person for the 1,813 images of set A and one with a person for the 1,813 images of set B. PJA is the mean
          of the model's person-mention rate on B and one minus its rate on A.
    APM / VPR  raw person-mention rates on the person-absent / person-present images (APM: lower is closer to
          humans, who are at 12.5%; VPR: humans are at 93.1%).
    95% intervals: bootstrap over scenes (an original image and its counterfactuals are resampled together) for
    PAMA / PPMA / APM / VPR, over images within each set for PJA.

Text-only: needs Python >= 3.9 and numpy, and runs in seconds on a CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from person_lexicon import mentions_person, normalize_caption

HERE = Path(__file__).resolve().parent
N_BOOT = 2000


# ---------------------------------------------------------------------------------------------------------------
def load_bench(path: str | Path | None = None) -> dict:
    path = Path(path) if path else HERE / "nobody_bench.json"
    if not path.exists():
        sys.exit(f"benchmark file not found: {path} (pass --bench /path/to/nobody_bench.json)")
    return json.loads(path.read_text(encoding="utf-8"))


def load_captions(path: str | Path) -> dict[str, str]:
    """{key: caption} from a CSV (id,caption) or a JSON file; keys are resolved against the benchmark later."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): "" if v is None else str(v) for k, v in raw.items()}
        return {str(r.get("id", r.get("image"))): str(r.get("caption") or "") for r in raw}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    kcol = next((c for c in ("id", "img_fn", "image", "file_name") if c in rows[0]), None)
    ccol = next((c for c in ("caption", "description", "text") if c in rows[0]), None)
    if kcol is None or ccol is None:
        sys.exit(f"{path}: expected the columns id,caption (found {list(rows[0])})")
    return {str(r[kcol]): str(r[ccol] or "") for r in rows}


def _resolve(items: list[dict], captions: dict[str, str]) -> dict[str, str]:
    """{benchmark id: caption}; a caption may be keyed by the id, the file name or the image path."""
    out = {}
    for it in items:
        for key in (it["id"], Path(it["image"]).name, it["image"], Path(it["image"]).stem):
            if key in captions:
                out[it["id"]] = captions[key]
                break
    return out


def _boot(vals: np.ndarray, clusters: np.ndarray, seed: int = 0) -> list[float]:
    """Percentile interval of the mean of `vals`, resampling whole clusters."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(uniq), (N_BOOT, len(uniq)))
    b = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ---------------------------------------------------------------------------------------------------------------
def caption_agreement(items: list[dict], captions: dict[str, str]) -> dict:
    """PAMA, PPMA, APM and VPR on the images with human captions."""
    out = {}
    for present, agree_name, rate_name in ((False, "pama", "apm"), (True, "ppma", "vpr")):
        sub = [it for it in items if it["person_present"] == present and it["id"] in captions]
        if not sub:
            continue
        m = np.array([float(mentions_person(normalize_caption(captions[it["id"]]))) for it in sub])
        h = np.array([np.mean([c["mentions_person"] for c in it["human_captions"]]) for it in sub])
        scene = np.array([it["scene_id"] for it in sub])
        agree = m * h + (1 - m) * (1 - h)
        out[agree_name] = float(agree.mean())
        out[agree_name + "_ci"] = _boot(agree, scene)
        out[rate_name] = float(m.mean())
        out[rate_name + "_ci"] = _boot(m, scene)
        out[f"n_person_{'present' if present else 'absent'}"] = len(sub)
    return out


def judge_agreement(items: list[dict], captions: dict[str, str]) -> dict:
    """PJA on the images whose human judges preferred a response without (set A) / with (set B) a person."""
    out, agree = {}, {}
    for st in ("A", "B"):
        sub = [it for it in items if it["set"] == st and it["id"] in captions]
        if not sub:
            return {}
        m = np.array([float(mentions_person(normalize_caption(captions[it["id"]]))) for it in sub])
        agree[st] = m if st == "B" else 1 - m
        out[f"mention_rate_set_{st}"] = float(m.mean())
        out[f"n_set_{st}"] = len(sub)
    out["pja"] = float((agree["A"].mean() + agree["B"].mean()) / 2)
    rng = np.random.default_rng(0)
    b = [(rng.choice(agree["A"], len(agree["A"])).mean() + rng.choice(agree["B"], len(agree["B"])).mean()) / 2
         for _ in range(N_BOOT)]
    out["pja_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


def human_reference(bench: dict) -> dict:
    """Inter-human agreement: every human caption against the other human captions of its image (leave-one-out)."""
    out = {}
    for present, agree_name, rate_name in ((False, "pama", "apm"), (True, "ppma", "vpr")):
        vals, rates = [], []
        for it in bench["caption_agreement"]:
            b = np.array([float(c["mentions_person"]) for c in it["human_captions"]])
            if it["person_present"] != present or len(b) < 2:
                continue
            others = (b.sum() - b) / (len(b) - 1)
            vals.append(float(np.mean(b * others + (1 - b) * (1 - others))))
            rates.append(float(b.mean()))
        out[agree_name], out[rate_name] = float(np.mean(vals)), float(np.mean(rates))
    return out


def evaluate(captions: dict[str, str], bench: dict | None = None, name: str = "model") -> dict:
    bench = bench or load_bench()
    ca, ja = bench["caption_agreement"], bench["judge_agreement"]
    cap = {**_resolve(ca, captions), **_resolve(ja, captions)}
    words = [len(normalize_caption(cap[it["id"]]).split()) for it in ca if it["id"] in cap]
    res = dict(name=name,
               n_captions=dict(caption_agreement=sum(it["id"] in cap for it in ca), of=len(ca),
                               judge_agreement=sum(it["id"] in cap for it in ja), of_judge=len(ja)))
    res.update(caption_agreement(ca, cap))
    res.update(judge_agreement(ja, cap))
    if words:
        res["mean_words"] = float(np.mean(words))
    res["human_reference"] = human_reference(bench)
    return res


def report(res: dict) -> str:
    hum = res["human_reference"]
    rows = [("PAMA", "pama", "up"), ("PPMA", "ppma", "up"), ("PJA", "pja", "up"), ("APM", "apm", ""), ("VPR", "vpr", "")]
    lines = [f"NOBODY-Bench  |  {res['name']}", f"{'measure':<8}{'model':>8}   {'95% CI':<16}{'human captions':>15}"]
    for label, key, _ in rows:
        if key not in res:
            lines.append(f"{label:<8}{'--':>8}   (no captions for this part of the benchmark)")
            continue
        lo, hi = res[key + "_ci"]
        h = f"{100 * hum[key]:.1f}" if key in hum else "--"
        lines.append(f"{label:<8}{100 * res[key]:>8.1f}   [{100 * lo:.1f}, {100 * hi:.1f}]{'':<3}{h:>12}")
    n = res["n_captions"]
    lines.append(f"captions scored: {n['caption_agreement']}/{n['of']} (PAMA, PPMA), "
                 f"{n['judge_agreement']}/{n['of_judge']} (PJA)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score captions on NOBODY-Bench (PAMA, PPMA, PJA).")
    ap.add_argument("--captions", required=True, help="CSV (id,caption) or JSON ({id: caption})")
    ap.add_argument("--bench", default=None, help="path to nobody_bench.json (default: next to this script)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="write the scores to this JSON file")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score a partial caption file (scores of partial files are not comparable)")
    a = ap.parse_args()
    bench = load_bench(a.bench)
    res = evaluate(load_captions(a.captions), bench, a.name or Path(a.captions).stem)
    n = res["n_captions"]
    missing = (n["of"] - n["caption_agreement"]) + (n["of_judge"] - n["judge_agreement"])
    if missing and not a.allow_missing:
        sys.exit(f"{missing} benchmark images have no caption ({n['caption_agreement']}/{n['of']} with human "
                 f"captions, {n['judge_agreement']}/{n['of_judge']} with human judges). Caption every image, or pass "
                 "--allow-missing to score the part you have.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    print(report(res))


if __name__ == "__main__":
    main()
