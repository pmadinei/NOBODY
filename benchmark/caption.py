"""Caption the NOBODY-Bench images with any vision-language model, then score the captions.

Three backends cover open and proprietary models:

    # 1) Hugging Face transformers (any model with a chat template and AutoModelForImageTextToText)
    python caption.py --backend transformers --model Qwen/Qwen3-VL-8B-Instruct

    # 2) vLLM, offline batched inference (fastest for open models; optional LoRA adapter)
    python caption.py --backend vllm --model Qwen/Qwen3-VL-8B-Instruct [--lora path/to/adapter]

    # 3) any OpenAI-compatible endpoint (OpenAI, Gemini / Claude compatibility endpoints, a local `vllm serve`, ...)
    export OPENAI_API_KEY=...
    python caption.py --backend openai --model <model name> [--base-url http://localhost:8000/v1]

Every image gets one fresh chat with the benchmark prompt and greedy decoding (temperature 0). Captions are written
to captions/<name>.csv (columns id,caption) after every batch, so an interrupted run resumes where it stopped, and
the scores to results/<name>.json. Use `--limit 20` for a quick smoke test (scores of partial runs are not
comparable) and `--subset caption_agreement` / `judge_agreement` to caption one part only.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from pathlib import Path

from PIL import Image

from evaluate import evaluate, load_bench, report

HERE = Path(__file__).resolve().parent


def load_image(path: Path, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    return im


def data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------------------------------------------
class TransformersBackend:
    def __init__(self, a):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
        dtype = getattr(torch, a.dtype) if a.dtype != "auto" else "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(
            a.model, torch_dtype=dtype, device_map="auto", trust_remote_code=a.trust_remote_code).eval()
        if a.lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, a.lora).eval()
        tok = self.proc.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.a = a

    def __call__(self, images: list[Image.Image], prompt: str) -> list[str]:
        convs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
                 for im in images]
        inputs = self.proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", padding=True, **self.a.chat_kwargs)
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs and self.torch.is_floating_point(inputs["pixel_values"]):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                      max_new_tokens=self.a.max_tokens)
        new = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.proc.batch_decode(new, skip_special_tokens=True)]


class VllmBackend:
    def __init__(self, a):
        from vllm import LLM, SamplingParams
        self.llm = LLM(a.model, max_model_len=a.max_model_len, limit_mm_per_prompt={"image": 1},
                       enable_lora=bool(a.lora), trust_remote_code=a.trust_remote_code, seed=0)
        self.sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0)
        self.lora = None
        if a.lora:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("adapter", 1, a.lora)
        self.a = a

    def __call__(self, images: list[Image.Image], prompt: str) -> list[str]:
        convs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                               {"type": "text", "text": prompt}]}] for im in images]
        outs = self.llm.chat(convs, self.sp, chat_template_kwargs=self.a.chat_kwargs or None,
                             lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]


class OpenAIBackend:
    def __init__(self, a):
        from openai import OpenAI
        self.client = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env) or "EMPTY", max_retries=6)
        self.a = a

    def _one(self, im: Image.Image, prompt: str) -> str:
        kw = dict(self.a.request_kwargs)
        if not self.a.no_temperature:
            kw.setdefault("temperature", 0)
        r = self.client.chat.completions.create(
            model=self.a.model, **{self.a.max_tokens_field: self.a.max_tokens}, **kw,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(im)}},
                                                   {"type": "text", "text": prompt}]}])
        return (r.choices[0].message.content or "").strip()

    def __call__(self, images: list[Image.Image], prompt: str) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.a.concurrency) as ex:
            return list(ex.map(lambda im: self._one(im, prompt), images))


BACKENDS = {"transformers": TransformersBackend, "vllm": VllmBackend, "openai": OpenAIBackend}


# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Caption the NOBODY-Bench images with a VLM and score the captions.")
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--model", required=True, help="Hugging Face model id / path, or the API model name")
    ap.add_argument("--name", default=None, help="run name (default: derived from the model)")
    ap.add_argument("--bench", default=None, help="nobody_bench.json (default: next to this script)")
    ap.add_argument("--images-root", default=None, help="folder that contains images/ (default: next to the JSON)")
    ap.add_argument("--subset", choices=["all", "caption_agreement", "judge_agreement"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="caption only N images of each part (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1195, help="longer image side in pixels (0: no resizing)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--chat-kwargs", type=json.loads, default={},
                    help='chat-template arguments as JSON, e.g. \'{"enable_thinking": false}\'')
    ap.add_argument("--lora", default=None, help="LoRA adapter (transformers / vllm)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="auto", help="transformers: auto, bfloat16, float16, ...")
    ap.add_argument("--max-model-len", type=int, default=8192, help="vllm")
    ap.add_argument("--base-url", default=None, help="openai: endpoint URL (default: api.openai.com)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="openai: environment variable with the key")
    ap.add_argument("--concurrency", type=int, default=8, help="openai: parallel requests")
    ap.add_argument("--max-tokens-field", default="max_tokens",
                    help="openai: name of the length argument (max_tokens or max_completion_tokens)")
    ap.add_argument("--no-temperature", action="store_true", help="openai: do not send temperature=0")
    ap.add_argument("--request-kwargs", type=json.loads, default={}, help="openai: extra request arguments as JSON")
    a = ap.parse_args()

    bench_path = Path(a.bench) if a.bench else HERE / "nobody_bench.json"
    bench = load_bench(bench_path)
    root = Path(a.images_root) if a.images_root else bench_path.parent
    name = a.name or a.model.rstrip("/").split("/")[-1] + ("+lora" if a.lora else "")
    parts = ["caption_agreement", "judge_agreement"] if a.subset == "all" else [a.subset]
    items = []
    for p in parts:      # --limit: N evenly spaced images of each part (both sets of judge_agreement stay covered)
        part = bench[p]
        step = max(1, len(part) // a.limit) if a.limit else 1
        items += part[::step][:a.limit] if a.limit else part
    missing = [it["image"] for it in items if not (root / it["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found under {root} (first: {missing[0]}); see the README for "
                         "the image download, or pass --images-root")

    out = HERE / "captions" / f"{name}.csv"
    out.parent.mkdir(exist_ok=True)
    done: dict[str, str] = {}
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            done = {r["id"]: r["caption"] for r in csv.DictReader(fh)}
    todo = [it for it in items if it["id"] not in done]
    print(f"{name}: {len(items)} images, {len(items) - len(todo)} already captioned", flush=True)
    if todo:
        backend = BACKENDS[a.backend](a)
        new_file = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["id", "caption"])
            for i in range(0, len(todo), a.batch_size):
                batch = todo[i:i + a.batch_size]
                caps = backend([load_image(root / it["image"], a.max_side) for it in batch], bench["prompt"])
                for it, c in zip(batch, caps):
                    w.writerow([it["id"], c])
                    done[it["id"]] = c
                fh.flush()
                print(f"  {min(i + a.batch_size, len(todo))}/{len(todo)}", flush=True)

    res = evaluate(done, bench, name)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(report(res))
    n = res["n_captions"]
    if n["caption_agreement"] < n["of"] or n["judge_agreement"] < n["of_judge"]:
        print("note: partial run; scores are comparable with the reference results only on the full benchmark")


if __name__ == "__main__":
    main()
