# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ducky",
#     "datasets>=4",
#     "pyarrow>=24.0.0",
#     "torch>=2.12.0",
#     "transformers>=5",
# ]
#
# [tool.uv.sources]
# ducky = { path = "../", editable = true }
# ///
"""Fine-tune a small BERT on AG News from token IDs streamed by ducky.

    uv run examples/text_classification.py prepare
    uv run examples/text_classification.py loader --loader ducky --workers 4
    uv run examples/text_classification.py loader --loader hf-batched --workers 4
    uv run examples/text_classification.py train --steps 500

``prepare`` downloads AG News (120k train, 7.6k test headlines), tokenizes it once
and writes the token IDs as Parquet shards. ``loader`` times one epoch of
batches without a model, comparing ducky with Hugging Face ``datasets`` behind a
torch DataLoader. ``train`` fine-tunes the model and reports how much of each
step was spent waiting for data.
"""

import argparse
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

import ducky

PARQUET = "https://huggingface.co/api/datasets/fancyzhx/ag_news/parquet/default/{split}/0.parquet"
MODEL = "google/bert_uncased_L-4_H-256_A-4"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
DATA = Path(__file__).parent / ".cache" / "ag_news"


def prepare(data_dir: Path, seq_len: int, shards: int) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    for split in ("train", "test"):
        raw = data_dir / f"raw-{split}.parquet"
        if not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(PARQUET.format(split=split), raw)
        table = pq.read_table(raw)
        start = time.perf_counter()
        ids = tokenizer(
            table["text"].to_pylist(),
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        print(f"{split}: tokenized {len(ids)} rows in {time.perf_counter() - start:.1f}s")
        out = pa.table(
            {
                "id": pa.array(np.arange(len(ids))),
                "label": table["label"].cast(pa.int8()),
                "input_ids": pa.FixedSizeListArray.from_arrays(ids.reshape(-1), seq_len),
            }
        )
        target = data_dir / split
        target.mkdir(exist_ok=True)
        n = shards if split == "train" else 1
        for i, rows in enumerate(np.array_split(np.arange(len(ids)), n)):
            pq.write_table(out.take(rows), target / f"part-{i:02d}.parquet")


def ducky_loader(
    data_dir: Path, split: str, seq_len: int, batch_size: int, workers: int, train: bool
) -> tuple[Any, ducky.Stream[torch.Tensor]]:
    stream = ducky.stream(
        str(data_dir / split / "*.parquet"),
        fields={
            "input_ids": ducky.tensor("input_ids", seq_len, dtype=torch.int32),
            "labels": ducky.vector("label", dtype=torch.int64),
        },
        key="id",
        batch_size=batch_size,
        shuffle=ducky.shuffle(64, seed=0) if train else None,
        drop_last=train,
        backend="torch",
    )
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=workers, persistent_workers=workers > 0
    )
    return loader, stream


def hf_loader(data_dir: Path, split: str, batch_size: int, workers: int, batched: bool) -> Any:
    """The usual ``datasets`` pipeline: a map-style dataset behind a DataLoader."""
    import datasets

    files = sorted(str(p) for p in (data_dir / split).glob("*.parquet"))
    ds = datasets.load_dataset("parquet", data_files=files, split="train")
    ds = ds.rename_column("label", "labels").with_format("torch", columns=["input_ids", "labels"])
    if not batched:  # one __getitem__ per row, collated by the DataLoader
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(ds), batch_size, drop_last=True
    )
    return torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def make_loader(args: argparse.Namespace, split: str = "train", train: bool = True) -> Any:
    if args.loader == "ducky":
        loader, _ = ducky_loader(
            args.data, split, args.seq_len, args.batch_size, args.workers, train
        )
        return loader
    return hf_loader(args.data, split, args.batch_size, args.workers, args.loader == "hf-batched")


def time_loader(args: argparse.Namespace) -> None:
    loader = make_loader(args)
    for epoch in range(args.epochs):
        if isinstance(loader.dataset, ducky.Stream):
            loader.dataset.set_epoch(epoch)
        start = time.perf_counter()
        first = rows = batches = 0
        for batch in loader:
            if not batches:
                first = time.perf_counter() - start
            rows += len(batch["labels"])
            batches += 1
        elapsed = time.perf_counter() - start
        print(
            f"{args.loader} epoch {epoch}: {batches} batches, {rows / elapsed:,.0f} rows/s, "
            f"{elapsed:.2f}s total, first batch after {first * 1000:.0f} ms"
        )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model: Any, args: argparse.Namespace) -> float:
    loader, _ = ducky_loader(args.data, "test", args.seq_len, 512, 0, train=False)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            logits = model(input_ids=input_ids, attention_mask=input_ids != 0).logits
            correct += int((logits.argmax(-1).cpu() == batch["labels"]).sum())
            total += len(batch["labels"])
    model.train()
    return correct / total


def train(args: argparse.Namespace) -> None:
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = make_loader(args)

    step = 0
    wait = compute = 0.0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        if isinstance(loader.dataset, ducky.Stream):
            loader.dataset.set_epoch(epoch)
        fetched = time.perf_counter()
        for batch in loader:
            ready = time.perf_counter()
            wait += ready - fetched
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            out = model(input_ids=input_ids, attention_mask=input_ids != 0, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss = out.loss.item()  # waits for the device, so the step is timed in full
            fetched = time.perf_counter()
            compute += fetched - ready
            step += 1
            if step % args.log_every == 0:
                print(
                    f"step {step:>5} | loss {loss:.4f} | {step / (fetched - start):.1f} steps/s"
                    f" | waiting on data {wait / (wait + compute):.1%}"
                )
            if step == args.steps:
                break
        if step == args.steps:
            break
    elapsed = time.perf_counter() - start
    print(
        f"\n{step} steps in {elapsed:.1f}s ({step * args.batch_size / elapsed:,.0f} rows/s) "
        f"on {args.device} with {args.loader}, {args.workers} workers: "
        f"{wait:.1f}s waiting on data, {compute:.1f}s in the model"
    )
    print(f"test accuracy {evaluate(model, args):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("prepare", "loader", "train"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--shards", type=int, default=8, help="training Parquet files")
    parser.add_argument("--loader", choices=("ducky", "hf", "hf-batched"), default="ducky")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="stop after this many; 0 for all")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.seq_len, args.shards)
    elif args.command == "loader":
        time_loader(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
