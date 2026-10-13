"""Train the landmark classifier.

python -m ssl_recognition.train --config configs/baseline.yaml [--test-signer S1]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import ClipDataset, build_label_map, load_dataset, save_labels, split_clips
from .evaluate import evaluate
from .model import build_model, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: dict, test_signer: str | None = None) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(cfg.get("out_dir", "runs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = int(cfg.get("seq_len", 32))

    clips = load_dataset(Path(cfg["data_dir"]))
    label_map = build_label_map(clips)
    train_clips, val_clips, test_clips = split_clips(
        clips,
        seed=int(cfg.get("seed", 42)),
        test_signer=test_signer,
        official=bool(cfg.get("official_split", False)) and test_signer is None,
    )
    print(
        f"clips: train={len(train_clips)} val={len(val_clips)} test={len(test_clips)} "
        f"classes={len(label_map)}"
    )
    save_labels(out_dir / "labels.json", label_map)

    saved = {**cfg, "test_signer": test_signer}
    (out_dir / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    train_ds = ClipDataset(
        train_clips, label_map, seq_len, train=bool(cfg.get("augment", True)), cfg=cfg
    )
    val_ds = ClipDataset(val_clips, label_map, seq_len, cfg=cfg)
    bs = int(cfg.get("batch_size", 32))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=bs)

    model = build_model(cfg, num_classes=len(label_map))
    print(f"model params: {count_params(model):,}")
    epochs = int(cfg.get("epochs", 30))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(cfg.get("lr", 1e-3)), total_steps=max(1, epochs * len(train_dl))
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_top1, best_epoch = -1.0, -1
    best_key = (-1.0, float("-inf"))
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        val_metrics = (
            evaluate(model, val_dl) if len(val_ds) else {"top1": 0.0, "top5": 0.0, "loss": 0.0}
        )
        rec = {"epoch": epoch, "train_loss": total / max(1, n), **val_metrics}
        history.append(rec)
        print(
            f"epoch {epoch:3d} loss {rec['train_loss']:.4f} "
            f"val top1 {rec['top1']:.3f} top5 {rec['top5']:.3f} vloss {rec['loss']:.4f}"
        )
        # Validation top-1 saturates at 1.0 on same-signer validation, so break ties with the
        # validation loss: a more confident, better-calibrated model generalises better.
        key = (rec["top1"], -rec["loss"])
        if key > best_key:
            best_key, best_top1, best_epoch = key, rec["top1"], epoch
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg, "num_classes": len(label_map)},
                out_dir / "best.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = {
        "best_val_top1": best_top1,
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 1),
    }
    if test_clips:
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        test_dl = DataLoader(ClipDataset(test_clips, label_map, seq_len, cfg=cfg), batch_size=bs)
        result["test"] = evaluate(model, test_dl)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--test-signer", default=None, help="hold out one signer as the test set")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    train(cfg, test_signer=args.test_signer)


if __name__ == "__main__":
    main()
