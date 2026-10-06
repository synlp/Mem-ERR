import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

from memerr.config import load_config, require
from memerr.corpus import file_fingerprint, read_annotations, split_report
from memerr.dataset import RadiologyDataset, collate_records
from memerr.extraction import SentencePool
from memerr.model import MemERR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    config = load_config(args.config)
    torch.manual_seed(int(require(config, "runtime", "seed")))
    annotations = read_annotations(require(config, "data", "annotations"))
    pool = SentencePool.load(args.pool)
    pool.assert_annotation_source(annotations)
    model_name = require(config, "model", "clip_model")
    processor = CLIPImageProcessor.from_pretrained(model_name)
    dataset = RadiologyDataset(
        annotations["train"],
        require(config, "data", "image_root"),
        require(config, "data", "segmented_root"),
        processor,
        "train",
        int(require(config, "data", "view_index")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(require(config, "training", "batch_size")),
        shuffle=True,
        collate_fn=collate_records,
    )
    model = MemERR(
        model_name,
        pool.memory_embeddings,
        hidden_dimension=int(require(config, "model", "hidden_dimension")),
        layers=int(require(config, "model", "fusion_layers")),
        heads=int(require(config, "fallbacks", "fusion_heads")),
        feedforward_dimension=int(require(config, "fallbacks", "feedforward_dimension")),
        dropout=float(require(config, "fallbacks", "dropout")),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(require(config, "training", "learning_rate")),
        weight_decay=float(require(config, "training", "weight_decay")),
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        matched_sentences = 0
        steps = 0
        for batch in loader:
            enhanced = model(batch["raw_pixels"].to(device), batch["segmented_pixels"].to(device))
            losses = []
            for index, report in enumerate(batch["reports"]):
                loss, matched = pool.training_loss(
                    enhanced[index],
                    split_report(report),
                    batch["case_ids"][index],
                    batch["image_paths"][index],
                    float(require(config, "extraction", "gamma_r")),
                    float(require(config, "extraction", "gamma_t")),
                )
                if matched:
                    losses.append(loss)
                    matched_sentences += matched
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        if not steps:
            raise RuntimeError("no reusable ground-truth sentence in training epoch")
        print(f"epoch={epoch + 1} loss={epoch_loss / steps:.6f} matched={matched_sentences}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pool_sha256 = file_fingerprint(args.pool)
    torch.save({"model": model.state_dict(), "config": config, "pool_sha256": pool_sha256}, output)


if __name__ == "__main__":
    main()
