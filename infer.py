import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

from memerr.config import load_config, require
from memerr.corpus import file_fingerprint, read_annotations
from memerr.dataset import RadiologyDataset, collate_records
from memerr.extraction import SentencePool, assemble_report
from memerr.model import MemERR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(require(config, "runtime", "seed"))
    torch.manual_seed(seed)
    annotations = read_annotations(require(config, "data", "annotations"))
    pool = SentencePool.load(args.pool)
    pool.assert_annotation_source(annotations)
    model_name = require(config, "model", "clip_model")
    processor = CLIPImageProcessor.from_pretrained(model_name)
    dataset = RadiologyDataset(
        annotations[args.split],
        require(config, "data", "image_root"),
        require(config, "data", "segmented_root"),
        processor,
        args.split,
        int(require(config, "data", "view_index")),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_records)
    model = MemERR(
        model_name,
        pool.memory_embeddings,
        hidden_dimension=int(require(config, "model", "hidden_dimension")),
        layers=int(require(config, "model", "fusion_layers")),
        heads=int(require(config, "fallbacks", "fusion_heads")),
        feedforward_dimension=int(require(config, "fallbacks", "feedforward_dimension")),
        dropout=float(require(config, "fallbacks", "dropout")),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    pool_sha256 = file_fingerprint(args.pool)
    if checkpoint.get("pool_sha256") != pool_sha256:
        raise ValueError("checkpoint was trained with a different sentence pool")
    for section in ("model", "fallbacks", "extraction"):
        if checkpoint["config"][section] != config[section]:
            raise ValueError(f"checkpoint {section} settings differ from config")
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            enhanced = model(batch["raw_pixels"].to(device), batch["segmented_pixels"].to(device))[0]
            candidates = pool.extract(
                enhanced,
                batch["case_ids"][0],
                batch["image_paths"][0],
                args.split,
                int(require(config, "extraction", "top_k")),
                float(require(config, "extraction", "gamma_r")),
                float(require(config, "extraction", "gamma_t")),
                float(require(config, "extraction", "redundancy_threshold")),
                seed + index,
                int(require(config, "fallbacks", "index_clusters")),
            )
            outputs.append(
                {
                    "id": batch["case_ids"][0],
                    "prediction": assemble_report(candidates),
                    "sentences": [candidate.sentence for candidate in candidates],
                    "scores": [candidate.score for candidate in candidates],
                    "source_case_ids": [list(candidate.source_case_ids) for candidate in candidates],
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
