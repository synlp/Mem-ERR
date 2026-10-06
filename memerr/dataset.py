from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .corpus import record_id, record_images, record_report


class RadiologyDataset(Dataset):
    def __init__(self, records, image_root, segmented_root, processor, split, view_index=0):
        if view_index < 0:
            raise ValueError("view_index must be nonnegative")
        self.records = records
        self.image_root = Path(image_root)
        self.segmented_root = Path(segmented_root)
        self.processor = processor
        self.split = split
        self.view_index = view_index

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        paths = record_images(record)
        if self.view_index >= len(paths):
            raise ValueError(f"view index unavailable for {record_id(record)}")
        relative_path = paths[self.view_index]
        with Image.open(self.image_root / relative_path) as image:
            raw = image.convert("RGB")
        with Image.open(self.segmented_root / relative_path) as image:
            segmented = image.convert("RGB")
        raw_pixels = self.processor(images=raw, return_tensors="pt")["pixel_values"][0]
        segmented_pixels = self.processor(images=segmented, return_tensors="pt")["pixel_values"][0]
        return {
            "case_id": record_id(record),
            "image_paths": paths,
            "raw_pixels": raw_pixels,
            "segmented_pixels": segmented_pixels,
            "report": record_report(record),
            "split": self.split,
        }


def collate_records(records):
    return {
        "case_ids": [record["case_id"] for record in records],
        "image_paths": [record["image_paths"] for record in records],
        "raw_pixels": torch.stack([record["raw_pixels"] for record in records]),
        "segmented_pixels": torch.stack([record["segmented_pixels"] for record in records]),
        "reports": [record["report"] for record in records],
        "splits": [record["split"] for record in records],
    }
