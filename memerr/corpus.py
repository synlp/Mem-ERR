import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sentence(text):
    return re.sub(r"\s+", " ", text.strip())


def split_report(report):
    text = report.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [normalize_sentence(piece) for piece in pieces if normalize_sentence(piece)]


def report_fingerprint(report):
    text = " ".join(split_report(report)).lower().encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def read_annotations(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for split in SPLITS:
        if split not in data or not isinstance(data[split], list):
            raise ValueError(f"annotation split is missing: {split}")
        data[split] = [
            record
            for record in data[split]
            if isinstance(record.get("findings", record.get("report")), str)
            and record.get("findings", record.get("report")).strip()
        ]
    validate_splits(data)
    return data


def record_id(record):
    value = record.get("id", record.get("study_id"))
    if value is None:
        raise ValueError("record id is missing")
    return str(value)


def record_images(record):
    value = record.get("image_path", record.get("images"))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"image paths are missing for {record_id(record)}")
    return [str(path) for path in value]


def record_report(record):
    value = record.get("findings", record.get("report"))
    if not isinstance(value, str):
        raise ValueError(f"findings report is missing for {record_id(record)}")
    return value


def validate_splits(data):
    case_sets = {}
    image_sets = {}
    for split in SPLITS:
        cases = [record_id(record) for record in data[split]]
        images = [image for record in data[split] for image in record_images(record)]
        if len(cases) != len(set(cases)):
            raise ValueError(f"duplicate case id in {split}")
        if len(images) != len(set(images)):
            raise ValueError(f"duplicate image path in {split}")
        case_sets[split] = set(cases)
        image_sets[split] = set(images)
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            if case_sets[left] & case_sets[right]:
                raise ValueError(f"case leakage between {left} and {right}")
            if image_sets[left] & image_sets[right]:
                raise ValueError(f"image leakage between {left} and {right}")
    return True


def build_training_corpus(data):
    validate_splits(data)
    sentence_counts = Counter()
    sentence_cases = defaultdict(set)
    sentence_images = defaultdict(set)
    case_reports = {}
    train_cases = set()
    train_images = set()
    for record in data["train"]:
        case_id = record_id(record)
        images = record_images(record)
        report = record_report(record)
        sentences = split_report(report)
        train_cases.add(case_id)
        train_images.update(images)
        case_reports[case_id] = report_fingerprint(report)
        for sentence in sentences:
            sentence_counts[sentence] += 1
            sentence_cases[sentence].add(case_id)
            sentence_images[sentence].update(images)
    return {
        "sentence_counts": dict(sentence_counts),
        "sentence_cases": {key: sorted(value) for key, value in sentence_cases.items()},
        "sentence_images": {key: sorted(value) for key, value in sentence_images.items()},
        "train_case_ids": sorted(train_cases),
        "train_image_paths": sorted(train_images),
        "train_report_fingerprints": case_reports,
    }
