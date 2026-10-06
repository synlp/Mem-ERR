import argparse
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from memerr.config import load_config, require
from memerr.corpus import build_training_corpus, read_annotations


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def stream_batches(values, size):
    iterator = iter(values)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def encode_sentences(model, tokenizer, sentences, batch_size, device):
    embeddings = []
    model.eval()
    for batch in batches(sentences, batch_size):
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model.text_model(**tokens).last_hidden_state
            projected = model.text_projection(hidden)
            mask = tokens["attention_mask"].unsqueeze(-1)
            mean = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings.append(mean.cpu())
    return torch.cat(embeddings, dim=0)


def encode_images(model, processor, image_root, image_paths, batch_size, device):
    from PIL import Image

    embeddings = []
    model.eval()
    for batch in batches(image_paths, batch_size):
        images = []
        for relative_path in batch:
            with Image.open(image_root / relative_path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            hidden = model.vision_model(pixel_values=pixels).last_hidden_state
            projected = model.visual_projection(hidden)
        embeddings.append(projected.cpu())
    return torch.cat(embeddings, dim=0)


def association_vectors(sentence_embeddings, image_embeddings, sentence_image_indices):
    for sentence_id, image_indices in enumerate(sentence_image_indices):
        text = sentence_embeddings[sentence_id].numpy()
        for image_index in image_indices:
            visual = image_embeddings[image_index].numpy().reshape(-1)
            yield sentence_id, np.concatenate((visual, text), axis=0)


def initialize_centers(sentence_embeddings, image_embeddings, sentence_image_indices, cluster_count, seed):
    rng = np.random.default_rng(seed)
    centers = []
    seen = 0
    for _, vector in association_vectors(sentence_embeddings, image_embeddings, sentence_image_indices):
        if seen < cluster_count:
            centers.append(vector.copy())
        else:
            index = int(rng.integers(0, seen + 1))
            if index < cluster_count:
                centers[index] = vector.copy()
        seen += 1
    if seen < cluster_count:
        raise ValueError("cluster count exceeds sentence-image pairs")
    return np.stack(centers)


def squared_distances(vectors, centers):
    vector_norms = torch.sum(vectors * vectors, dim=1, keepdim=True)
    center_norms = torch.sum(centers * centers, dim=1).unsqueeze(0)
    return (vector_norms + center_norms - 2.0 * vectors @ centers.transpose(0, 1)).clamp_min(0.0)


def soft_kmeans_centers(sentence_embeddings, image_embeddings, sentence_image_indices, cluster_count, batch_size, temperature, iterations, seed, device):
    if temperature <= 0:
        raise ValueError("soft k-means temperature must be positive")
    if iterations < 1:
        raise ValueError("soft k-means iterations must be positive")
    initialized = initialize_centers(
        sentence_embeddings,
        image_embeddings,
        sentence_image_indices,
        cluster_count,
        seed,
    )
    centers = torch.from_numpy(initialized).float().to(device)
    with torch.no_grad():
        for _ in range(iterations):
            sums = torch.zeros_like(centers)
            weights = torch.zeros(cluster_count, dtype=centers.dtype, device=device)
            pairs = association_vectors(sentence_embeddings, image_embeddings, sentence_image_indices)
            for batch in stream_batches(pairs, batch_size):
                vectors = torch.from_numpy(np.stack([vector for _, vector in batch])).float().to(device)
                responsibilities = torch.softmax(-squared_distances(vectors, centers) / temperature, dim=1)
                sums.add_(responsibilities.transpose(0, 1) @ vectors)
                weights.add_(responsibilities.sum(dim=0))
            active = weights > torch.finfo(weights.dtype).eps
            centers[active] = sums[active] / weights[active].unsqueeze(1)
    return centers


def closest_cluster_mapping(sentence_embeddings, image_embeddings, sentence_image_indices, centers, batch_size, device):
    cluster_sentences = defaultdict(set)
    pairs = association_vectors(sentence_embeddings, image_embeddings, sentence_image_indices)
    with torch.no_grad():
        for batch in stream_batches(pairs, batch_size):
            vectors = torch.from_numpy(np.stack([vector for _, vector in batch])).float().to(device)
            labels = torch.argmin(squared_distances(vectors, centers), dim=1).cpu().tolist()
            for (sentence_id, _), label in zip(batch, labels):
                cluster_sentences[label].add(sentence_id)
    return [sorted(cluster_sentences[index]) for index in range(centers.shape[0])]


def build_index(sentence_embeddings, image_embeddings, sentence_image_indices, cluster_count, batch_size, temperature, iterations, seed, device):
    if cluster_count < 1:
        raise ValueError("cluster count must be positive")
    if batch_size < 1:
        raise ValueError("cluster batch size must be positive")
    pair_count = sum(len(indices) for indices in sentence_image_indices)
    if pair_count < cluster_count:
        raise ValueError("cluster count exceeds sentence-image pairs")
    centers = soft_kmeans_centers(
        sentence_embeddings,
        image_embeddings,
        sentence_image_indices,
        cluster_count,
        batch_size,
        temperature,
        iterations,
        seed,
        device,
    )
    mapping = closest_cluster_mapping(
        sentence_embeddings,
        image_embeddings,
        sentence_image_indices,
        centers,
        batch_size,
        device,
    )
    return centers.cpu(), mapping


def memory_row_status(actual, expected):
    expected_value = None if expected is None else int(expected)
    return {
        "actual": int(actual),
        "expected": expected_value,
        "match": expected_value is None or int(actual) == expected_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

    config = load_config(args.config)
    if require(config, "fallbacks", "cluster_algorithm") != "soft_kmeans":
        raise ValueError("unsupported cluster algorithm")
    annotations = read_annotations(require(config, "data", "annotations"))
    corpus = build_training_corpus(annotations)
    image_root = Path(require(config, "data", "image_root"))
    model_name = require(config, "model", "clip_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    sentences = sorted(corpus["sentence_counts"])
    image_paths = corpus["train_image_paths"]
    image_to_index = {path: index for index, path in enumerate(image_paths)}
    case_by_image = {}
    for record in annotations["train"]:
        case_id = str(record.get("id", record.get("study_id")))
        paths = record.get("image_path", record.get("images"))
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            case_by_image[str(path)] = case_id

    sentence_embeddings = encode_sentences(
        model,
        tokenizer,
        sentences,
        int(require(config, "pool", "encoding_batch_size")),
        device,
    )
    image_embeddings = encode_images(
        model,
        processor,
        image_root,
        image_paths,
        int(require(config, "pool", "encoding_batch_size")),
        device,
    )
    sentence_image_indices = [
        [image_to_index[path] for path in corpus["sentence_images"][sentence]]
        for sentence in sentences
    ]
    sentence_case_ids = [corpus["sentence_cases"][sentence] for sentence in sentences]
    memory_sentence_ids = [
        index
        for index, sentence in enumerate(sentences)
        if corpus["sentence_counts"][sentence] > int(require(config, "pool", "memory_frequency_threshold"))
    ]
    if not memory_sentence_ids:
        raise ValueError("no sentence exceeds memory frequency threshold")
    memory_status = memory_row_status(len(memory_sentence_ids), config["pool"].get("expected_memory_rows"))
    if memory_status["expected"] is None:
        print(f"memory_rows={memory_status['actual']}")
    else:
        print(f"memory_rows={memory_status['actual']} expected={memory_status['expected']} match={str(memory_status['match']).lower()}")
    centers, cluster_sentence_ids = build_index(
        sentence_embeddings,
        image_embeddings,
        sentence_image_indices,
        int(require(config, "pool", "clusters")),
        int(require(config, "pool", "cluster_batch_size")),
        float(require(config, "fallbacks", "soft_kmeans_temperature")),
        int(require(config, "fallbacks", "soft_kmeans_iterations")),
        int(require(config, "runtime", "seed")),
        device,
    )
    payload = {
        "version": 1,
        "source_split": "train",
        "sentences": sentences,
        "sentence_embeddings": sentence_embeddings,
        "image_embeddings": image_embeddings,
        "sentence_image_indices": sentence_image_indices,
        "sentence_case_ids": sentence_case_ids,
        "image_case_ids": [case_by_image[path] for path in image_paths],
        "image_paths": image_paths,
        "cluster_centers": centers,
        "cluster_sentence_ids": cluster_sentence_ids,
        "memory_sentence_ids": memory_sentence_ids,
        "expected_memory_rows": memory_status["expected"],
        "memory_rows_match": memory_status["match"],
        "train_case_ids": corpus["train_case_ids"],
        "train_image_paths": corpus["train_image_paths"],
        "train_report_fingerprints": corpus["train_report_fingerprints"],
        "model_name": model_name,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


if __name__ == "__main__":
    main()
