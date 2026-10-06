import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

from .corpus import record_id, record_images, record_report, report_fingerprint


@dataclass(frozen=True)
class Candidate:
    sentence_id: int
    sentence: str
    score: float
    visual_score: float
    text_score: float
    source_case_ids: tuple


class SentencePool:
    def __init__(self, payload):
        self.sentences = payload["sentences"]
        self.sentence_embeddings = payload["sentence_embeddings"].float()
        self.image_embeddings = payload["image_embeddings"].float()
        self.sentence_image_indices = payload["sentence_image_indices"]
        self.sentence_case_ids = payload["sentence_case_ids"]
        self.image_case_ids = payload["image_case_ids"]
        self.image_paths = payload["image_paths"]
        self.cluster_centers = payload["cluster_centers"].float()
        self.cluster_sentence_ids = payload["cluster_sentence_ids"]
        self.train_case_ids = set(payload["train_case_ids"])
        self.train_image_paths = set(payload["train_image_paths"])
        self.source_split = payload["source_split"]
        self.memory_sentence_ids = payload["memory_sentence_ids"]
        self.train_report_fingerprints = payload.get("train_report_fingerprints")
        self.sentence_to_id = {sentence: index for index, sentence in enumerate(self.sentences)}
        self._validate()

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(payload)

    def _validate(self):
        for values in (self.sentence_embeddings, self.image_embeddings, self.cluster_centers):
            if not torch.isfinite(values).all():
                raise ValueError("sentence pool contains non-finite embeddings")
        if self.source_split != "train":
            raise ValueError("sentence pool must originate from train")
        if len(self.sentences) != self.sentence_embeddings.shape[0]:
            raise ValueError("sentence embedding count mismatch")
        if len(self.sentence_image_indices) != len(self.sentences):
            raise ValueError("sentence image mapping count mismatch")
        if len(self.sentence_case_ids) != len(self.sentences):
            raise ValueError("sentence case mapping count mismatch")
        if len(self.image_case_ids) != self.image_embeddings.shape[0]:
            raise ValueError("image provenance count mismatch")
        if len(self.image_paths) != self.image_embeddings.shape[0]:
            raise ValueError("image path count mismatch")
        if len(self.cluster_sentence_ids) != self.cluster_centers.shape[0]:
            raise ValueError("cluster mapping count mismatch")
        if self.sentence_embeddings.shape[1] != self.image_embeddings.shape[2]:
            raise ValueError("sentence and image dimensions differ")
        expected_index_dimension = self.image_embeddings.shape[1] * self.image_embeddings.shape[2] + self.sentence_embeddings.shape[1]
        if self.cluster_centers.shape[1] != expected_index_dimension:
            raise ValueError("cluster center dimension mismatch")
        if not set(self.image_case_ids) <= self.train_case_ids:
            raise ValueError("non-train case in sentence pool")
        if not set(self.image_paths) <= self.train_image_paths:
            raise ValueError("non-train image in sentence pool")
        for case_ids in self.sentence_case_ids:
            if not set(case_ids) <= self.train_case_ids:
                raise ValueError("non-train sentence provenance")
        sentence_ids = set(range(len(self.sentences)))
        image_ids = set(range(len(self.image_paths)))
        if not set(self.memory_sentence_ids) <= sentence_ids:
            raise ValueError("invalid memory sentence id")
        for sentence_id, indices in enumerate(self.sentence_image_indices):
            if not set(indices) <= image_ids:
                raise ValueError("invalid sentence image id")
            associated_cases = {self.image_case_ids[index] for index in indices}
            if not associated_cases <= set(self.sentence_case_ids[sentence_id]):
                raise ValueError("sentence association provenance mismatch")
        for cluster in self.cluster_sentence_ids:
            if not set(cluster) <= sentence_ids:
                raise ValueError("invalid cluster sentence id")

    def assert_annotation_source(self, annotations):
        train_cases = {record_id(record) for record in annotations["train"]}
        train_images = {image for record in annotations["train"] for image in record_images(record)}
        prohibited_cases = {record_id(record) for split in ("val", "test") for record in annotations[split]}
        prohibited_images = {image for split in ("val", "test") for record in annotations[split] for image in record_images(record)}
        if self.train_case_ids != train_cases:
            raise ValueError("pool case manifest differs from train split")
        if self.train_image_paths != train_images:
            raise ValueError("pool image manifest differs from train split")
        if self.train_case_ids & prohibited_cases:
            raise ValueError("non-train case in pool manifest")
        if self.train_image_paths & prohibited_images:
            raise ValueError("non-train image in pool manifest")
        fingerprints = {record_id(record): report_fingerprint(record_report(record)) for record in annotations["train"]}
        if self.train_report_fingerprints != fingerprints:
            raise ValueError("training reports differ from the pool; rebuild the sentence pool")
        return True

    @property
    def memory_embeddings(self):
        indices = torch.tensor(self.memory_sentence_ids, dtype=torch.long)
        return self.sentence_embeddings.index_select(0, indices)

    def _allowed_images(self, sentence_id, query_case_id, query_image_paths, exclude_query):
        if not exclude_query:
            return list(self.sentence_image_indices[sentence_id])
        blocked_paths = set(query_image_paths)
        allowed = []
        for image_index in self.sentence_image_indices[sentence_id]:
            if self.image_case_ids[image_index] == query_case_id:
                continue
            if self.image_paths[image_index] in blocked_paths:
                continue
            allowed.append(image_index)
        return allowed

    def _candidate_sentence_ids(self, enhanced_visual, index_clusters):
        mean_visual = enhanced_visual.mean(dim=0)
        query = torch.cat((enhanced_visual.flatten(), mean_visual), dim=0)
        centers = self.cluster_centers.to(query.device)
        if centers.shape[1] != query.shape[0]:
            raise ValueError("cluster index dimension mismatch")
        if index_clusters < 1 or index_clusters > centers.shape[0]:
            raise ValueError("invalid number of index clusters")
        populated = [index for index, sentence_ids in enumerate(self.cluster_sentence_ids) if sentence_ids]
        if index_clusters > len(populated):
            raise ValueError("index clusters exceed populated clusters")
        populated_tensor = torch.tensor(populated, dtype=torch.long, device=query.device)
        distances = torch.sum((centers.index_select(0, populated_tensor) - query.unsqueeze(0)) ** 2, dim=1)
        local_ids = torch.topk(distances, index_clusters, largest=False).indices.tolist()
        cluster_ids = [populated[index] for index in local_ids]
        return sorted({sentence_id for cluster_id in cluster_ids for sentence_id in self.cluster_sentence_ids[cluster_id]})

    def score_sentence(self, enhanced_visual, sentence_id, query_case_id, query_image_paths, gamma_r, gamma_t, exclude_query):
        allowed = self._allowed_images(sentence_id, query_case_id, query_image_paths, exclude_query)
        if not allowed:
            return None
        images = self.image_embeddings[allowed].to(enhanced_visual.device)
        if images.shape[1:] != enhanced_visual.shape:
            raise ValueError("image representation shape mismatch")
        visual_score = torch.sum((images - enhanced_visual.unsqueeze(0)) ** 2, dim=(1, 2)).min()
        mean_visual = enhanced_visual.mean(dim=0)
        text = self.sentence_embeddings[sentence_id].to(enhanced_visual.device)
        text_score = torch.sum((text - mean_visual) ** 2)
        score = gamma_r * visual_score + gamma_t * text_score
        return score, visual_score, text_score

    def training_loss(self, enhanced_visual, gold_sentences, query_case_id, query_image_paths, gamma_r=0.5, gamma_t=0.5):
        scores = []
        for sentence in gold_sentences:
            sentence_id = self.sentence_to_id.get(sentence)
            if sentence_id is None:
                continue
            result = self.score_sentence(
                enhanced_visual,
                sentence_id,
                query_case_id,
                query_image_paths,
                gamma_r,
                gamma_t,
                False,
            )
            if result is not None:
                scores.append(result[0])
        if not scores:
            return enhanced_visual.sum() * 0.0, 0
        return torch.stack(scores).mean(), len(scores)

    def extract(self, enhanced_visual, query_case_id, query_image_paths, query_split, top_k=5, gamma_r=0.5, gamma_t=0.5, redundancy_threshold=0.8, seed=0, index_clusters=1):
        if query_split not in {"train", "val", "test"}:
            raise ValueError("invalid query split")
        if query_split != "train" and query_case_id in self.train_case_ids:
            raise ValueError("evaluation case is present in training pool")
        if set(query_image_paths) & self.train_image_paths and query_split != "train":
            raise ValueError("evaluation image is present in training pool")
        candidates = []
        for sentence_id in self._candidate_sentence_ids(enhanced_visual, index_clusters):
            result = self.score_sentence(
                enhanced_visual,
                sentence_id,
                query_case_id,
                query_image_paths,
                gamma_r,
                gamma_t,
                True,
            )
            if result is None:
                continue
            score, visual_score, text_score = result
            source_cases = tuple(case_id for case_id in self.sentence_case_ids[sentence_id] if case_id != query_case_id)
            candidates.append(
                Candidate(
                    sentence_id=sentence_id,
                    sentence=self.sentences[sentence_id],
                    score=float(score.detach().cpu()),
                    visual_score=float(visual_score.detach().cpu()),
                    text_score=float(text_score.detach().cpu()),
                    source_case_ids=source_cases,
                )
            )
        candidates.sort(key=lambda candidate: (candidate.score, candidate.sentence_id))
        selected = candidates[:top_k]
        return filter_candidates(selected, self.sentence_embeddings, redundancy_threshold, seed)


def filter_candidates(candidates, sentence_embeddings, threshold=0.8, seed=0):
    rng = random.Random(seed)
    active = set(range(len(candidates)))
    normalized = functional.normalize(sentence_embeddings.float(), dim=1)
    for left in range(len(candidates)):
        if left not in active:
            continue
        for right in range(left + 1, len(candidates)):
            if right not in active:
                continue
            left_id = candidates[left].sentence_id
            right_id = candidates[right].sentence_id
            similarity = torch.dot(normalized[left_id], normalized[right_id]).item()
            if similarity > threshold:
                removed = rng.choice((left, right))
                active.remove(removed)
                if removed == left:
                    break
    return [candidates[index] for index in sorted(active, key=lambda index: (candidates[index].score, candidates[index].sentence_id))]


def assemble_report(candidates):
    return " ".join(candidate.sentence for candidate in candidates)
