# Extractive Radiology Reporting With Memory-Based Cross-Modal Representations

This repository contains the implementation of [Extractive Radiology Reporting With Memory-Based Cross-Modal Representations](https://doi.org/10.1109/TMI.2025.3636868), published in IEEE Transactions on Medical Imaging, 45(4):1686–1697, 2026.

## Requirements

Python 3.10 or later and a Java runtime for METEOR are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The foundation model is [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32). Transformers obtains its model and processor resources. [TorchXRayVision](https://github.com/mlmed/torchxrayvision) obtains the anatomical PSPNet weights for segmentation.

## Data

- **IU X-Ray:** obtain the Indiana University chest X-ray PNG images and XML reports through [NLM Open-i's collection download instructions](https://openi-vip.nlm.nih.gov/faq). Associate report IDs and image paths using the XML image tags, extract the findings section, and exclude studies with empty findings. Preserve the study-level train/validation/test manifest used for the paper, whose reported sizes are approximately 2.8K/0.4K/0.6K. Convert this manifest into the annotation schema below. Place it at `data/iu_xray/annotation.json` and images at `data/iu_xray/images`, or change the paths in `configs/iu_xray.yaml`.
- **MIMIC-CXR:** obtain credentialed access through [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) and its [JPG distribution](https://physionet.org/content/mimic-cxr-jpg/2.1.0/). Complete the provider's training and data-use agreement. Join report findings and JPG image paths by study ID, filter empty findings, and retain the paper's study-level split manifest. Convert the result into the same annotation schema. A copy of the IU configuration can point to these local files with `data.dataset: mimic_cxr`, 2,000 pool clusters, and `pool.expected_memory_rows: 115974`.

Annotations are a JSON object with `train`, `val`, and `test` lists. Each study record contains a unique `id`, an `image_path` list relative to the image root, and findings text in `findings` or `report`. Case IDs and image paths must be disjoint across splits. `data.view_index` selects the radiograph path. The segmented root mirrors the original image paths. Sentence-pool construction uses training reports and images; the checkpoint is bound to that pool.

## Run

Set `EPOCHS` to the desired training duration; the paper leaves this value unspecified.

```bash
python run.py --config configs/iu_xray.yaml --epochs "$EPOCHS" --output-dir outputs/iu_xray
```

This command generates anatomical overlays, builds the CLIP sentence-image pool and soft K-means index, trains the fused visual encoders and memory, extracts test reports, and writes metrics. It reads all dataset paths and the segmentation threshold from the selected configuration.

For extraction from an existing trained checkpoint:

```bash
python infer.py --config configs/iu_xray.yaml --pool outputs/iu_xray/pool.pt --checkpoint outputs/iu_xray/checkpoint.pt --split test --output outputs/iu_xray/predictions.json
```

The default IU setting uses 300 index clusters, sentences occurring more than five times for memory initialization, visual/text score weights `0.5/0.5`, five candidates, redundancy threshold `0.8`, a 12-layer fusion Transformer of width 512, Adam at `5e-5`, batch size 16, weight decay `0.01`, and linear learning-rate decay. The configuration exposes the implementation's attention, segmentation, indexing, and view-selection settings.

## Evaluation

```bash
python evaluate.py --annotations data/iu_xray/annotation.json --predictions outputs/iu_xray/predictions.json --split test --output outputs/iu_xray/metrics.json
```

Evaluation reports BLEU-1 through BLEU-4, METEOR, and ROUGE-L. For CE on MIMIC-CXR and RadGraph term matching, run the [official CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler) and [RadGraph tools](https://github.com/Stanford-AIMI/radgraph) separately on predictions and findings. Convert their term annotations to JSON objects mapping study IDs to term lists, retaining label states in term identifiers. Pass paired files with `--chexpert-reference-terms`, `--chexpert-prediction-terms`, `--radgraph-reference-terms`, and `--radgraph-prediction-terms`. The evaluator computes corpus-level term precision, recall, and F1.

## Structure

`segment.py` produces aligned anatomical overlays. `build_pool.py` constructs training sentence representations and the embedding index. `memerr/model.py` implements visual fusion and memory retrieval. `memerr/extraction.py` implements scoring, candidate selection, and redundancy filtering. `run.py` runs the approach with the selected configuration.

## Citation

If you use this code, please cite the paper.

```bibtex
@article{tian2026extractive,
  title={Extractive Radiology Reporting With Memory-Based Cross-Modal Representations},
  author={Tian, Yuanhe and Yan, Zexuan and Lyu, Nenan and Song, Yan},
  journal={IEEE Transactions on Medical Imaging},
  volume={45},
  number={4},
  pages={1686--1697},
  year={2026}
}
```
