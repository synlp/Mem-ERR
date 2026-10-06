import argparse
import json
from pathlib import Path

from memerr.corpus import read_annotations, record_id, record_report
from memerr.metrics import nlg_scores, term_f1


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--chexpert-reference-terms")
    parser.add_argument("--chexpert-prediction-terms")
    parser.add_argument("--radgraph-reference-terms")
    parser.add_argument("--radgraph-prediction-terms")
    args = parser.parse_args()

    annotations = read_annotations(args.annotations)
    predictions = load_json(args.predictions)
    prediction_by_id = {str(item["id"]): item["prediction"] for item in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("duplicate prediction ids")
    ids = [record_id(record) for record in annotations[args.split]]
    if set(ids) != set(prediction_by_id):
        raise ValueError("prediction ids do not match evaluation split")
    references = [record_report(record) for record in annotations[args.split]]
    outputs = [prediction_by_id[case_id] for case_id in ids]
    scores = nlg_scores(references, outputs)
    optional = (
        ("CE", args.chexpert_reference_terms, args.chexpert_prediction_terms),
        ("RadGraph", args.radgraph_reference_terms, args.radgraph_prediction_terms),
    )
    for name, reference_path, prediction_path in optional:
        if bool(reference_path) != bool(prediction_path):
            raise ValueError(f"both {name} term files are required")
        if reference_path:
            reference_terms = load_json(reference_path)
            prediction_terms = load_json(prediction_path)
            if set(reference_terms) != set(ids) or set(prediction_terms) != set(ids):
                raise ValueError(f"{name} ids do not match evaluation split")
            scores[name] = term_f1(
                [reference_terms[case_id] for case_id in ids],
                [prediction_terms[case_id] for case_id in ids],
            )
    print(json.dumps(scores, indent=2, sort_keys=True))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
