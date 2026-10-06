from collections import Counter


def nlg_scores(references, predictions):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    reference_map = {index: [text] for index, text in enumerate(references)}
    prediction_map = {index: [text] for index, text in enumerate(predictions)}
    bleu, _ = Bleu(4).compute_score(reference_map, prediction_map)
    meteor, _ = Meteor().compute_score(reference_map, prediction_map)
    rouge, _ = Rouge().compute_score(reference_map, prediction_map)
    return {
        "BLEU_1": float(bleu[0]),
        "BLEU_2": float(bleu[1]),
        "BLEU_3": float(bleu[2]),
        "BLEU_4": float(bleu[3]),
        "METEOR": float(meteor),
        "ROUGE_L": float(rouge),
    }


def term_f1(reference_terms, prediction_terms):
    true_positive = 0
    predicted = 0
    reference = 0
    for expected, actual in zip(reference_terms, prediction_terms):
        expected_counts = Counter(expected)
        actual_counts = Counter(actual)
        true_positive += sum((expected_counts & actual_counts).values())
        predicted += sum(actual_counts.values())
        reference += sum(expected_counts.values())
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
