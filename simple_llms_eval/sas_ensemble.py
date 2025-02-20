import torch
import datasets
import evaluate
import numpy as np
from transformers import AutoConfig
from .sas import SemanticAnswerSimilarity
from .biencoder import BiEncoderScore



_CITATION = """\
@misc{risch2021semantic,
      title={Semantic Answer Similarity for Evaluating Question Answering Models},
      author={Julian Risch and Timo Möller and Julian Gutsch and Malte Pietsch},
      year={2021},
      eprint={2108.06130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""



_DESCRIPTION = """\
Ensemble of Semantic Answer Similarity and BiEncoder metrics.
Improves assessment by applying cross-encoder and bi-encoder models and averages their results.
"""



_KWARGS_DESCRIPTION = """
Compute Ensemble of Semantic Answer Similarity and BiEncoder metric.

Args:
    predictions: list of strings. The predicted answers or sentences.
    references: list of strings. The correct reference answers or sentences.
    model_names: list of strings. The list of cross-encoder and/or bi-encoder model names or paths to be used for ensemble evaluation.
    batch_size: int, optional (default=64). The batch size to use for embedding computation.
    return_average: bool, optional (default=False). If True, returns both the individual model scores and the average score.

Returns:
    list of float or tuple of (list of float, float):
        - If return_average is False, returns a list of average similarity scores from the ensemble models.
        - If return_average is True, returns a tuple containing the list of similarity scores and the average similarity score.

Examples:

    >>> references = ["El sol brilla en el cielo.", "Las bicicletas son ecológicas.", "El café es una bebida popular."]
    >>> predictions = ["El sol está en el cielo.", "Las bicicletas son buenas para el ambiente.", "El café es adictivo."]
    >>> models = ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "cross-encoder/stsb-roberta-large"]
    >>> metric = SASEnsemble()
    >>> scores, avg_score = metric.compute(models=models, predictions=predictions, references=references, batch_size=4)
    >>> print(scores)
    [0.9336, 0.7649, 0.7458]
    >>> print(avg_score)
    0.8148
"""



@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SASEnsemble(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value(dtype="string"),
                    "references": datasets.Value(dtype="string"),
                },
            ),
            reference_urls=["https://arxiv.org/abs/2108.06130"],
        )


    def _compute(
        self, 
        model_names: list[str], 
        predictions: list[str], 
        references: list[str],
        return_average: bool = False,
        batch_size: int = 64
    ) -> list[float] | tuple[list[float], float]:
        
        metric_scores = []

        for model_name in model_names:
            model_config = AutoConfig.from_pretrained(model_name)

            if model_config.architectures:
                is_cross_encoder = any(architectures.endswith("ForSequenceClassification") for architectures in model_config.architectures)

            if is_cross_encoder:
                metric = SemanticAnswerSimilarity(model_name=model_name)
            else:
                metric = BiEncoderScore(model_name=model_name)

            scores = metric.compute(predictions=predictions, references=references, batch_size=batch_size)
            metric_scores.append(scores)
        
        scores = np.mean(np.array(metric_scores), axis=0).tolist()
        
        if return_average:
            avg_score = float(np.mean(scores))
            return scores, avg_score
        
        return scores
