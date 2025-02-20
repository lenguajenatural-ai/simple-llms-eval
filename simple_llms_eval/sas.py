import torch
import datasets
import evaluate
import numpy as np
from transformers import AutoConfig
from sentence_transformers import CrossEncoder



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
SAS utilizes a cross-encoder model where prediction and reference are joined together with a separator token.
The model then generates a similarity score ranging from 0 to 1.
"""



_KWARGS_DESCRIPTION = """
Compute Semantic Answer Similarity with a Cross Encoder.

Args:
    predictions: list of strings. The predicted answers or sentences.
    references: list of strings. The correct reference answers or sentences.
    model_name: string, optional (default="cross-encoder/stsb-roberta-large"). The name of the Cross Encoder model to be used.
    batch_size: int, optional (default=64). The batch size to use for embedding computation.
    return_average: bool, optional (default=False). If True, returns both the similarity scores and the average similarity score.

Returns:
    list of float or tuple of (list of float, float):
        - If return_average is False, returns a list of similarity scores between the prediction and reference pairs.
        - If return_average is True, returns a tuple containing the list of similarity scores and the average similarity score.

Examples:

    >>> references = ["El sol brilla en el cielo.", "Las bicicletas son ecológicas.", "El café es una bebida popular."]
    >>> predictions = ["El sol está en el cielo.", "Las bicicletas son buenas para el ambiente.", "El café es adictivo."]
    >>> metric = SemanticAnswerSimilarity()
    >>> scores, avg_score = metric.compute(predictions=predictions, references=references, batch_size=4)
    >>> print(scores)
    [0.8882, 0.5813, 0.6448]
    >>> print(avg_score)
    0.7048
"""



@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SemanticAnswerSimilarity(evaluate.Metric):
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
        predictions: list[str], 
        references: list[str],
        model_name: str = "cross-encoder/stsb-roberta-large",
        return_average: bool = False,
        batch_size: int = 64, 
    ) -> list[float] | tuple[list[float], float]:
        
        model_config = AutoConfig.from_pretrained(model_name)
        if model_config.architectures:
            is_cross_encoder = any(architectures.endswith("ForSequenceClassification") for architectures in model_config.architectures)

        if not is_cross_encoder:
            print(f"Invalid model architecture, {model_name} is not a cross-encoder.")
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(model_name, device=device)
        pairs = []
        
        for prediction, reference in zip(predictions, references):
            pairs.append([prediction, reference])
        
        scores = model.predict(pairs, batch_size=batch_size).tolist()
        
        if return_average:
            avg_score = float(np.mean(scores))
            return scores, avg_score
        
        return scores
