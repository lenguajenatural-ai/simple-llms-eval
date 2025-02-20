import torch
import datasets
import evaluate
import numpy as np
from torch.nn import CosineSimilarity
from sentence_transformers import SentenceTransformer



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
The BiEncoder score computes the cosine similarity (ranges from -1  to 1) between prediction and reference
after encoding them with a sentence transformer model. 
"""



_KWARGS_DESCRIPTION = """
Computes BiEncoder Score.

Args:
    predictions: list of strings. The predicted text sequences.
    references: list of strings. The reference text sequences.
    model_name: string, optional (default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"). 
        The name of the Sentence Transformer model to be used.
    batch_size: int, optional (default=64). The batch size to use for embedding computation.
    return_average: bool, optional (default=False). If True, returns both the individual similarity scores and the average score.

Returns:
    list of float or tuple of (list of float, float):
        - If return_average is False, returns a list of similarity scores (Cosine similarity) between the prediction and reference pairs.
        - If return_average is True, returns a tuple containing the list of similarity scores and the average similarity score.

Examples:

    >>> references = ["El sol brilla en el cielo.", "Las bicicletas son ecológicas.", "El café es una bebida popular."]
    >>> predictions = ["El sol está en el cielo.", "Las bicicletas son buenas para el ambiente.", "El café es adictivo."]
    >>> metric = BiEncoderScore()
    >>> scores, avg_score = metric.compute(predictions=predictions, references=references, batch_size=4)
    >>> print(scores)
    [0.9791, 0.9485, 0.8468]
    >>> print(avg_score)
    0.9248
"""



@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BiEncoderScore(evaluate.Metric):
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
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        return_average: bool = False,
        batch_size: int = 64, 
    ) -> list[float] | tuple[list[float], float]:        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = CosineSimilarity(dim=1)
        model = SentenceTransformer(model_name, device=device)

        predictions_embeddings = model.encode(predictions, batch_size=batch_size, convert_to_tensor=True)
        references_embeddings = model.encode(references, batch_size=batch_size, convert_to_tensor=True)
        scores = metric(predictions_embeddings, references_embeddings).tolist()
        
        if return_average:
            avg_score = float(np.mean(scores))
            return scores, avg_score

        return scores

