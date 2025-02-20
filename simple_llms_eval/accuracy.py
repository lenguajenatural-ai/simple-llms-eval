import datasets
import evaluate
import numpy as np



_CITATION = ""



_DESCRIPTION = """
This metric computes accuracy for multiple-choice or test-type tasks by comparing the predicted answers to the reference answers.
It checks for exact equality between the predictions and references after converting both to lowercase and stripping extra spaces.
"""



_KWARGS_DESCRIPTION = """
Compute the accuracy for multiple-choice or test-type tasks.

Args:
    predictions: list of strings. The predicted answers.
    references: list of strings. The correct reference answers.
    return_average: bool, optional (default=False). If True, returns the accuracy score along with the individual comparison results.

Returns:
    list of bool or tuple of (list of bool, float): 
        - If return_average is False, returns a list of boolean values indicating whether each prediction matches the reference.
        - If return_average is True, returns a tuple containing the list of boolean values and the average accuracy score.

Examples:

    >>> references = ["A", "B", "C"]
    >>> predictions = ["A", "D", "C"]
    >>> metric = Accuracy()
    >>> scores, avg_score = metric.compute(predictions=predictions, references=references, return_average=True)
    >>> print(scores)
    [True, False, True]
    >>> print(avg_score)
    0.6667
"""



class Accuracy(evaluate.Metric):
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
        )


    def _compute(
        self, 
        predictions: list[str], 
        references: list[str], 
        return_average: bool = False
    ) -> list[bool] | tuple[list[bool], float]:
        
        scores = [
            prediction.lower().strip() == reference.lower().strip()
            for prediction, reference in zip(predictions, references)
        ]
        
        if return_average:
            avg_score = float(np.mean(scores))
            return scores, avg_score
        
        return scores