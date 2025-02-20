from simple_llms_eval import Accuracy


references = ["A", "B", "C"]
predictions = ["A", "D", "C"]


def test_default():
    metric = Accuracy()
    scores = metric.compute(predictions=predictions, references=references)

    assert isinstance(scores, list)
    assert all(isinstance(score, bool) for score in scores)



def test_return_average():
    metric = Accuracy()
    scores, avg_score = metric.compute(predictions=predictions, references=references, return_average=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, bool) for score in scores)
    assert isinstance(avg_score, float)
    assert 0 <= avg_score <= 1



if __name__ == "__main__":
    test_default()
    test_return_average()

    print("All tests passed for Accuracy!")
