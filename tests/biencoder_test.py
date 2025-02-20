from simple_llms_eval import BiEncoderScore


references = [
    "El sol brilla en el cielo.",
    "Las bicicletas son ecológicas.",
    "El café es una bebida popular."
]


predictions = [
    "El sol está en el cielo.",
    "Las bicicletas son buenas para el ambiente.",
    "El café es adictivo."
]


def test_default():
    metric = BiEncoderScore()
    scores = metric.compute(predictions=predictions, references=references)
    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)



def test_return_average():
    metric = BiEncoderScore()
    scores, avg_score = metric.compute(predictions=predictions, references=references, return_average=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)
    assert isinstance(avg_score, float)
    assert -1 <= avg_score <= 1



if __name__ == "__main__":
    test_default()
    test_return_average()

    print("All tests passed for Bi-Encoder!")
