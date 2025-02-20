from simple_llms_eval import SemanticAnswerSimilarity


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
    metric = SemanticAnswerSimilarity()
    scores = metric.compute(predictions=predictions, references=references)
    
    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)



def test_return_average():
    metric = SemanticAnswerSimilarity()
    scores, avg_score = metric.compute(predictions=predictions, references=references, return_average=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)
    assert isinstance(avg_score, float)
    assert 0 <= avg_score <= 1



def test_invalid_model():
    metric = SemanticAnswerSimilarity()
    scores = metric.compute(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",predictions=predictions, references=references)

    assert scores is None



if __name__ == "__main__":
    test_default()
    test_return_average()
    test_invalid_model()

    print("All tests passed for Semantic Answer Similarity!")

