from simple_llms_eval import SASEnsemble


model_names = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
    "cross-encoder/stsb-roberta-large"
]


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
    metric = SASEnsemble()
    scores = metric.compute(model_names=model_names, predictions=predictions, references=references)
    
    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)



def test_return_average():
    metric = SASEnsemble()
    scores, avg_score = metric.compute(model_names=model_names, predictions=predictions, references=references, return_average=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, float) for score in scores)
    assert isinstance(avg_score, float)
    assert -1 <= avg_score <= 1



if __name__ == "__main__":
    test_default()
    test_return_average()

    print("All tests passed for SAS Ensemble!")
