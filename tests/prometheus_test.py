from simple_llms_eval import PrometheusScore


model_name = "meta-llama/Llama-3.1-8B-Instruct"


predictions = [
    "El sol brilla debido a reacciones de fusión nuclear en su núcleo.",
    "La capital de Francia es París."
]

references = [
    "El sol produce luz y calor mediante reacciones de fusión nuclear en su núcleo.",
    "París es la capital de Francia."
]


contexts = [
    "El Sol emite energía en forma de luz y calor, generada a través de reacciones de fusión nuclear en su núcleo.",
    "Francia es un país europeo cuya capital es París."
]


previous_conversations = [
    [
        {"role": "user", "content": "¿Qué es una estrella?"},
        {"role": "assistant", "content": "Una estrella es una esfera de plasma, principalmente hidrógeno y helio, que emite luz y calor debido a las reacciones de fusión nuclear en su núcleo."},
        {"role": "user", "content": "¿Por qué el Sol brilla?"}
    ],
    [
        {"role": "user", "content": "¿Qué es Francia?"},
        {"role": "assistant", "content": "Francia es un país europeo conocido por su historia, su cultura y su gastronomía. Su capital es París."},
        {"role": "user", "content": "¿Cuáles son algunos de los monumentos famosos de Francia?"}
    ]
]



def test_default():
    metric = PrometheusScore()
    scores = metric.compute(model_name=model_name, predictions=predictions, references=references, contexts=contexts, previous_conversations=previous_conversations)
    
    assert isinstance(scores, list)
    assert all(isinstance(score, int) for score in scores)



def test_no_previous_conversation():
    metric = PrometheusScore()
    scores = metric.compute(model_name=model_name, predictions=predictions, references=references, contexts=contexts)
    
    assert isinstance(scores, list)
    assert all(isinstance(score, int) for score in scores)



def test_return_average():
    metric = PrometheusScore()
    scores, avg_score = metric.compute(model_name=model_name, predictions=predictions, references=references, contexts=contexts, previous_conversations=previous_conversations, return_average=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, int) for score in scores)
    assert isinstance(avg_score, float)
    assert 1 <= avg_score <= 10



def test_return_feedbacks():
    metric = PrometheusScore()
    scores, feedbacks = metric.compute(model_name=model_name, predictions=predictions, references=references, contexts=contexts, previous_conversations=previous_conversations, return_feedbacks=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, int) for score in scores)
    assert isinstance(feedbacks, list)
    assert all(isinstance(feedback, str) for feedback in feedbacks)



def test_return_all():
    metric = PrometheusScore()
    scores, feedbacks, avg_score = metric.compute(model_name=model_name, predictions=predictions, references=references, contexts=contexts, previous_conversations=previous_conversations, return_feedbacks=True, return_average=True)

    assert isinstance(scores, list)
    assert all(isinstance(score, int) for score in scores)
    assert isinstance(feedbacks, list)
    assert all(isinstance(feedback, str) for feedback in feedbacks)
    assert isinstance(avg_score, float)
    assert 1 <= avg_score <= 10



if __name__ == "__main__":
    test_default()
    test_no_previous_conversation()
    test_return_average()
    test_return_feedbacks()
    test_return_all()

    print("All tests passed for Prometheus Score!")