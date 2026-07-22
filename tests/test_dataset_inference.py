from backend.app import infer_dataset_from_query


def test_infers_scientific_questions_to_scifact():
    assert infer_dataset_from_query("What is the effect of vitamin D on bone density?") == "scifact"


def test_infers_medical_questions_to_nfcorpus():
    assert infer_dataset_from_query("How does diet influence type 2 diabetes?") == "nfcorpus"


def test_infers_argumentative_questions_to_arguana():
    assert infer_dataset_from_query("Should social media be regulated?") == "arguana"


def test_infers_finance_questions_to_fiqa():
    assert infer_dataset_from_query("How should I invest in index funds?") == "fiqa"
