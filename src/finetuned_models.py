import torch
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, BertTokenizer

login(token="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")


def pull_model_from_huggingface(essay_set):
    model_id = f"elisaklunder/finetuned_bert_for_asap_sas_essayset_{essay_set}"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model


def score_essay(essay_text, essay_set):
    model_id = f"elisaklunder/finetuned_bert_for_asap_sas_essayset_{essay_set}"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        essay_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_score = torch.argmax(logits, dim=-1).item()

    print(f"Predicted score for the essay: {predicted_score}")
    return predicted_score
