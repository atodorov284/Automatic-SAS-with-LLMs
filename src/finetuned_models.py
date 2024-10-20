import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer

def score_essay(essay_text, essay_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_path = f"best_models/best_model_{essay_set}/"
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    
    model.eval()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        essay_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_score = torch.argmax(logits, dim=-1).item()
    return predicted_score
