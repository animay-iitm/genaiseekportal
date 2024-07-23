from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

app = FastAPI()

tokenizer_classification = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model_classification = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

tokenizer_generation = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model_generation = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")


class CodeSnippet(BaseModel):
    code: str


class_labels = ["Poor quality code", "Good quality code"]


@app.post("/evaluate_code")
async def evaluate_code(snippet: CodeSnippet):
    try:
        # Classification
        inputs_classification = tokenizer_classification(snippet.code, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs_classification = model_classification(**inputs_classification)
        logits = outputs_classification.logits
        predictions = torch.softmax(logits, dim=-1).tolist()[0]

        # Interpret the results
        evaluation = {
            class_labels[0]: predictions[0],
            class_labels[1]: predictions[1]
        }

        most_likely_class = class_labels[predictions.index(max(predictions))]

        inputs_generation = tokenizer_generation(snippet.code, return_tensors="pt")
        suggestions = model_generation.generate(**inputs_generation, max_length=100, num_return_sequences=1)
        suggestion = tokenizer_generation.decode(suggestions[0], skip_special_tokens=True)

        return {
            "evaluation": evaluation,
            "most_likely_class": most_likely_class,
            "suggestion": suggestion
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with: uvicorn main:app --reload
