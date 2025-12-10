from fastapi import FastAPI
from pydantic import BaseModel
from extractive import TextRankSummarizer
from abstractive import AbstractiveSummarizer
from rouge_score import rouge_scorer

app = FastAPI()

# Load both models ONCE when the server starts
extractive_model = TextRankSummarizer()
abstractive_model = AbstractiveSummarizer()

class SummaryRequest(BaseModel):
    text: str
    mode: str = "extractive"

class ComparisonRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(req: SummaryRequest):
    if req.mode == "extractive":
        result = extractive_model.summarize(req.text)

    elif req.mode == "abstractive":
        result = abstractive_model.summarize(req.text)

    else:
        return {"error": "Invalid mode. Use 'extractive' or 'abstractive'."}

    return {"summary": result}

@app.post("/compare")
def compare(req: ComparisonRequest):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], 
        use_stemmer=True
    )

    # Generate both summaries
    extractive_sum = extractive_model.summarize(req.text)
    abstractive_sum = abstractive_model.summarize(req.text)

    # Compute ROUGE scores (extractive vs abstractive)
    scores = scorer.score(target=abstractive_sum, prediction=extractive_sum)

    # Format output
    rouge_output = {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4)
    }

    return {
        "extractive_summary": extractive_sum,
        "abstractive_summary": abstractive_sum,
        "rouge_scores": rouge_output
    }
