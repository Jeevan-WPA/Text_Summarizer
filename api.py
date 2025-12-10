from fastapi import FastAPI
from pydantic import BaseModel
from extractive import TextRankSummarizer
from abstractive import AbstractiveSummarizer
from gpt_summarizer import GptSummarizer
from rouge_score import rouge_scorer

app = FastAPI()

# Load both models ONCE when the server starts
extractive_model = TextRankSummarizer()
abstractive_model = AbstractiveSummarizer()
gpt_model = GptSummarizer()

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

    elif req.mode == "llm":
        result = gpt_model.summarize(req.text)
    
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
    
    gpt_sum = gpt_model.summarize(req.text)

    # Compute ROUGE scores (extractive vs GPT and abstractive vs GPT)
    extractive_score = scorer.score(target=gpt_sum, prediction=extractive_sum)
    abstractive_score = scorer.score(target=gpt_sum, prediction=abstractive_sum)
    
    # Format output
    e_rouge_output = {
        "rouge1": round(extractive_score["rouge1"].fmeasure, 4),
        "rouge2": round(extractive_score["rouge2"].fmeasure, 4),
        "rougeL": round(extractive_score["rougeL"].fmeasure, 4)
    }

    a_rouge_output = {
        "rouge1": round(abstractive_score["rouge1"].fmeasure, 4),
        "rouge2": round(abstractive_score["rouge2"].fmeasure, 4),
        "rougeL": round(abstractive_score["rougeL"].fmeasure, 4)
    }
    
    return {
        "extractive_summary": extractive_sum,
        "abstractive_summary": abstractive_sum,
        "gpt_summary": gpt_sum,
        "extractive_rouge_scores": e_rouge_output,
        "abstractive_rouge_scores": a_rouge_output
    }
