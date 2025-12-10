import streamlit as st
from api_client import summarize, compare

st.set_page_config(page_title="AI Summarizer", layout="wide")

st.title("📝 AI Summarization Engine")
st.write("Extractive,Abstractive and LLM Based Summary Generator (FastAPI + Streamlit)")

# Text input
text = st.text_area("Enter your text:", height=250)

# Mode selection
mode = st.selectbox(
    "Choose summarization mode:",
    ["extractive", "abstractive","llm"]
)

# Generate summary
if st.button("Generate Summary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize(text, mode)
        st.success("Summary generated!")
        st.text_area("Summary Output", summary, height=200)

# Divider
st.markdown("---")

# Comparison section
st.subheader("🔍 Compare Extractive vs Abstractive (ROUGE Scores)")
if st.button("Compare Models"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Comparing..."):
            results = compare(text)

        st.write("### Extractive Summary")
        st.write(results["extractive_summary"])

        st.write("### Abstractive Summary")
        st.write(results["abstractive_summary"])
        
        st.write("### GPT Summary")
        st.write(results["gpt_summary"])
        
        st.write("### Extractive ROUGE Scores")
        st.json(results["extractive_rouge_scores"])
        
        st.write("### Abstractive ROUGE Scores")
        st.json(results["abstractive_rouge_scores"])
