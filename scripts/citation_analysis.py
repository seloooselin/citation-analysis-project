import streamlit as st
import fitz  # PyMuPDF for PDF processing
import re
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load fine-tuned MedBERT from Hugging Face
model_name = "seloooselin/citation-analysis-medbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file in Streamlit"""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text("text") for page in doc])
    return text


def extract_citations(text):
    """Extract in-text citations (APA/IEEE formats)"""
    apa_citations = re.findall(r'\(([^)]+, \d{4})\)', text)  # (Author, Year)
    ieee_citations = re.findall(r'\[(\d+)\]', text)  # [12]
    return set(apa_citations + ieee_citations)

def extract_references(text):
    """Extract reference list (simplified regex for now)"""
    references = re.findall(r'(\d+\.\s+.+?\n)', text)  # IEEE style references
    return references

def search_pubmed(query):
    """Search PubMed API for a paper and return its abstract."""
    base_url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/"
    params = {"format": "json", "title": query}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "abstract" in data:
            return data["abstract"]
    return None

import asyncio

def classify_citation_medbert(text):
    """Run MedBERT classification synchronously."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    labels = ["ACCURATE", "NOT_ACCURATE", "IRRELEVANT"]
    return labels[prediction]



def classify_citation_gpt3(context, evidence):
    """Run GPT-3 few-shot classification."""
    prompt = f"""
    You are assessing citations in biomedical articles for accuracy.\n\nA citation is classified as:\n- ACCURATE: The citation context aligns well with the reference evidence.\n- NOT_ACCURATE: The citation contradicts, misquotes, oversimplifies, or does not fully substantiate the reference evidence.\n- IRRELEVANT: The citation refers to content that is completely unrelated to the reference.\n\nContext:\n{context}\n\nEvidence:\n{evidence}\n\nPrediction:
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip().upper()

def classify_citation(text, reference):
    """Hybrid classification approach without async conflicts."""
    medbert_label = classify_citation_medbert(text + " [SEP] " + reference)
    if medbert_label in ["NOT_ACCURATE", "IRRELEVANT"]:
        return classify_citation_gpt3(text, reference)
    return medbert_label


def main():
    st.title("📚 Citation Accuracy Analyzer")
    uploaded_file = st.file_uploader("Upload an academic paper (PDF)", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        citations = extract_citations(text)
        references = extract_references(text)
        st.write(f"**Extracted Citations:** {citations}")
        st.write(f"**Extracted References:** {references}")
        
        results = []
        

        async def process_citations(citations):
            results = []
            for citation in citations:
                reference_text = search_pubmed(citation)
                if reference_text:
                    classification = await classify_citation(citation, reference_text)
                    results.append((citation, classification))
            return results

        if citations:
            results = asyncio.run(process_citations(citations))


        st.write("### Citation Classification Results")
        for citation, label in results:
            st.write(f"📖 {citation} → **{label}**")

if __name__ == "__main__":
    main()
