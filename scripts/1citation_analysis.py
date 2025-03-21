import streamlit as st
import fitz  # PyMuPDF for PDF processing
import re
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
import json
import os

# Load fine-tuned MedBERT from Hugging Face
model_name = "seloooselin/citation-analysis-medbert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)  # Ensure model is moved to the right device
    return tokenizer, model

tokenizer, model = load_model()

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file in Streamlit"""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_citations(text, max_citation_number=None):
    """Extract in-text citations in various formats."""
    
    # Improved regex to detect multiple citation styles
    citations = re.findall(r'\((\d+(?:[,-]\s*\d+)*)\)|\[(\d+(?:[,-]\s*\d+)*)\]', text)

    print("Raw extracted citations:", citations)  # Debugging output

    # Flatten citations by splitting multiple numbers in one citation
    all_citations = set()
    for citation_group in citations:
        for group in citation_group:
            if group:  # Ignore empty matches
                numbers = [num.strip() for num in re.split(r'[,-]\s*', group)]
                all_citations.update(numbers)

    # Convert to numbers and filter out invalid values
    valid_citations = {
        num for num in all_citations
        if num.isdigit() and 1 <= int(num) <= (max_citation_number if max_citation_number else 200)
    }

    print("Filtered citations:", valid_citations)  # Debugging output
    return valid_citations

def extract_references(text):
    """Extract reference list while ensuring correct boundaries."""
    
    # Regex adjusted to avoid grabbing excess text
    references = re.findall(r'(\d+)\.\s+(.+?)(?=\n\d+\.\s|\Z)', text, re.DOTALL)

    reference_dict = {}
    for num, ref in references:
        ref = ref.strip()
        if ref.endswith("-") and len(ref) < 100:  # Avoid split words
            ref += " " + text.split(num + ".")[-1].split("\n")[0].strip()

        reference_dict[num] = ref

    # Debugging output
    print("Extracted References:", reference_dict)

    # Extract the highest citation number detected
    max_citation_number = max(map(int, reference_dict.keys()), default=None)

    return reference_dict, max_citation_number

def search_pubmed(reference_text):
    """Search PubMed API for a paper using full reference title and return its abstract."""
    base_url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/"
    params = {"format": "json", "title": reference_text}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "abstract" in data:
            return data["abstract"]

    # Try searching with only the first sentence of reference text
    alt_query = reference_text.split(".")[0]  # Extract the first sentence
    params = {"format": "json", "title": alt_query}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "abstract" in data:
            return data["abstract"]

    return None

def classify_citation_medbert(text):
    """Run MedBERT classification synchronously."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    labels = ["ACCURATE", "NOT_ACCURATE", "IRRELEVANT"]
    return labels[prediction]

 



import openai

def classify_citation_gpt3(context, evidence):
    """Run GPT-3 few-shot classification using OpenAI API."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    You are assessing citations in biomedical articles for accuracy.

    A citation is classified as:
    - ACCURATE: The citation context aligns well with the reference evidence.
    - NOT_ACCURATE: The citation contradicts, misquotes, oversimplifies, or does not fully substantiate the reference evidence.
    - IRRELEVANT: The citation refers to content that is completely unrelated to the reference.

    Citation Context:
    "{context}"

    Reference Evidence:
    "{evidence}"

    Classification (Choose one: ACCURATE, NOT_ACCURATE, IRRELEVANT):
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a citation classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().upper()

def classify_citation(text, reference):
    """Hybrid classification approach with debug indicators."""
    medbert_label = classify_citation_medbert(text + " [SEP] " + reference)

    if medbert_label in ["NOT_ACCURATE", "IRRELEVANT"]:
        gpt_label = classify_citation_gpt3(text, reference)
        print(f"📌 GPT-3.5 Used for Citation: {text[:30]}...")  # Debugging output
        return f"GPT-3.5: {gpt_label}"  

    print(f"✅ MedBERT Used for Citation: {text[:30]}...")  # Debugging output
    return f"MedBERT: {medbert_label}"

import os
import torch

# Fix PyTorch async issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    
    torch.set_num_threads(1)  # Prevent PyTorch from using multiple CPU threads
    return tokenizer, model

def main():
    st.title("📚 Citation Accuracy Analyzer")
    uploaded_file = st.file_uploader("Upload an academic paper (PDF)", type="pdf")
   
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        references, max_citation_number = extract_references(text)

        # Debugging: Print max citation number
        st.write(f"**Max Citation Number Detected:** {max_citation_number}")

        citations = extract_citations(text, max_citation_number)

        st.write(f"**Extracted Citations:** {citations}")
        st.write(f"**Extracted References:** {references}")

        results = []

        if citations:
            for citation in citations:
                if citation in references:
                    reference_text = references[citation]
                    
                    # Extract the actual sentence where the citation appears
                    # Extract the actual sentence where the citation appears
                    context_match = re.search(rf"([^.]*?\({citation}\)[^.]*\.)", text)  # Extracts full sentence
                    citation_context = context_match.group(1).strip() if context_match else "No context found."

                    classification = classify_citation(citation_context, reference_text)
                    results.append((citation_context, reference_text, classification))

        if len(results) > 0:
            st.write("### Citation Classification Results")
            for context, reference, label in results:
                st.write(f"📖 **Citation Sentence:** {context}")
                st.write(f"📄 **Referenced Paper:** {reference}")
                st.write(f"🔍 **Classification:** {label}")
                st.write("---")  # Adds a separator
        elif citations:
            st.write("⚠️ No citations classified. Check extracted citations or reference retrieval.")
        else:
            st.write("⚠️ No citations detected in the document. Try another file.")

if __name__ == "__main__":
    main()
