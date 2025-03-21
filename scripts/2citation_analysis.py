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
    return valid_citations

def extract_references(text):
    """Extract reference list while ensuring correct boundaries."""
    
    # Regex adjusted to avoid grabbing excess text
    references = re.findall(r'(\d+)\.\s+(.+?)(?=\n\d+\.\s|\Z)', text, re.DOTALL)

    reference_dict = {}
    for num, ref in references:
        ref = ref.strip()
        reference_dict[num] = ref

    # Extract the highest citation number detected
    max_citation_number = max(map(int, reference_dict.keys()), default=None)

    return reference_dict, max_citation_number

def classify_citation_medbert(text):
    """Run MedBERT classification synchronously."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    labels = ["ACCURATE", "NOT_ACCURATE", "IRRELEVANT"]
    return labels[prediction]

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
        return f"GPT-3.5: {gpt_label}"  
    return f"MedBERT: {medbert_label}"

import io  # Import for handling in-memory file writing

def main():
    st.title("📚 Citation Accuracy Analyzer")
    uploaded_file = st.file_uploader("Upload an academic paper (PDF)", type="pdf")
   
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)

        # **Display extracted text preview**
        st.write("### Extracted Text Preview")
        st.text_area("Extracted Text", text[:5000], height=300)  # Show first 5000 chars for readability

        # **Provide a Download Button for the Extracted Text**
        text_file = io.BytesIO(text.encode("utf-8"))
        st.download_button(label="📥 Download Extracted Text",
                           data=text_file,
                           file_name="extracted_text.txt",
                           mime="text/plain")

        references, max_citation_number = extract_references(text)
        citations = extract_citations(text, max_citation_number)

        st.write(f"**Extracted Citations:** {citations}")
        st.write(f"**Extracted References:** {references}")

        results = []
        if citations:
            for citation in citations:
                if citation in references:
                    reference_text = references[citation]
                    context_match = re.search(rf"([^.]*?\({citation}\)[^.]*\.)", text)
                    citation_context = context_match.group(1).strip() if context_match else "No context found."
                    classification = classify_citation(citation_context, reference_text)
                    results.append((citation_context, reference_text, classification))

        if results:
            st.write("### Citation Classification Results")
            for context, reference, label in results:
                st.write(f"📖 **Citation Sentence:** {context}")
                st.write(f"📄 **Referenced Paper:** {reference}")
                st.write(f"🔍 **Classification:** {label}")
                st.write("---")
        elif citations:
            st.write("⚠️ No citations classified. Check extracted citations or reference retrieval.")
        else:
            st.write("⚠️ No citations detected in the document. Try another file.")

if __name__ == "__main__":
    main()

