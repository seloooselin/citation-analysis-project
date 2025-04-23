# citation_analysis.py

import streamlit as st
import re
import streamlit as st




st.set_page_config(page_title="Citation Classifier", layout="wide")

st.title("PubMed Citation Quality Analysis")

# Step 1: User input
user_input = st.text_input("Enter a PubMed article URL, DOI, or PMID:")

def extract_pmid(text):
    """Extract PMID from input (URL, DOI, or plain ID)."""
    # Example PubMed URL: https://pubmed.ncbi.nlm.nih.gov/12345678/
    pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', text)
    if pmid_match:
        return pmid_match.group(1)
    
    # If just a number (PMID)
    if text.strip().isdigit():
        return text.strip()
    
    # Example DOI to PMID (for later, optional)
    return None

if user_input:
    pmid = extract_pmid(user_input)
    if pmid:
        st.success(f"Detected PMID: {pmid}")
        # Proceed to fetch article next step
    else:
        st.error("Could not extract a valid PubMed ID from the input.")

from Bio import Entrez
import xml.etree.ElementTree as ET

# Always set your email (NCBI requirement)
Entrez.email = "selindenizisken@gmail.com"

def fetch_pubmed_article(pmid):
    """Fetch abstract and references from PubMed."""
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
        records = Entrez.read(handle)
        article = records['PubmedArticle'][0]

        # Extract title
        title = article['MedlineCitation']['Article']['ArticleTitle']

        # Extract abstract
        abstract_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
        abstract = " ".join(abstract_list)

        # Extract references (some may not have any)
        references = []
        try:
            ref_list = article['PubmedData']['ReferenceList'][0]['Reference']
            for ref in ref_list:
                citation = ref.get('Citation', 'No citation available')
                article_ids = ref.get('ArticleIdList', [])
                pmid = None
                for aid in article_ids:
                    if aid.attributes.get('IdType') == 'pubmed':
                        pmid = aid
                references.append({"citation": citation, "pmid": str(pmid) if pmid else None})
        except Exception:
            pass  # No references available

        return {
            "title": title,
            "abstract": abstract,
            "references": references
        }

    except Exception as e:
        st.error(f"Failed to fetch article: {e}")
        return None
if user_input:
    pmid = extract_pmid(user_input)
    if pmid:
        st.success(f"Detected PMID: {pmid}")
        article_data = fetch_pubmed_article(pmid)

        if article_data:
            st.subheader("Article Title")
            st.write(article_data['title'])

            st.subheader("Abstract")
            st.write(article_data['abstract'])

            st.subheader("References")
            for i, ref in enumerate(article_data['references']):
                st.markdown(f"**[{i+1}]** {ref['citation']} ({ref['pmid']})")
    else:
        st.error("Could not extract a valid PubMed ID from the input.")
def naive_sent_tokenize(text):
    """Very basic sentence splitter using punctuation."""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def generate_claim_reference_pairs(abstract, references):
    sentences = naive_sent_tokenize(abstract)
    pairs = []
    for sentence in sentences:
        for ref in references:
            pairs.append({
                "claim": sentence,
                "reference": ref["citation"],
                "reference_pmid": ref.get("pmid")
            })
    return pairs


st.subheader("Generated Claim–Reference Pairs")
pairs = generate_claim_reference_pairs(article_data['abstract'], article_data['references'])

for idx, pair in enumerate(pairs[:10]):  # limit for readability
    st.markdown(f"**Pair {idx + 1}:**")
    st.markdown(f"- **Claim:** {pair['claim']}")
    st.markdown(f"- **Reference:** {pair['reference']}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load MedBERT (fine-tuned version)
@st.cache_resource
def load_medbert_model():
    tokenizer = AutoTokenizer.from_pretrained("seloooselin/citation-analysis-medbert")
    model = AutoModelForSequenceClassification.from_pretrained("seloooselin/citation-analysis-medbert")
    return tokenizer, model

tokenizer, model = load_medbert_model()
label_map = {0: "ACCURATE", 1: "NOT_ACCURATE", 2: "IRRELEVANT"}

def classify_with_medbert(claim, reference):
    """Run MedBERT classification on a claim–reference pair."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_text = f"{claim} [SEP] {reference}"
    encoded = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return label_map[prediction]

st.subheader("MedBERT Classification Results")

for idx, pair in enumerate(pairs[:10]):
    label = classify_with_medbert(pair['claim'], pair['reference'])
    st.markdown(f"**Pair {idx + 1}:**")
    st.markdown(f"- **Claim:** {pair['claim']}")
    st.markdown(f"- **Reference:** {pair['reference']}")
    st.markdown(f"- 🧠 **MedBERT Classification:** `{label}`")
    st.markdown("---")



from openai import OpenAI
import os
# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_with_gpt35(claim, reference):
    prompt = (
        "You are an AI citation quality assessor. Given a claim from an abstract and a cited reference, "
        "classify the relationship as ACCURATE, NOT_ACCURATE, or IRRELEVANT. Then provide a one-sentence justification.\n\n"
        f"Claim: {claim}\n"
        f"Reference: {reference}\n\n"
        "Label: <ACCURATE / NOT_ACCURATE / IRRELEVANT>\n"
        "Reason: <one sentence explanation>"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a citation analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        label_line = next((line for line in content.splitlines() if "Label:" in line), None)
        reason_line = next((line for line in content.splitlines() if "Reason:" in line), None)

        label = label_line.split("Label:")[1].strip().upper() if label_line else "UNKNOWN"
        reason = reason_line.split("Reason:")[1].strip() if reason_line else "No reason found."

        return label, reason

    except Exception as e:
        return "ERROR", str(e)



st.subheader("MedBERT + GPT-3.5 Classification Comparison")

for idx, pair in enumerate(pairs[:5]):  # fewer pairs since GPT has cost/time
    claim = pair['claim']
    ref = pair['reference']

    medbert_label = classify_with_medbert(claim, ref)
    gpt_label, gpt_reason = classify_with_gpt35(claim, ref)

    st.markdown(f"### Pair {idx + 1}")
    st.markdown(f"**Claim:** {claim}")
    st.markdown(f"**Reference:** {ref}")
    st.markdown(f"🧠 **MedBERT Prediction:** `{medbert_label}`")
    st.markdown(f"🔍 **GPT-3.5 Prediction:** `{gpt_label}`")
    st.markdown(f"💬 **GPT-3.5 Reason:** {gpt_reason}")
    st.markdown("---")

