
# Citation Analysis Project

This project analyses academic paper citations to determine whether they are accurate, irrelevant, or misrepresenting the source. It focuses on medical literature and uses fine-tuned NLP models to evaluate citation quality.

## Features
- Upload a PDF of an academic paper OR input a PubMed link.
- Extracts citation sentences from the text.
- Classifies each citation using a fine-tuned **MedBERT** model or **GPT-3.5** via OpenAI API.
- Displays each citation with its reference and classification result.

## Installation

```bash
git clone https://github.com/seloooselin/citation-analysis-project.git
cd citation-analysis-project
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run citation_analysis.py
```

## Input Options

You can:
1. Upload a PDF file of an academic paper.
2. Paste a PubMed URL or PMID to fetch paper data directly from PubMed.

## Output Format

Each detected citation will include:
- The **citation sentence**
- The **cited reference**
- The **classification** (ACCURATE, NOT_ACCURATE, or IRRELEVANT)
- The **model used** (MedBERT or GPT-3.5)

## Notes

- GPT-3.5 requires an OpenAI API key set in your environment as `OPENAI_API_KEY`.
- PubMed extraction requires an active internet connection.

## License

MIT License
