# Research Paper Summarizer

A web application that uses OpenAI's GPT-4 to generate comprehensive summaries of academic research papers.

## Features

- **PDF Upload**: Upload research papers directly as PDF files
- **arXiv Support**: Enter an arXiv URL to automatically download and summarize papers
- **Structured Summaries**: Get organized summaries with:
  - Main contributions
  - Problem statement
  - Methodology
  - Key results
  - Limitations
  - Significance
- **Clean UI**: Simple, modern interface with drag-and-drop support

## Setup

### 1. Clone the repository
```bash
cd paper-summarizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up OpenAI API key

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 4. Run the application
```bash
python app.py
```

Open your browser and go to: http://localhost:5000

## Usage

### Option 1: Upload PDF
1. Click the "Upload PDF" tab
2. Drag and drop your PDF or click to browse
3. Wait for the summary to be generated

### Option 2: arXiv URL
1. Click the "arXiv URL" tab
2. Enter the arXiv URL (e.g., https://arxiv.org/abs/2301.00001)
3. Click "Summarize"

## How it works

1. **PDF Processing**: Extracts text from uploaded PDFs using PyPDF2
2. **arXiv Integration**: Downloads papers directly from arXiv
3. **AI Summarization**: Sends the paper text to GPT-4 for analysis
4. **Structured Output**: Returns a well-organized summary with key sections

## Limitations

- Maximum file size: 50MB
- Papers are truncated at ~12,500 tokens to fit GPT-4's context window
- Only supports PDF format
- Requires OpenAI API key (paid service)

## Cost Estimation

- Each paper summary uses approximately 2,000-4,000 tokens
- With GPT-4-turbo pricing (~$0.01 per 1K tokens), each summary costs ~$0.02-0.04

## File Structure

```
paper-summarizer/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
├── templates/
│   └── index.html     # Web interface
└── uploads/           # Temporary file storage (created automatically)
```