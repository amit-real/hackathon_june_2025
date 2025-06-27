from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import os
import re
import json
import tempfile
import requests
import textwrap
import pdfplumber
import tiktoken
from pathlib import Path
from werkzeug.utils import secure_filename
from openai import OpenAI
from datetime import datetime
from typing import Dict, List

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

socketio = SocketIO(app, cors_allowed_origins="*")

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- config ----------
MODEL_NAME = "gemini-2.0-flash"     # Using Gemini model
TOKENS_PER_CHUNK = 3000              # leave head-room under model context window
FINAL_BULLETS = 8                    # how many bullets / items to keep
TIMEOUT = 30                         # seconds for HTTP download

# Target JSON schema all prompts must return
JSON_KEYS = [
    "bullets",
    "use_cases",
    "strengths",
    "limitations",
    "open_questions",
]
# ----------------------------

# Initialize Gemini via OpenAI-compatible endpoint
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    try:
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        print("Gemini API configured successfully")
        model_initialized = True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        model_initialized = False
        client = None
else:
    print("=" * 60)
    print("Gemini API key not found!")
    print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
    print("=" * 60)
    model_initialized = False
    client = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def download_arxiv_pdf(arxiv_id: str, out_dir: Path) -> Path:
    """Download PDF from arXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = out_dir / f"{arxiv_id}.pdf"
    
    socketio.emit('progress', {'message': f'Downloading arXiv:{arxiv_id}...', 'percentage': 10})
    
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percentage = min(20, 10 + (downloaded / total_size) * 10)
                    socketio.emit('progress', {'percentage': percentage})
                    
    return out_path

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber"""
    socketio.emit('progress', {'message': 'Extracting text from PDF...', 'percentage': 25})
    
    text_parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text_parts.append(text)
            percentage = 25 + (i / total_pages) * 10
            socketio.emit('progress', {'percentage': percentage})
            
    return "\n".join(text_parts)

def chunk_text(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks with <= max_tokens each"""
    socketio.emit('progress', {'message': 'Chunking document...', 'percentage': 40})
    
    # Use a standard encoding instead of one tied to a specific OpenAI model name
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    
    chunks: List[str] = []
    current_words: List[str] = []
    current_tokens = 0
    
    for w in words:
        # Encoding the space separately can be more accurate
        tok_len = len(enc.encode(" " + w))
        if current_tokens + tok_len > max_tokens:
            chunks.append(" ".join(current_words))
            current_words, current_tokens = [], 0
        current_words.append(w)
        current_tokens += tok_len
        
    if current_words:
        chunks.append(" ".join(current_words))
        
    return chunks

def summarize_chunk(chunk: str, chunk_num: int, total_chunks: int) -> Dict[str, List[str]]:
    """Call the API on a chunk, asking for structured JSON"""
    system = (
        "You are an expert research analyst. "
        "Your task is to extract specific information from a fragment of an academic paper. "
        "Read the text and return a single, valid JSON object and nothing else. "
        "Do NOT add explanations, apologies, or markdown formatting like ```json. "
        f"The JSON object must have these exact keys: {JSON_KEYS}. "
        "Each key's value must be a list of short, concise strings (max 25 words)."
    )
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
            temperature=0.2,
        )
        
        # Update progress
        percentage = 45 + (chunk_num / total_chunks) * 40
        socketio.emit('progress', {
            'message': f'Analyzing chunk {chunk_num}/{total_chunks}...', 
            'percentage': percentage
        })
        
        raw_content = resp.choices[0].message.content or ""
        
        # Aggressively find a JSON object within the response string
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print(f"⚠️  WARNING: API did not return a JSON object for chunk {chunk_num}")
            return {k: [] for k in JSON_KEYS}
            
    except json.JSONDecodeError:
        print(f"❌ ERROR: Failed to decode extracted JSON for chunk {chunk_num}")
        return {k: [] for k in JSON_KEYS}
    except Exception as e:
        print(f"❌ An unexpected error occurred during API call: {e}")
        return {k: [] for k in JSON_KEYS}

def merge_dicts(dicts: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Combine lists under each JSON key, deduplicating"""
    merged = {k: [] for k in JSON_KEYS}
    seen_per_key = {k: set() for k in JSON_KEYS}
    
    for d in dicts:
        for k in JSON_KEYS:
            for item in d.get(k, []):
                norm = item.lower().strip()
                if norm not in seen_per_key[k]:
                    merged[k].append(item.strip())
                    seen_per_key[k].add(norm)
    return merged

def condense_json(raw_json: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Ask the API to rank/trim each list down to FINAL_BULLETS items"""
    socketio.emit('progress', {'message': 'Creating final summary...', 'percentage': 90})
    
    prompt = textwrap.dedent(
        f"""
        Merge / deduplicate / polish the lists in the JSON below.
        For each key keep the **{FINAL_BULLETS} most important** items (or fewer
        if content is lacking). Rephrase for clarity, keep technical precision.
        Return a single, valid JSON object and nothing else. Do not add explanations or markdown.
        """
    )
    
    input_json_str = json.dumps(raw_json, indent=2)
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful JSON processing assistant."},
                {"role": "user", "content": prompt + "\n\n" + input_json_str},
            ],
            temperature=0.2,
        )
        
        socketio.emit('progress', {'message': 'Complete!', 'percentage': 100})
        
        raw_content = resp.choices[0].message.content or ""
        
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print(f"⚠️  WARNING: Condenser API did not return a JSON object. Returning original merged JSON.")
            return raw_json  # Fallback to un-condensed JSON
            
    except Exception as e:
        print(f"❌ An error occurred during condensation step: {e}. Returning original merged JSON.")
        return raw_json  # Fallback to un-condensed JSON

def format_summary_html(result: Dict[str, List[str]]) -> str:
    """Convert the structured result to HTML format"""
    html = []
    
    # Key Takeaways
    html.append("<h2>🔹 Key Takeaways</h2>")
    html.append("<ul class='summary-bullets'>")
    for bullet in result.get("bullets", []):
        html.append(f"<li>{bullet}</li>")
    html.append("</ul>")
    
    # Helper function for sections
    def add_section(title: str, emoji: str, key: str):
        items = result.get(key, [])
        if items:
            html.append(f"<h2>{emoji} {title}</h2>")
            html.append("<ol class='summary-list'>")
            for item in items:
                html.append(f"<li>{item}</li>")
            html.append("</ol>")
    
    add_section("Potential Use Cases", "🔸", "use_cases")
    add_section("Strengths", "💪", "strengths")
    add_section("Limitations", "⚠️", "limitations")
    add_section("Open Questions / Next Steps", "🔍", "open_questions")
    
    return "\n".join(html)

def process_paper(arxiv_url=None, pdf_path=None):
    """Process paper and return structured summary"""
    if not model_initialized or not client:
        return None, "Gemini API not initialized"
    
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            
            if arxiv_url:
                arxiv_id = re.sub(r"(https?://arxiv\.org/(abs|pdf)/)?", "", arxiv_url)
                pdf_path = download_arxiv_pdf(arxiv_id, tmp_dir)
            else:
                pdf_path = Path(pdf_path)
                if not pdf_path.exists():
                    return None, "PDF file not found"
                    
            # Extract text
            full_text = extract_text_from_pdf(pdf_path)
            
            if not full_text.strip():
                return None, "Could not extract text from PDF"
            
            # Chunk text
            chunks = chunk_text(full_text, TOKENS_PER_CHUNK)
            total_chunks = len(chunks)
            
            socketio.emit('progress', {
                'message': f'Processing {total_chunks} chunks...', 
                'percentage': 45
            })
            
            # Summarize each chunk
            partial_dicts = []
            for i, chunk in enumerate(chunks, 1):
                summary = summarize_chunk(chunk, i, total_chunks)
                partial_dicts.append(summary)
            
            # Merge and condense
            merged = merge_dicts(partial_dicts)
            final = condense_json(merged)
            
            # Format as HTML
            html_summary = format_summary_html(final)
            
            return html_summary, None
            
    except Exception as e:
        return None, f"Error processing paper: {str(e)}"

@app.route('/')
def index():
    return render_template('index_v2.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle both file upload and arXiv URL"""
    pdf_path = None
    temp_file = False
    
    try:
        # Check if it's an arXiv URL
        if 'arxiv_url' in request.form and request.form['arxiv_url']:
            arxiv_url = request.form['arxiv_url'].strip()
            summary, error = process_paper(arxiv_url=arxiv_url)
            
        # Check if it's a file upload
        elif 'paper' in request.files:
            file = request.files['paper']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(pdf_path)
                temp_file = True
                
                summary, error = process_paper(pdf_path=pdf_path)
            else:
                return jsonify({'error': 'Please upload a PDF file'}), 400
        else:
            return jsonify({'error': 'No file or URL provided'}), 400
        
        # Clean up temporary file
        if temp_file and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        if error:
            return jsonify({'error': error}), 500
            
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        # Clean up on error
        if temp_file and pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    socketio.run(app, debug=debug, host='0.0.0.0', port=port)