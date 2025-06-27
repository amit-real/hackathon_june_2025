# 📄 Research Paper Summarizer

A powerful web application that generates structured, multi-category summaries of academic papers using Google's Gemini 2.0 Flash model. Upload a PDF or provide an arXiv URL to get comprehensive insights in seconds.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **📤 Dual Input Methods**: Upload PDF files directly or paste arXiv URLs
- **🤖 AI-Powered Analysis**: Uses Google's Gemini 2.0 Flash for intelligent summarization
- **📊 Structured Output**: Generates 5 categories of insights:
  - 🔹 **Key Takeaways**: Main findings and contributions
  - 🔸 **Use Cases**: Practical applications
  - 💪 **Strengths**: Methodological advantages
  - ⚠️ **Limitations**: Caveats and weaknesses
  - 🔍 **Open Questions**: Future research directions
- **⚡ Real-time Progress**: WebSocket-powered progress tracking
- **🎨 Modern UI**: Dark-themed, responsive interface
- **📦 Smart Chunking**: Handles long papers with token-based text splitting

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Google Gemini API key](https://makersuite.google.com/app/apikey)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/paper-summarizer.git
   cd paper-summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

4. **Run the application**
   ```bash
   ./run_web.sh
   # Or directly:
   python app_web.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file with:

```env
GEMINI_API_KEY=your-gemini-api-key-here
SECRET_KEY=your-secret-key-here  # Optional, for production
FLASK_ENV=development  # or 'production'
```

### Key Settings

Edit these in `app_web.py`:

```python
MODEL_NAME = "gemini-2.0-flash"  # AI model
TOKENS_PER_CHUNK = 3000          # Max tokens per chunk
FINAL_BULLETS = 8                # Items per category
```

## 📖 Usage

### Web Interface

1. **Choose input method**:
   - **Upload PDF**: Drag & drop or click to browse
   - **arXiv URL**: Paste the paper URL

2. **Process paper**:
   - Click "Summarize" 
   - Watch real-time progress
   - Get structured summary in ~30-60 seconds

### CLI Version

For command-line usage:

```bash
# Summarize from arXiv
python app.py --arxiv 2301.00001

# Summarize local PDF
python app.py --pdf path/to/paper.pdf
```

## 🏗️ Architecture

```
paper-summarizer/
├── app_web.py          # Flask web application
├── app.py              # CLI version
├── templates/
│   └── index_v2.html   # Web interface
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── Procfile           # Heroku/Render process file
└── README.md          # This file
```

### How It Works

1. **Text Extraction**: Uses `pdfplumber` to extract text from PDFs
2. **Smart Chunking**: Splits text into manageable chunks using `tiktoken`
3. **Parallel Processing**: Analyzes each chunk with Gemini API
4. **Intelligent Merging**: Combines and deduplicates insights
5. **Final Condensation**: Ranks and polishes top findings

## 🚀 Deployment

### Deploy to Render

1. Push code to GitHub
2. Create new Web Service on [Render](https://render.com)
3. Connect GitHub repository
4. Add environment variables
5. Deploy!

See [README_DEPLOY.md](README_DEPLOY.md) for detailed instructions.

### Deploy to Heroku

```bash
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your-key-here
git push heroku main
```

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black app_web.py

# Check style
flake8 app_web.py
```

## 📊 Example Output

```
🔹 Key Takeaways
• Novel approach to transformer architecture reducing complexity from O(n²) to O(n log n)
• Achieves 95.2% accuracy on benchmark dataset, outperforming previous SOTA by 3.1%
• ...

🔸 Potential Use Cases
1. Real-time language translation for mobile devices
2. Efficient document summarization for large-scale systems
3. ...

💪 Strengths
1. Significantly reduced computational requirements
2. Maintains accuracy while improving speed
3. ...
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini team for the powerful API
- OpenAI for the compatible endpoint format
- Flask and SocketIO communities
- arXiv for providing open access to papers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/paper-summarizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/paper-summarizer/discussions)

## 🔮 Future Enhancements

- [ ] Support for more document formats (DOCX, TXT)
- [ ] Batch processing for multiple papers
- [ ] Export summaries to various formats (PDF, Markdown)
- [ ] Integration with reference managers (Zotero, Mendeley)
- [ ] Custom prompt templates
- [ ] Multi-language support

---

Made with ❤️ by [Your Name]