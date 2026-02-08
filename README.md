# ☕ Barista Video Analysis

A local computer vision application that analyzes barista POV YouTube videos to detect actions, drink types, visible ingredients, and preparation stages.

The system downloads a YouTube video, extracts frames at intervals, and uses a local vision-language model (Moondream via Ollama) to generate a timeline and ingredient insights.

## 🔧 Tech Stack
- Python
- Ollama (Moondream)
- LlamaIndex Workflows
- OpenCV
- yt-dlp
- Gradio

## ▶️ Run Locally
```bash
ollama serve
ollama pull moondream
pip install gradio opencv-python yt-dlp httpx pandas llama-index
python test.py
