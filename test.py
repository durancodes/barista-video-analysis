"""
Barista Video Analysis - Gradio Web App
========================================
Local Ollama (moondream) + LlamaIndex Workflows
Compatible with Gradio 6.0+
"""

import asyncio
import base64
from pathlib import Path
from typing import List

import cv2
import yt_dlp
import httpx
import gradio as gr
import pandas as pd

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    Event,
    Context,
    step,
)


# ============================================
# CUSTOM EVENTS
# ============================================

class FramesExtractedEvent(Event):
    frames: List[tuple]


class AnalysisCompleteEvent(Event):
    all_analyses: List[dict]


# ============================================
# OLLAMA CLIENT
# ============================================

class OllamaVision:
    def __init__(self, model: str = "moondream", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def is_available(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False


# ============================================
# WORKFLOW
# ============================================

class BaristaAnalysisWorkflow(Workflow):
    
    def __init__(self, model: str = "moondream", progress_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.vlm = OllamaVision(model=model)
        self.progress_callback = progress_callback
    
    def _update_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)
    
    @step
    async def extract_frames(self, ctx: Context, ev: StartEvent) -> FramesExtractedEvent:
        video_url = ev.video_url
        sample_interval = getattr(ev, 'sample_interval', 30)
        max_frames = getattr(ev, 'max_frames', 10)
        
        output_dir = Path("./frames")
        output_dir.mkdir(exist_ok=True)
        
        self._update_progress(f"📥 Downloading video...")
        video_path = "./barista_video.mp4"
        
        if Path(video_path).exists():
            Path(video_path).unlink()
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': video_path,
            'quiet': True,
            'overwrites': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        self._update_progress(f"📹 Extracting frames every {sample_interval}s...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frame_interval = int(fps * sample_interval)
        frames = []
        frame_idx = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frame_path = output_dir / f"frame_{timestamp:.0f}s.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append((timestamp, str(frame_path)))
            
            frame_idx += 1
        
        cap.release()
        
        await ctx.store.set("video_duration", duration)
        await ctx.store.set("total_frames", len(frames))
        
        self._update_progress(f"✅ Extracted {len(frames)} frames from {duration:.0f}s video")
        
        return FramesExtractedEvent(frames=frames)
    
    @step
    async def analyze_frames(self, ctx: Context, ev: FramesExtractedEvent) -> AnalysisCompleteEvent:
        analyses = []
        
        prompt = """Analyze this barista POV image. Respond in this exact format:

ACTION: [What the barista is doing right now]
DRINK_TYPE: [Type of drink being made, or "unknown"]
INGREDIENTS_VISIBLE: [Comma-separated list: milk, cream, whip cream, syrup, espresso, ice, etc.]
STAGE: [preparation/pouring/topping/serving/idle]

Be specific and only list ingredients you can actually see."""

        for i, (timestamp, frame_path) in enumerate(ev.frames):
            self._update_progress(f"🔍 Analyzing frame {i+1}/{len(ev.frames)} ({timestamp:.0f}s)...")
            
            try:
                text = self.vlm.analyze_image(frame_path, prompt)
                
                analysis = {
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "raw_response": text,
                    "ingredients": self._extract_ingredients(text),
                    "stage": self._extract_field(text, "STAGE"),
                    "drink_type": self._extract_field(text, "DRINK_TYPE"),
                    "action": self._extract_field(text, "ACTION"),
                }
            except Exception as e:
                analysis = {
                    "timestamp": timestamp,
                    "frame_path": frame_path,
                    "raw_response": f"ERROR: {e}",
                    "ingredients": [],
                    "stage": "error",
                    "drink_type": "unknown",
                    "action": f"Error: {e}",
                }
            
            analyses.append(analysis)
        
        await ctx.store.set("analyses", analyses)
        return AnalysisCompleteEvent(all_analyses=analyses)
    
    @step
    async def generate_summary(self, ctx: Context, ev: AnalysisCompleteEvent) -> StopEvent:
        self._update_progress("📊 Generating summary...")
        
        analyses = ev.all_analyses
        
        ingredient_counts = {}
        drink_types = {}
        stages = {}
        
        for a in analyses:
            for ing in a["ingredients"]:
                ingredient_counts[ing] = ingredient_counts.get(ing, 0) + 1
            
            drink = a["drink_type"]
            if drink and drink.lower() not in ["unknown", "error"]:
                drink_types[drink] = drink_types.get(drink, 0) + 1
            
            stage = a["stage"]
            if stage and stage != "error":
                stages[stage] = stages.get(stage, 0) + 1
        
        whip_cream_frames = [
            a for a in analyses 
            if any(x in str(a["ingredients"]).lower() for x in ["whip", "cream"])
        ]
        
        video_duration = await ctx.store.get("video_duration")
        
        summary = {
            "total_frames": len(analyses),
            "video_duration": video_duration,
            "ingredient_frequency": dict(sorted(ingredient_counts.items(), key=lambda x: -x[1])),
            "drink_types_seen": drink_types,
            "activity_breakdown": stages,
            "whip_cream_count": len(whip_cream_frames),
            "whip_cream_timestamps": [f"{a['timestamp']:.0f}s" for a in whip_cream_frames],
            "timeline": analyses,
        }
        
        self._update_progress("✅ Analysis complete!")
        return StopEvent(result=summary)
    
    def _extract_ingredients(self, text: str) -> List[str]:
        for line in text.split("\n"):
            if "INGREDIENTS_VISIBLE:" in line.upper():
                ingredients_str = line.split(":", 1)[-1].strip()
                ingredients = [i.strip().lower() for i in ingredients_str.split(",")]
                return [i for i in ingredients if i and i not in ["none", "n/a", "", "unknown"]]
        return []
    
    def _extract_field(self, text: str, field: str) -> str:
        for line in text.split("\n"):
            if f"{field}:" in line.upper():
                return line.split(":", 1)[-1].strip()
        return ""


# ============================================
# GRADIO APP
# ============================================

def check_ollama_status():
    vlm = OllamaVision()
    if vlm.is_available():
        return "✅ Ollama is running"
    else:
        return "❌ Ollama not detected. Run: `ollama serve` and `ollama pull moondream`"


def create_timeline_df(timeline: List[dict]) -> pd.DataFrame:
    rows = []
    for entry in timeline:
        rows.append({
            "Time": f"{entry['timestamp']:.0f}s",
            "Action": entry['action'][:100] if entry['action'] else "N/A",
            "Stage": entry['stage'] or "N/A",
            "Ingredients": ", ".join(entry['ingredients']) if entry['ingredients'] else "None detected",
            "Drink Type": entry['drink_type'] or "Unknown",
        })
    return pd.DataFrame(rows)


def create_ingredients_df(ingredient_counts: dict) -> pd.DataFrame:
    if not ingredient_counts:
        return pd.DataFrame({"Ingredient": [], "Count": []})
    
    items = list(ingredient_counts.items())[:10]
    return pd.DataFrame({
        "Ingredient": [x[0] for x in items],
        "Count": [x[1] for x in items],
    })


async def analyze_video(video_url: str, sample_interval: int, max_frames: int, progress=gr.Progress()):
    vlm = OllamaVision()
    if not vlm.is_available():
        return (
            "❌ Error: Ollama is not running. Start it with `ollama serve`",
            None, None, None, []
        )
    
    status_messages = []
    def update_progress(msg):
        status_messages.append(msg)
        progress(len(status_messages) / 15, desc=msg)
    
    try:
        workflow = BaristaAnalysisWorkflow(
            model="moondream",
            progress_callback=update_progress,
            timeout=1800,
            verbose=False,
        )
        
        result = await workflow.run(
            video_url=video_url,
            sample_interval=sample_interval,
            max_frames=max_frames,
        )
        
        summary_text = f"""
## 📊 Analysis Summary

- **Video Duration:** {result['video_duration']:.0f} seconds
- **Frames Analyzed:** {result['total_frames']}
- **Whip Cream Appearances:** {result['whip_cream_count']}
{f"  - Timestamps: {', '.join(result['whip_cream_timestamps'])}" if result['whip_cream_timestamps'] else ""}

### 🧪 Ingredients Detected
{chr(10).join([f"- **{ing}**: {count} frames" for ing, count in result['ingredient_frequency'].items()]) if result['ingredient_frequency'] else "No ingredients detected"}

### ☕ Drink Types
{chr(10).join([f"- {drink}: {count}" for drink, count in result['drink_types_seen'].items()]) if result['drink_types_seen'] else "No specific drink types identified"}

### 📍 Activity Breakdown
{chr(10).join([f"- {stage}: {count} frames" for stage, count in result['activity_breakdown'].items()]) if result['activity_breakdown'] else "No activity stages detected"}
"""
        
        timeline_df = create_timeline_df(result['timeline'])
        ingredients_df = create_ingredients_df(result['ingredient_frequency'])
        
        frame_images = [entry['frame_path'] for entry in result['timeline'] if Path(entry['frame_path']).exists()]
        
        return (
            summary_text,
            timeline_df,
            ingredients_df,
            result,
            frame_images[:10],
        )
        
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            None, None, None, []
        )


def run_analysis(video_url, sample_interval, max_frames):
    return asyncio.run(analyze_video(video_url, sample_interval, max_frames))


# ============================================
# BUILD UI (Gradio 6.0 compatible)
# ============================================

with gr.Blocks(title="🎬 Barista Video Analysis") as app:
    
    gr.Markdown("""
    # 🎬 Barista Video Analysis
    ### Powered by LlamaIndex Workflows + Ollama (Moondream)
    
    Analyze barista POV videos to track ingredients, drink types, and activities.
    """)
    
    with gr.Row():
        status_text = gr.Markdown(check_ollama_status())
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(fn=check_ollama_status, outputs=status_text)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_url = gr.Textbox(
                label="YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                value="https://www.youtube.com/watch?v=3euggX69wzQ",
            )
        
        with gr.Column(scale=1):
            sample_interval = gr.Slider(
                minimum=10, maximum=120, value=60, step=10,
                label="Sample Interval (seconds)",
                info="Extract a frame every N seconds"
            )
            max_frames = gr.Slider(
                minimum=5, maximum=30, value=10, step=1,
                label="Max Frames",
                info="Maximum frames to analyze"
            )
    
    analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    with gr.Tabs():
        with gr.TabItem("📋 Summary"):
            summary_output = gr.Markdown("*Results will appear here after analysis*")
        
        with gr.TabItem("📜 Timeline"):
            timeline_table = gr.DataFrame(
                label="Frame-by-Frame Analysis",
                wrap=True,
            )
        
        with gr.TabItem("📊 Ingredients"):
            ingredients_table = gr.DataFrame(
                label="Ingredient Frequency",
            )
        
        with gr.TabItem("🖼️ Frames"):
            frame_gallery = gr.Gallery(
                label="Analyzed Frames",
                columns=5,
                height="auto",
            )
        
        with gr.TabItem("🔧 Raw Data"):
            raw_json = gr.JSON(label="Full Analysis Data")
    
    analyze_btn.click(
        fn=run_analysis,
        inputs=[video_url, sample_interval, max_frames],
        outputs=[summary_output, timeline_table, ingredients_table, raw_json, frame_gallery],
    )
    
    gr.Markdown("""
    ---
    **Tips:**
    - Make sure Ollama is running: `ollama serve`
    - First run will download the video (may take a moment)
    - Each frame takes ~2-4 seconds to analyze locally
    """)


# ============================================
# LAUNCH
# ============================================

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )