from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/html.info.html")

# Load model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
print("Device set to use cpu")

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

@app.post("/summarize")
async def summarize(request: Request):
    try:
        data = await request.json()
        url = data.get("url")
        video_id = extract_video_id(url)

        if not video_id:
            return JSONResponse(content={"summary": "Invalid YouTube URL."}, status_code=400)

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])

        if len(text) > 1000:
            text = text[:1000]

        result = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return {"summary": result[0]["summary_text"]}

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(content={"summary": "Oops! Something went wrong!"}, status_code=500)
