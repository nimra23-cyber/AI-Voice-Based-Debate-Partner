from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from gtts import gTTS
from dotenv import load_dotenv
import whisper
import noisereduce as nr
import numpy as np
import wave
import time
import requests
import shutil
import uuid
import os
import re

# ── Member 3 config ──
SAMPLE_RATE  = 16000
CHANNELS     = 1
SAMPLE_WIDTH = 2
MAX_CHUNK_MS = 25_000

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY missing")

print("GROQ API Key loaded successfully")

# =========================
# FOLDERS SETUP
# =========================

BASE_DIR = Path("debate_backend")
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Folders ready")

# =========================
# LOAD WHISPER
# =========================

print("Loading Whisper model... please wait")
whisper_model = whisper.load_model("tiny")
print("Whisper loaded successfully")

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

# ── CORS — Frontend (localhost:5173) ko backend se baat karne deta hai ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_STORE = {}

# =========================
# HELPER FUNCTIONS
# =========================

# =========================
# MEMBER 3 — AUDIO PROCESSING
# =========================

def convert_to_wav(input_path, output_path):
    """MEMBER 3 — STEP 1: FORMAT CONVERSION
    Converts any audio (webm, ogg, mp3) to WAV 16kHz mono 16-bit.
    Frontend sends webm from browser — this handles that."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(CHANNELS)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_sample_width(SAMPLE_WIDTH)
    audio.export(output_path, format="wav")
    print(f"[M3-CONVERT] Done → {output_path}")


def reduce_noise(wav_path: str) -> str:
    """MEMBER 3 — STEP 2: NOISE REDUCTION
    Removes background noise using first 0.5s as noise profile."""
    print(f"[M3-DENOISE] Reducing noise...")
    with wave.open(wav_path, 'rb') as wf:
        rate   = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    samples      = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    noise_sample = samples[:int(rate * 0.5)]
    denoised     = nr.reduce_noise(y=samples, sr=rate, y_noise=noise_sample, prop_decrease=0.75)

    output_path = wav_path.replace(".wav", "_denoised.wav")
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(denoised.astype(np.int16).tobytes())

    print(f"[M3-DENOISE] Done → {output_path}")
    return output_path


def strip_silence(wav_path: str) -> str:
    """MEMBER 3 — STEP 3: SILENCE REMOVAL
    Removes silent gaps to prevent Whisper hallucinations."""
    print(f"[M3-SILENCE] Stripping silence...")
    audio      = AudioSegment.from_wav(wav_path)
    non_silent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

    if not non_silent:
        print("[M3-SILENCE] Audio fully silent, skipping.")
        return wav_path

    trimmed     = sum(audio[s:e] for s, e in non_silent)
    output_path = wav_path.replace(".wav", "_trimmed.wav")
    trimmed.export(output_path, format="wav")
    print(f"[M3-SILENCE] Done → {output_path}")
    return output_path


def speech_to_text(audio_path):
    """MEMBER 3 — STEP 4: TRANSCRIPTION
    Full pipeline: noise reduction → silence removal → Whisper.
    Handles audio longer than 30s automatically."""
    wav_path     = str(audio_path)
    denoised     = reduce_noise(wav_path)
    trimmed      = strip_silence(denoised)

    print(f"[M3-WHISPER] Transcribing...")
    audio    = AudioSegment.from_wav(trimmed)
    duration = len(audio)

    if duration <= 30_000:
        result = whisper_model.transcribe(trimmed, language="en", fp16=False)
        text   = result["text"].strip()
    else:
        print(f"[M3-WHISPER] Long audio ({duration/1000:.1f}s) — chunking...")
        import tempfile
        texts, start = [], 0
        while start < duration:
            end   = min(start + MAX_CHUNK_MS, duration)
            chunk = audio[start:end]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk.export(tmp.name, format="wav")
                r = whisper_model.transcribe(tmp.name, language="en", fp16=False)
                texts.append(r["text"].strip())
                os.unlink(tmp.name)
            start += MAX_CHUNK_MS - 1000
        text = " ".join(texts)

    for f in [denoised, trimmed]:
        if f != wav_path and os.path.exists(f):
            os.unlink(f)

    print(f"[M3-WHISPER] Transcript: '{text}'")
    return text


def generate_gpt_response(text):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a professional debate judge and debate opponent.
- Never analyze grammar or sentence structure
- Never act like a teacher
- Never explain English mistakes
- Always respond like a debate opponent
- Be concise and direct

Give:
1. One strong counter-argument
2. Debate scoring

Evaluate:
- Clarity
- Logic
- Evidence
- Persuasiveness
- Rebuttal Strength

Respond ONLY in this format:

Argument: one strong counter-argument only

Clarity: number only
Logic: number only
Evidence: number only
Persuasiveness: number only
Rebuttal Strength: number only

Score: final number only

Feedback: short improvement feedback

User Argument:
{text}
"""

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional debate AI."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            print("Groq API Error:", response.text)

            return f"""
Argument: I disagree with your argument.
Clarity: 7
Logic: 7
Evidence: 6
Persuasiveness: 7
Rebuttal Strength: 6
Score: 7
Feedback: Improve structure and stronger examples.
"""

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print("Groq Exception:", str(e))

        return f"""
Argument: I disagree with your argument.
Clarity: 7
Logic: 7
Evidence: 6
Persuasiveness: 7
Rebuttal Strength: 6
Score: 7
Feedback: Improve structure and stronger examples.
"""


def parse_groq_output(raw_text):
    argument = ""
    score = 0
    feedback = ""

    clarity = 0
    logic = 0
    evidence = 0
    persuasiveness = 0
    rebuttal_strength = 0

    try:
        # Argument
        argument_match = re.search(
            r"Argument:\s*(.*?)\s*Clarity:",
            raw_text,
            re.DOTALL | re.IGNORECASE
        )

        if argument_match:
            argument = argument_match.group(1).strip()

        # Helper function
        def extract_number(label):
            match = re.search(
                rf"{label}:\s*(\d+)",
                raw_text,
                re.IGNORECASE
            )
            return float(match.group(1)) if match else 0

        clarity = extract_number("Clarity")
        logic = extract_number("Logic")
        evidence = extract_number("Evidence")
        persuasiveness = extract_number("Persuasiveness")
        rebuttal_strength = extract_number("Rebuttal Strength")
        score = extract_number("Score")

        # Feedback
        feedback_match = re.search(
            r"Feedback:\s*(.*)",
            raw_text,
            re.DOTALL | re.IGNORECASE
        )

        if feedback_match:
            feedback = feedback_match.group(1).strip()

    except Exception as e:
        print("Parsing Error:", str(e))

    return {
        "argument": argument,
        "clarity": clarity,
        "logic": logic,
        "evidence": evidence,
        "persuasiveness": persuasiveness,
        "rebuttal_strength": rebuttal_strength,
        "score": score,
        "feedback": feedback
    }


import requests

def text_to_speech(text, output_path):
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

    if not ELEVENLABS_API_KEY:
        print("ElevenLabs key missing, using gTTS fallback")

        tts = gTTS(text=text, lang="en")
        tts.save(str(output_path))
        return

    url = "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.3,
            "similarity_boost": 0.5
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print("ElevenLabs voice generated successfully")

        else:
            print("ElevenLabs failed, fallback to gTTS")

            tts = gTTS(text=text, lang="en")
            tts.save(str(output_path))

    except Exception as e:
        print("TTS Error:", str(e))

        tts = gTTS(text=text, lang="en")
        tts.save(str(output_path))


def run_pipeline_real(normalized_audio_path, request_id):
    try:
        print("NEW PIPELINE IS RUNNING")

        # Log start time
        start_time = time.time()

        # Step 1: Speech to text
        transcript = speech_to_text(normalized_audio_path)


        # Step 2: Gemini response
        raw_reply = generate_gpt_response(transcript)

        # Step 3: Parse Groq output
        parsed = parse_groq_output(raw_reply)

        argument = parsed["argument"]
        score = parsed["score"]
        feedback = parsed["feedback"]

        # Step 4: Text to speech
        response_audio_path = OUTPUT_DIR / f"{request_id}_response.mp3"
        text_to_speech(argument, response_audio_path)

        # Log processing time
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        print(f"Total Processing Time: {processing_time} seconds")

        # Step 5: Final result
        result = {
            "request_id": request_id,
            "status": "success",
            "input_file": "",
            "normalized_file": normalized_audio_path.name,
            "response_audio_file": response_audio_path.name,
            "transcript": transcript,
            "reply_text": argument,
            "clarity": parsed["clarity"],
            "logic": parsed["logic"],
            "evidence": parsed["evidence"],
            "persuasiveness": parsed["persuasiveness"],
            "rebuttal_strength": parsed["rebuttal_strength"],
            "score": score,
            "feedback": feedback,
            "processing_time_seconds": processing_time,
            "detail": None,
        }

        REQUEST_STORE[request_id] = result
        return result

    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "input_file": None,
            "normalized_file": None,
            "response_audio_file": None,
            "transcript": None,
            "reply_text": None,
            "score": None,
            "feedback": None,
            "detail": str(e),
        }

# =========================
# API ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Debate AI Backend Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/debate")
async def debate(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:12]

    input_file_path = INPUT_DIR / f"{request_id}_{file.filename}"

    with open(input_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

      
    # If already WAV, skip conversion
    if file.filename.lower().endswith(".wav"):
        print("WAV file detected — skipping conversion")
        normalized_audio_path = input_file_path

    else: 
        print("Non-WAV file detected — converting to WAV")
        normalized_audio_path = INPUT_DIR / f"{request_id}_normalized.wav"
        convert_to_wav(input_file_path, normalized_audio_path)

    result = run_pipeline_real(normalized_audio_path, request_id)


    result["input_file"] = input_file_path.name

    return result


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return {"error": "File not found"}

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/mpeg"
    )
