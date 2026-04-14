from dotenv import load_dotenv
import os
import openai
import io
import numpy as np
import soundfile as sf
import time
import re
import uuid
from datetime import datetime, timezone

#general filler words to ignore
TRANSCRIBED_ARTIFACTS = [
    r'\[.*?\]',          # [BLANK_AUDIO], [MUSIC], [NOISE]
    r'\(.*?\)',          # (inaudible), (crosstalk)
    r'\bum+\b',         # um, umm
    r'\buh+\b',         # uh, uhh
    r'\blike\b(?=\s)',  # filler "like" (only when followed by space)
    r'\byou know\b',
    r'\bi mean\b',
]

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    # Convert numpy array -> WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    buffer.name = "turn.wav"  # OpenAI SDK checks for a filename
    return buffer

def transcribe_turn(audio: np.ndarray, sample_rate: int) -> dict:
    # Send audio turn to gpt-40-mini-transcribe. Returns raw transcript + latency.
    wav_buffer = numpy_to_wav_bytes(audio, sample_rate)
    
    t_start = time.perf_counter()
    
    response = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=wav_buffer,
        response_format="text"
    )
    
    latency_ms = int((time.perf_counter() - t_start) * 1000)
    
    return {
        "raw_transcript": response.strip(),
        "model_latency_ms": latency_ms
    }

#removing filler words and normalising whitespace
def clean_transcript(raw: str) -> str:
    text = raw.lower().strip()
    
    for pattern in TRANSCRIBED_ARTIFACTS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_transcript_record(
    audio: np.ndarray,
    sample_rate: int,
    #TO-D): Implement badge_id to tell who is really speaking
    badge_id: str = "badge_001"
) -> dict:
    # audio array → structured transcript record.
    duration_seconds = round(len(audio) / sample_rate, 3)
    
    ai_result = transcribe_turn(audio, sample_rate)
    raw = ai_result["raw_transcript"]
    clean = clean_transcript(raw)
    
    record = {
        "turn_id":            str(uuid.uuid4()),
        "badge_id":           badge_id,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "duration_seconds":   duration_seconds,
        "raw_transcript":     raw,
        "clean_transcript":   clean,
        "word_count":         len(clean.split()) if clean else 0,
        "latency_ms": {
            "gpt-4o": ai_result["model_latency_ms"]
        }
    }
    
    return record


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from audio_processing import ingest   # your Step 1 file

    ROOT = Path(__file__).parent.parent
    filepath = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "audio" / "casual_chat.wav")

    for i, (turn_audio, sr) in enumerate(ingest(filepath)):
        print(f"\n--- Turn {i+1} ---")
        record = build_transcript_record(turn_audio, sr)
        print(f"Duration:   {record['duration_seconds']}s")
        print(f"Raw:        {record['raw_transcript']}")
        print(f"Clean:      {record['clean_transcript']}")
        print(f"Words:      {record['word_count']}")
        print(f"Latency:    {record['latency_ms']['gpt-4o']}ms")