# Threat Detector from Speech 

Real-time safety pipeline that ingests WAV audio, segments speech turns, transcribes content, scores threat risk, applies incident logic, and publishes structured alerts for downstream operators.

## 1) Setup Instructions

### 1) Prerequisites

- Python 3.11+ (3.12 tested)
- `ffmpeg` (required by `librosa` workflows in many environments)
- OpenAI API key
- Docker Desktop (opt.)

### 2) Clone and create environment

```bash
cd /path/to/eNOugh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) .env file

Create a .env file in the repository root and in it write: 
```
OPENAI_API_KEY= *your_openai_api_key*
```

### 4) Setup audio 

Place the provided sample .wav files into an audio/ directory

## 2) Running WAV Harness

Enter the eNOugh directory and run the below locally:
```
python scripts/run_pipeline.py --audio audio/*audio file name*
```

Below is the pipeline flow:
- Segment WAV into turns
- Transcribe each turn
- Run threat assessment + fusion
- Apply incident/action logic
- Publish alert events
- Print formatted alert messages (medium/high)

## 3) Docker sandbox run 

To complete the dockerization, simply run 
```
docker compose up --build
```

## 4) Time breakdown

5 hours - online research, looking at given files, decomposing given task, going through tasks \
8 hours - building the product, refining model to decrease latency \
1 hour - dockerization \
<1 hour - README development 

## 5) Design Notes

#### VAD / End-of-Turn strategy
Turn segmentation is silence-based: \
CHUNK_SIZE = 512 samples \
SILENCE_THRESHOLD = 0.03 (RMS) \
SILENCE_DURATION = 0.6s to mark turn end \
Turns shorter than 0.3s are dropped as probable noise. This provides lightweight VAD-like behavior for offline WAV harnessing. 

#### Signal fusion weights
Final fused score uses weighted signals: \
Keywords: 32% \
Tone: 48% \
Turn escalation: 10% \
Volume: 6% \
Audio escalation: 4% \

If the directionality is ambient, the danger score is capped at 0.15

#### Threshold strategy:
Two-level thresholding: 
Severity bands \
>= 0.75 → critical \
>= 0.50 → warning \
>= 0.30 → monitor \
< 0.30 → safe 

Incident action thresholds 
>= 0.8 → high alert \
>= 0.6 → medium alert \
>= 0.3 → low (no active intervention) \
< 0.3 → very_low (no action) 

#### Cooldown / duplicate suppression
Same/lower incoming level than active action → suppressed \
Higher level than active action → escalate + fire action \ 
Return to low/very_low → incident closes and state resets 

#### Alert Schema

Below fields were added: 
- badge_id
- alert_id
- recording_start_time
- turn_start_time
- transcription
- alert_level
- action
- action_reason

## 6) Testing 
```
python3 -m unittest tests/test_pipeline_basics.py
```
Tests for the below:
Alert schema includes exactly required keys
Non-actionable events are not published
Actionable events are queued/published

## 7) Known Limitations

- LLM/transcription calls depend on network/API availability and users may not always be in areas that have APIs available
- The weightages I provided are rough estimates. Not strong recommendations
- Noisy environments may affect the boundaries between turns, especially if people are talking over each other. This can make it difficult to assess escalation 
- I have not established how to tell between different speakers. However, this feature is not one of the key features. 

## 8) Notes

I tried adding an Area risk feature, but decided against it because it caused the latency to increase greatly. However, the code still works with this dataset: https://www.kaggle.com/datasets/rahulladhani/london-street-level-crime-data-2024?resource=download

