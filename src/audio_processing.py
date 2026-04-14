import soundfile as sf
import numpy as np
import librosa
import time
import queue
import threading

#reasonable values to compute silence 
SILENCE_THRESHOLD = 0.03
SILENCE_DURATION = 0.6
# samples per "frame". Chose 512 to mimic eNO wearable buffer size
CHUNK_SIZE = 512 

def load_audio(filepath: str):
    #loading the wav file into a np array
    samples, sample_rate = sf.read(filepath, dtype='float32')
    
    # if 2d array, make it a 1d array by finding the mean of two columns
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    return samples, sample_rate

def compute_rms(turn: np.ndarray) -> float:
    rms_turn = np.sqrt(np.mean(turn**2))
    return float(rms_turn)

def stream_wav(samples: np.ndarray, 
                sample_rate: int, 
                audio_queue: queue.Queue):
    """
    Pushing chunks of audio onto queue to simulate live audio stream
    using real-time pacing
    """
    total_chunks = len(samples)//CHUNK_SIZE
    for i in range(total_chunks):
        start = i * CHUNK_SIZE
        end   = start + CHUNK_SIZE
        chunk = samples[start:end]
        # you enqueue an item with .put 
        audio_queue.put(chunk)
        
        # simulating real time w/ sleep 
        time.sleep(CHUNK_SIZE / sample_rate)
    
    # show the stream is done
    audio_queue.put(None)

def segment_turns(audio_queue: queue.Queue,
                  sample_rate: int):
    """
    Converts speech turns --> numpy arrays.
    1 turn --> silence exceeds SILENCE_DURATION seconds.
    """
    silence_chunks_needed = int(
        (SILENCE_DURATION * sample_rate) / CHUNK_SIZE
    )
    
    current_turn   = []   # chunks collected so far
    silence_count  = 0    # consecutive silent chunks
    
    while True:
        #dequeue in python is .get
        chunk = audio_queue.get()
        # None = stream finished
        if chunk is None:
            if current_turn:
                yield np.concatenate(current_turn)
                
            break
        
        rms = compute_rms(chunk)
        
        if rms < SILENCE_THRESHOLD:
            silence_count += 1
            
            # Still buffer silence (speech often has brief pauses)
            if current_turn:
                current_turn.append(chunk)
            
            # Enough silence --> turn is over
            if silence_count >= silence_chunks_needed and current_turn:
                turn_audio = np.concatenate(current_turn)
                yield turn_audio
                current_turn  = []
                silence_count = 0
        else:
            # Active speech — reset silence counter, add to turn
            #allows micro pauses to not impact sentence processing
            silence_count = 0
            current_turn.append(chunk)
            
def extract_audio_signals(audio: np.ndarray, sr: int, prev_audio: np.ndarray = None) -> dict:
    """
    Extract real-time acoustic threat indicators.
    Returns normalized 0–1 features.
    """

    # Convert to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # ─────────────────────────────
    # 1. Current volume (RMS energy)
    # ─────────────────────────────
    rms = librosa.feature.rms(y=audio)[0]
    current_volume = float(np.mean(rms))
    current_volume_norm = min(current_volume * 10, 1.0)  # crude normalization

    # ─────────────────────────────
    # 2. Voice pitch (fundamental frequency)
    # ─────────────────────────────
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_norm = min(max(pitch / 300.0, 0.0), 1.0)  # ~300Hz cap for normalization
    except:
        pitch_norm = 0.0

    # ─────────────────────────────
    # 3. Rate of volume increase (stress escalation)
    # ─────────────────────────────
    if prev_audio is None:
        volume_delta = 0.0
    else:
        prev_rms = librosa.feature.rms(y=prev_audio)[0]
        prev_volume = np.mean(prev_rms)
        volume_delta = current_volume - prev_volume

    volume_escalation = float(np.clip(volume_delta * 10, 0.0, 1.0))

    return {
        "current_volume": current_volume_norm,
        "volume_escalation": volume_escalation,
        "voice_pitch": pitch_norm
    }
            
def ingest(filepath: str):
    """
    Returns a generator of turn audio arrays.
    
    Usage:
        for turn_audio, sample_rate in ingest("audio/heated.wav"):
        --> turn_audio is a numpy float32 array, pass it to the transcription step
    """
    samples, sample_rate = load_audio(filepath)
    audio_queue = queue.Queue()
    
    # Stream in a background thread (mimics hardware interrupt)
    stream_thread = threading.Thread(
        target=stream_wav,
        args=(samples, sample_rate, audio_queue),
        daemon=True
    )
    stream_thread.start()
    
    for turn_audio in segment_turns(audio_queue, sample_rate):
        duration = len(turn_audio) / sample_rate
        
        # skip turns shorter than 0.3s --> more likely noise than speech
        if duration < 0.3:
            continue
            
        yield turn_audio, sample_rate
        
        
#TEST FUNCTION:
if __name__ == "__main__":
    import sys
    
    filepath = sys.argv[1]  # e.g. python ingestion.py audio_samples/casual_chat.wav
    
    for i, (turn, sr) in enumerate(ingest(filepath)):
        duration = len(turn) / sr
        rms = compute_rms(turn)
        print(f"Turn {i+1}: {duration:.2f}s | RMS: {rms:.4f}")
        
        