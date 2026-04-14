import os
import json
import time
import uuid
import re
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
import openai

#for the test block
import sys
from pathlib import Path
from audio_processing import ingest
from transcription import build_transcript_record

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# KEYWORD TAXONOMY — tiered by severity, focussing on spoken threats

KEYWORD_TIERS = {
    "tier_1": {
        "score": 1.0,
        "phrases": {
            # Direct threats — explicit harm
            "direct": [
                "i'll kill you", "i will kill you", "i'm going to kill you",
                "i'll murder you", "i will murder you", "i'm going to murder you",
                "i'll shoot you", "i will shoot you", "i'm going to shoot you",
                "i'll stab you", "i will stab you", "i'm going to stab you",
                "i'll hurt you", "i will hurt you", "i'm going to hurt you",
                "i'll beat you", "i will beat you", "i'm going to beat you",
                "i'll attack you", "i will attack you", "i'm going to attack you",
                "i'll choke you", "i will choke you",
                "i'll break your neck", "i will break your neck",
                "i'll break your bones", "i will break your bones",
                "i'll bury you", "i will bury you",
                "i'll end you", "i will end you",
                "you're dead", "you are dead", "you're finished", "you are finished",
                "you're done", "you are done",
                "you won't survive", "you won't make it",
                "i have a gun", "i've got a gun", "i got a gun",
                "i have a knife", "i've got a knife", "i got a knife",
                "i have a weapon", "i've got a weapon",
                "hands up", "don't move", "freeze right there",
                "get down", "get on the ground", "face down now",
                "give me your wallet", "give me your phone",
                "empty your pockets", "drop your bag now",
                "get in the car", "get in the van",
                "come with me now", "move right now",
                "i'm coming for you", "i am coming for you",
                "i'll find you", "i will find you",
                "i'll hunt you down", "i will hunt you down",
                "watch me kill you", "i'll make you bleed",
                "i'll crush you", "i'll destroy you",
                "i'll take you out", "i'll finish you",
                "you die tonight", "this ends tonight",
                "say goodbye", "your time is up",
                "you're next", "you are next",
                "i'll make you suffer", "i'll ruin your life",
                "i'll drag you out", "i'll smash your face",
                "i'll cut you", "i'll jump you"
            ],
        }
    },

    "tier_2": {
        "score": 0.65,
        "phrases": {
            # Indirect threats / distress
            "indirect": [
                "somebody help", "someone help", "help me please",
                "help me now", "please help me",
                "call the police", "call 999", "call the cops",
                "call 911", "get security now",
                "he has a weapon", "she has a weapon", "they have a weapon",
                "he has a gun", "she has a gun", "they have a gun",
                "he has a knife", "she has a knife", "they have a knife",
                "stop following me", "you're following me", "he's following me",
                "she's following me", "they're following me",
                "get away from me", "stay away from me", "leave me alone",
                "back away now", "do not touch me",
                "i'm being attacked", "i'm being mugged", "i'm being robbed",
                "i'm being followed", "i'm being threatened",
                "let go of me", "get off me", "get your hands off me",
                "stop touching me", "stop grabbing me",
                "i don't know you", "i don't want trouble",
                "please stop", "leave us alone",
                "he won't leave me alone", "she won't leave me alone",
                "they won't leave me alone",
                "this man is threatening me", "this person is threatening me",
                "i feel unsafe", "i need help now",
                "send help", "please call someone",
                "he's blocking me", "she's blocking me",
                "they're blocking me", "i can't get out",
                "don't come closer", "stay back",
                "i said no", "stop right now",
                "he's yelling at me", "she's yelling at me",
                "they're threatening me", "i need the police",
                "help us", "we need help",
                "i'm scared", "i'm terrified",
                "please hurry", "someone intervene",
                "this is harassment", "he's cornering me",
                "she's cornering me", "they're cornering me",
                "i need security", "please make him stop",
                "please make her stop", "please make them stop"
            ],

            # Conditional threats
            "conditional": [
                "do what i say or", "do as i say or",
                "if you scream i'll", "if you run i'll", "if you move i'll",
                "if you speak i'll", "if you tell anyone i'll",
                "or i'll hurt you", "or i'll kill you", "or else",
                "don't make me hurt you", "don't make me do this",
                "don't test me", "don't try anything",
                "one more word and", "say another word and",
                "you'll regret it", "you'll regret this",
                "if you leave i'll", "if you call police i'll",
                "if you resist i'll", "if you fight back i'll",
                "if you look at me again i'll", "if you touch me i'll",
                "keep talking and i'll", "keep laughing and i'll",
                "move again and i'll", "take one step and i'll",
                "try me and see", "push me and see",
                "if you disobey i'll", "if you lie i'll",
                "if you betray me i'll", "if you say no i'll",
                "if you walk away i'll", "if you open that door i'll",
                "if you hang up i'll", "if you block me i'll",
                "if you ignore me i'll", "if you make me angry i'll",
                "if you embarrass me i'll", "if you report me i'll",
                "if you run now i'll", "if you call for help i'll",
                "don't force my hand", "don't make this worse",
                "cooperate or else", "listen or else",
                "do it now or else", "answer me or else",
                "last chance or", "obey me or",
                "if i lose control you'll", "if i snap you'll",
                "if you keep this up i'll", "if you cross me i'll",
                "if you expose me i'll", "if you touch that i'll",
                "if you warn them i'll", "if you tell them i'll",
                "if you scream again i'll", "if you look away i'll",
                "if you don't comply i'll", "if you argue i'll",
                "if you challenge me i'll", "if you run from me i'll"
            ],
        }
    },

    "tier_3": {
        "score": 0.35,
        "phrases": {
            # Implied threats
            "implied": [
                "you know what happens", "you know what i'm capable of",
                "you know what i did last time", "remember what happened",
                "last warning", "final warning", "this is your last chance",
                "i'm warning you", "consider yourself warned",
                "you brought this on yourself", "you asked for this",
                "i know where you live", "i know where you work",
                "i know where your family is", "i know your address",
                "this isn't over", "it's not over",
                "watch your back", "you better watch yourself",
                "don't walk home alone", "be careful walking home",
                "accidents happen", "people disappear",
                "bad things happen", "shame if something happened",
                "nice place you have", "hope nothing happens",
                "sleep with one eye open", "tick tock",
                "your days are numbered", "you've made a mistake",
                "this won't end well", "actions have consequences",
                "you'll learn soon enough", "i never forget",
                "i know your schedule", "keep your door locked",
                "be careful tomorrow", "you won't see it coming",
                "you'll get what's coming", "some people need lessons",
                "i'd hate to lose control", "don't push me",
                "you crossed the line", "you'll regret that choice",
                "things can get ugly", "i can't promise safety",
                "this is your final chance", "keep laughing",
                "you'll remember this day", "i hope you're prepared",
                "we'll settle this later", "see you soon",
                "i know people", "i have ways of dealing with this",
                "handle it before i do", "don't make me come there",
                "you're playing a dangerous game", "careful what you wish for",
                "you should be nervous", "you should worry",
                "better sleep lightly", "you'll pay somehow",
                "not smart to ignore me", "i wouldn't do that",
                "wrong move", "huge mistake",
                "you've been warned", "remember this moment",
                "i'm keeping score", "i'm not done with you",
                "soon you'll understand", "wait and see",
                "your turn is coming", "nothing stays hidden",
                "i see everything", "i know more than you think",
                "you'll wish you listened", "we're not finished",
                "count your days", "keep looking over your shoulder",
                "i'll be around", "you know why",
                "you know what's next", "there will be consequences",
                "time is almost up", "clock is ticking"
            ],
        }
    }
}

# building a flat index: (phrase, tier_score, threat_type) from the dictionary above

def build_keyword_index(tiers: dict) -> list[tuple[str, float, str]]:
    index = []
    for tier_name, tier_data in tiers.items():
        score = tier_data["score"]
        for threat_type, phrases in tier_data["phrases"].items():
            for phrase in phrases:
                index.append((phrase, score, threat_type))
    return sorted(index, key=lambda x: len(x[0]), reverse=True)

KEYWORD_INDEX = build_keyword_index(KEYWORD_TIERS)

# ─────────────────────────────────────────────
# TONE SCORE MAPPING (50% weight)
# ─────────────────────────────────────────────

TONE_SCORES = {
    "aggressive": 1.0,
    "angry":      0.9,
    "fearful":    0.75,
    "distressed": 0.6,
    "tense":      0.5,
    "neutral":    0.1,
    "calm":       0.0,
}

# ─────────────────────────────────────────────
# VOLUME SENSITIVITY
# Raw RMS is typically 0.001–0.05 for speech.
# We map this to 0–1 using a log scale so that
# quiet speech still registers and loud shouting
# saturates toward 1.0.
# ─────────────────────────────────────────────

VOLUME_FLOOR = 0.001   # below this = silence
VOLUME_CEIL  = 0.15    # above this = saturates at 1.0

def normalise_volume(raw_rms: float) -> float:
    """
    Map raw RMS amplitude to a 0–1 threat-relevant volume score.
    Uses log scaling so small increases in loud audio still register.
    """
    import math
    rms = max(raw_rms, VOLUME_FLOOR)
    log_val = math.log10(rms / VOLUME_FLOOR)
    log_ceil = math.log10(VOLUME_CEIL / VOLUME_FLOOR)
    return round(min(log_val / log_ceil, 1.0), 4)


# ─────────────────────────────────────────────
# LLM JSON SCHEMA
# ─────────────────────────────────────────────

THREAT_SCHEMA = {
    "type": "object",
    "properties": {
        "threat_type": {
            "type": "string",
            "enum": ["direct", "indirect", "conditional", "implied", "none"],
        },
        "directionality": {
            "type": "string",
            "enum": ["targeted", "ambient", "undetermined"],
        },
        "tone_assessment": {
            "type": "object",
            "properties": {
                "tone": {
                    "type": "string",
                    "enum": ["aggressive", "angry", "fearful", "distressed", "tense", "neutral", "calm"]
                },
                "tone_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "tone_reasoning": {
                    "type": "string"
                }
            },
            "required": ["tone", "tone_confidence", "tone_reasoning"],
            "additionalProperties": False
        },
        "threat_score_raw": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "LLM's own confidence this is a genuine threat before weighting"
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence for the human operator explaining threat score"
        },
        "keywords_detected": {
            "type": "array",
            "items": {"type": "string"}
        },
    },
    "required": [
        "threat_type",
        "directionality",
        "tone_assessment",
        "threat_score_raw",
        "reasoning",
        "keywords_detected",
    ],
    "additionalProperties": False
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """UK Safety Wearable: Classify audio transcript.
THREAT TYPES: 
- direct: explicit threat of immediate harm
- indirect: wearer is signalling distress or calling for help
- conditional: harm contingent on the wearer's actions
- implied: non-explicit but clearly threatening ("watch your back", "I know where you live", "last warning")
- none: no threat present, casual conversation

DIRECTIONALITY:
- targeted: threat or distress is aimed at / coming from the wearer's immediate situation
- ambient: audio is clearly from background media (TV, radio, film, podcast) or it represents a conversation between others — NOT directed at wearer
- undetermined: cannot tell from transcript alone

TONE — this is your most important signal:
- aggressive: hostile, dominant, intimidating language
- angry: speaking fast, speaker sounds mad
- fearful: speaker sounds scared, trembling, pleading
- distressed: panic, urgency, desperation
- tense: elevated stress, clipped responses, heightened alertness
- neutral: matter-of-fact, no emotional charge
- calm: relaxed, conversational

SCORING GUIDANCE for threat_score_raw:
- 0.0 to 0.2: clearly safe
- 0.3 to 0.5: mildly concerning
- 0.6 to 0.8: likely threat
- 0.9 to 1.0: certain threat

RULES:
- If TV/Film/Podcast: DIRECTION = ambient
- Bias: Conservative (False Positive > Missed Threat)
- Output: VALID JSON ONLY. No preamble."""


def build_user_prompt(transcript: str, duration: float, word_count: int) -> str:
    return f"""Assess this transcript from a personal safety wearable device:

Transcript: "{transcript}"
Duration: {duration}s  
Word count: {word_count}

Classify the threat type, directionality, and tone."""


# ─────────────────────────────────────────────
# STEP A — Keyword check
# ─────────────────────────────────────────────

def run_keyword_check(transcript: str) -> dict:
    """
    Fast deterministic scan. No API call.
    Returns best match (highest tier score found).
    """
    text = transcript.lower()
    best_match = None

    for phrase, tier_score, threat_type in KEYWORD_INDEX:
        if phrase in text:
            if best_match is None or tier_score > best_match["tier_score"]:
                best_match = {
                    "keyword_match": True,
                    "matched_phrase": phrase,
                    "detected_threat_type": threat_type,
                    "tier_score": tier_score,
                }

    if best_match:
        return best_match

    return {
        "keyword_match": False,
        "matched_phrase": None,
        "detected_threat_type": None,
        "tier_score": 0.0,
    }


# ─────────────────────────────────────────────
# STEP B — LLM assessment
# ─────────────────────────────────────────────

def run_llm_detection(record: dict) -> dict:
    """Send transcript to GPT-4o with structured JSON schema."""
    transcript = record["clean_transcript"]

    if not transcript or record["word_count"] == 0:
        return {
            "threat_type": "none",
            "directionality": "undetermined",
            "tone_assessment": {
                "tone": "neutral",
                "tone_confidence": 0.0,
                "tone_reasoning": "Empty transcript — no assessment possible."
            },
            "threat_score_raw": 0.0,
            "reasoning": "Empty or inaudible transcript.",
            "keywords_detected": [],
            "llm_latency_ms": 0
        }

    t_start = time.perf_counter()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                transcript,
                record["duration_seconds"],
                record["word_count"]
            )}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "threat_assessment",
                "strict": True,
                "schema": THREAT_SCHEMA
            }
        },
        temperature=0,
        max_tokens=200
    )

    latency_ms = int((time.perf_counter() - t_start) * 1000)
    result = json.loads(response.choices[0].message.content)
    result["llm_latency_ms"] = latency_ms

    return result


# ─────────────────────────────────────────────
# STEP C — Tension escalation (10% weight)
# Looks at score trend across last 3 turns
# ─────────────────────────────────────────────

def compute_escalation_score(turn_history: list[dict]) -> float:
    """
    Returns 0.0–1.0 based on how fast scores are rising.
    Needs at least 2 previous turns to compute.
    """
    if len(turn_history) < 2:
        return 0.0

    recent = turn_history[-3:]
    scores = [t.get("fused_score", 0.0) for t in recent]

    if len(scores) < 2:
        return 0.0

    delta = scores[-1] - scores[0]
    num_steps = len(scores) - 1
    escalation_rate = delta / num_steps

    return max(0.0, min(escalation_rate, 1.0))


# ─────────────────────────────────────────────
# STEP D — Weighted signal fusion
#
# Weights (sum to 1.0):
#   Keywords:         32%
#   Tone:             48%
#   Turn escalation:  10%
#   Volume:            6%
#   Audio escalation:  4%
# ─────────────────────────────────────────────

def compute_audio_escalation(rms_history: list[float]) -> float:
    """
    Rate of volume change between the current turn and the previous turn.
    Returns 0–1: positive = audio got louder, 0 = same or quieter.
    """
    if len(rms_history) < 2:
        return 0.0
    
    prev = normalise_volume(rms_history[-2])
    curr = normalise_volume(rms_history[-1])
    delta = curr - prev
    
    # Only positive change is a threat signal
    return round(max(-1.0, min(delta, 1.0)), 4)

def fuse_signals(
    keyword_result: dict,
    llm_result: dict,
    escalation_score: float,
    volume_score: float,
    escalation_audio_score: float,
) -> dict:
    """
    Combine all signals using weighted scoring.

    Args:
        keyword_result:         output of run_keyword_check()
        llm_result:             output of run_llm_detection()
        escalation_score:       0–1 turn-history escalation signal
        volume_score:           0–1 log-normalised RMS volume (via normalise_volume())
        escalation_audio_score: 0–1 rate-of-change in audio energy across turns

    Returns:
        dict with fused_score, severity, and score_breakdown
    """

    # — Keyword score (32%) —
    keyword_weighted = keyword_result["tier_score"] * 0.32

    # — Tone score (48%) —
    tone = llm_result["tone_assessment"]["tone"]
    tone_confidence = llm_result["tone_assessment"]["tone_confidence"]
    tone_base = TONE_SCORES.get(tone, 0.0)
    tone_weighted = tone_base * tone_confidence * 0.48

    # — Turn-history escalation (10%) —
    escalation_weighted = escalation_score * 0.10

    # — Volume (6%) — expects log-normalised score from normalise_volume() —
    volume_weighted = float(volume_score) * 0.06

    # — Audio escalation (4%) —
    escalation_audio_weighted = float(escalation_audio_score) * 0.04

    # — Final fused score —
    fused_score = min(
        keyword_weighted
        + tone_weighted
        + escalation_weighted
        + volume_weighted
        + escalation_audio_weighted,
        1.0
    )

    # — Ambient override — never alert on background media —
    if llm_result["directionality"] == "ambient":
        fused_score = min(fused_score, 0.15)

    # — Severity banding —
    if fused_score >= 0.75:
        severity = "critical"
    elif fused_score >= 0.50:
        severity = "warning"
    elif fused_score >= 0.30:
        severity = "monitor"
    else:
        severity = "safe"

    return {
        "fused_score": round(fused_score, 4),
        "severity": severity,
        "score_breakdown": {
            "keyword":          round(keyword_weighted, 4),
            "tone":             round(tone_weighted, 4),
            "escalation":       round(escalation_weighted, 4),
            "volume":           round(volume_weighted, 4),
            "audio_escalation": round(escalation_audio_weighted, 4),
        }
    }


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────



def assess_threat(
    record: dict,
    turn_history: list[dict] = [],
    rms_history: list[float] = [],
) -> dict:
    """
    Full threat assessment for one turn.

    Args:
        record:                 transcript record from transcription.py
        turn_history:           list of previous assess_threat() outputs (for escalation)
        raw_rms:                raw RMS amplitude from audio_processing (will be log-normalised here)
        escalation_audio_score: 0–1 rate-of-change in audio energy across turns

    Returns:
        Full structured assessment dict ready for alerting
    """

    stage_latency_ms = {}

    # A — Fast keyword scan
    t_stage = time.perf_counter()
    keyword_result = run_keyword_check(record["clean_transcript"])
    stage_latency_ms["keyword_scan"] = int((time.perf_counter() - t_stage) * 1000)

    # B — LLM assessment
    t_stage = time.perf_counter()
    llm_result = run_llm_detection(record)
    stage_latency_ms["llm_assessment"] = int((time.perf_counter() - t_stage) * 1000)

    # C — Turn-history escalation
    t_stage = time.perf_counter()
    escalation_score = compute_escalation_score(turn_history)
    stage_latency_ms["history_escalation"] = int((time.perf_counter() - t_stage) * 1000)

    # D — Compute volume and audio escalation from rms_history
    t_stage = time.perf_counter()
    raw_rms = rms_history[-1] if rms_history else 0.0
    volume_score = normalise_volume(raw_rms)
    escalation_audio_score = compute_audio_escalation(rms_history)
    stage_latency_ms["audio_features"] = int((time.perf_counter() - t_stage) * 1000)

    # E — Fuse everything
    t_stage = time.perf_counter()
    fusion = fuse_signals(
        keyword_result,
        llm_result,
        escalation_score,
        volume_score,
        escalation_audio_score,
    )
    stage_latency_ms["signal_fusion"] = int((time.perf_counter() - t_stage) * 1000)

    # F — Keyword override safety net:
    # If heuristic caught a tier_1 phrase but LLM says "none", trust the keyword
    if (keyword_result["keyword_match"]
            and keyword_result["tier_score"] == 1.0
            and llm_result["threat_type"] == "none"):
        llm_result["threat_type"] = keyword_result["detected_threat_type"]
        llm_result["reasoning"] += " [keyword override: tier-1 phrase detected]"
        fusion["fused_score"] = max(fusion["fused_score"], 0.55)
        fusion["severity"] = "warning" if fusion["fused_score"] < 0.75 else "critical"

    return {
        "turn_id":                record["turn_id"],
        "badge_id":               record["badge_id"],
        "timestamp":              record["timestamp"],
        "transcript":             record["clean_transcript"],
        "threat_type":            llm_result["threat_type"],
        "directionality":         llm_result["directionality"],
        "tone":                   llm_result["tone_assessment"]["tone"],
        "tone_confidence":        llm_result["tone_assessment"]["tone_confidence"],
        "tone_reasoning":         llm_result["tone_assessment"]["tone_reasoning"],
        "keyword_check":          keyword_result,
        "escalation_score":       round(escalation_score, 4),
        "volume_score":           volume_score,
        "audio_escalation_score": round(float(escalation_audio_score), 4),
        "fused_score":            fusion["fused_score"],
        "severity":               fusion["severity"],
        "score_breakdown":        fusion["score_breakdown"],
        "reasoning":              llm_result["reasoning"],
        "latency_ms": {
            **record["latency_ms"],
            "llm": llm_result["llm_latency_ms"],
            **stage_latency_ms,
        }
    }


# ─────────────────────────────────────────────
# TEST BLOCK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from incident_manager import IncidentManager
    from publisher import (
        build_alert_event,
        publish_alert,
        start_subscriber,
        stop_subscriber,
    )

    ROOT = Path(__file__).parent.parent
    filepath = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "audio" / "heated_argument.wav")

    # Start subscriber before pipeline
    start_subscriber()

    recording_start_time = datetime.now(timezone.utc).isoformat()
    turn_history = []
    rms_history  = []
    manager      = IncidentManager()

    for i, (turn_audio, sr) in enumerate(ingest(filepath)):
        t_turn_start = time.perf_counter()
        t_stage = time.perf_counter()
        audio_float = turn_audio.astype(np.float32)
        raw_rms = float(np.sqrt(np.mean(audio_float ** 2)))
        rms_history.append(raw_rms)
        ingest_ms = int((time.perf_counter() - t_stage) * 1000)

        t_stage = time.perf_counter()
        record     = build_transcript_record(turn_audio, sr)
        transcription_ms = int((time.perf_counter() - t_stage) * 1000)

        t_stage = time.perf_counter()
        assessment = assess_threat(
            record,
            turn_history=turn_history,
            rms_history=rms_history,
        )
        assess_ms = int((time.perf_counter() - t_stage) * 1000)

        t_stage = time.perf_counter()
        incident_result = manager.process(assessment)
        incident_ms = int((time.perf_counter() - t_stage) * 1000)
        publish_ms = 0

        # Only publish if an action was actually fired
        if incident_result["decision"] == "action_fired":
            t_stage = time.perf_counter()
            alert_event = build_alert_event(
                assessment,
                incident_result,
                recording_start_time,
            )
            publish_alert(alert_event)
            publish_ms = int((time.perf_counter() - t_stage) * 1000)

        assessment["latency_ms"]["pipeline"] = {
            "ingest": ingest_ms,
            "transcription_record": transcription_ms,
            "threat_assessment": assess_ms,
            "incident_decision": incident_ms,
            "publish_alert": publish_ms,
            "turn_total": int((time.perf_counter() - t_turn_start) * 1000),
        }

        # print(f"\n{'='*50}")
        # print(f"Turn {i+1}")
        # print(f"{'='*50}")
        # print(f"Transcript:    {assessment['transcript']}")
        # print(f"Fused score:   {assessment['fused_score']}")
        # print(f"Alert level:   {incident_result['alert_level']}")
        # print(f"Decision:      {incident_result['decision']}")

        turn_history.append(assessment)

    stop_subscriber()