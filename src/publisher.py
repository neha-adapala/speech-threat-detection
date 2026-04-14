"""
publisher.py

Real-time alert publishing for the eNO badge.
Uses an in-process queue to simulate a message streaming service (e.g. Pub/Sub).
A background subscriber thread consumes and prints alerts as they arrive,
mimicking a real ARC dashboard or emergency contact system.

In production: replace the in-process queue with google.cloud.pubsub_v1
and push to a real Pub/Sub topic.
"""

import json
import os
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# MESSAGE BUS (simulated Pub/Sub)
# ─────────────────────────────────────────────

_alert_queue: queue.Queue = queue.Queue()
_subscriber_thread: Optional[threading.Thread] = None


def _use_pubsub() -> bool:
    return bool(os.getenv("PUBSUB_EMULATOR_HOST"))


def _pubsub_config() -> tuple[str, str, str]:
    project_id = os.getenv("PUBSUB_PROJECT_ID", "eno-local")
    topic_id = os.getenv("PUBSUB_TOPIC", "alerts")
    subscription_id = os.getenv("PUBSUB_SUBSCRIPTION", "alerts-sub")
    return project_id, topic_id, subscription_id


# ─────────────────────────────────────────────
# ALERT SCHEMA BUILDER
# ─────────────────────────────────────────────

def build_alert_event(
    assessment: dict,
    incident_result: dict,
    recording_start_time: str,
) -> dict:
    """
    Build a fully-structured alert event from the assessment
    and incident manager output.

    Args:
        assessment:           output of assess_threat()
        incident_result:      output of IncidentManager.process()
        recording_start_time: ISO timestamp when the WAV/session started

    Returns:
        Alert event dict matching the eNO alert schema.
    """

    alert_level  = incident_result["alert_level"]
    action       = incident_result["action_taken"]

    # ── action_reason: human-readable explanation for the operator ──
    reason_parts = []

    # Tone
    tone     = assessment["tone"]
    tone_conf = assessment["tone_confidence"]
    reason_parts.append(
        f"Tone detected as '{tone}' (confidence: {tone_conf:.0%}). "
        f"{assessment['tone_reasoning']}"
    )

    # Keywords
    kw = assessment["keyword_check"]
    if kw["keyword_match"]:
        reason_parts.append(
            f"Keyword match: '{kw['matched_phrase']}' "
            f"(type: {kw['detected_threat_type']}, "
            f"tier score: {kw['tier_score']})."
        )
    else:
        reason_parts.append("No direct keyword match found.")

    # Escalation
    esc = assessment["escalation_score"]
    if esc > 0:
        reason_parts.append(
            f"Situation escalating — turn escalation score: {esc:.2f}."
        )

    # Audio
    vol  = assessment["volume_score"]
    adelta = assessment["audio_escalation_score"]
    reason_parts.append(
        f"Audio signals — volume: {vol:.2f}, "
        f"audio escalation: {adelta:.2f}."
    )

    # LLM reasoning
    reason_parts.append(f"Model reasoning: {assessment['reasoning']}")

    # Inside build_alert_event
    if alert_level in ["high", "medium"]:
        # High-priority summary for the messaging app
        action_reason = f"EMERGENCY: {tone.upper()} tone & {kw['detected_threat_type']} detected."
    else:
        # Full technical breakdown for the logs
        action_reason = " ".join(reason_parts)

    return {
        "badge_id":             assessment["badge_id"],
        "alert_id":             str(uuid.uuid4()),
        "recording_start_time": recording_start_time,
        "turn_start_time":      assessment["timestamp"],
        "transcription":        assessment["transcript"],
        "alert_level":          alert_level,
        "action":               action,
        "action_reason":        action_reason,
    }


# ─────────────────────────────────────────────
# PUBLISHER
# ─────────────────────────────────────────────

def publish_alert(alert_event: dict):
    """
    Only publish alerts that require active intervention.
    """
    # Define which actions are considered "messaging app alerts"
    meaningful_actions = ["emergency_contact_alert", "arc_operator_alert"]
    
    if alert_event.get("action") in meaningful_actions:
        if _use_pubsub():
            from google.cloud import pubsub_v1

            project_id, topic_id, _ = _pubsub_config()
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(project_id, topic_id)
            payload = json.dumps(alert_event).encode("utf-8")
            publisher.publish(topic_path, payload).result(timeout=10)
        else:
            _alert_queue.put(alert_event)
    else:
        # Do nothing for non-alert actions to keep terminal output clean.
        pass


# ─────────────────────────────────────────────
# SUBSCRIBER (background thread)
# Mimics an ARC dashboard or emergency contact system consuming events
# ─────────────────────────────────────────────

def _format_message(event: dict) -> str:
    """
    Format the alert event into a human-readable message
    for the operator / emergency contact.
    """
    level_emoji = {
        "very_low": "🟢",
        "low":      "🔵",
        "medium":   "🟡",
        "high":     "🔴",
    }.get(event["alert_level"], "⚪")

    action_label = {
        "emergency_contact_alert": "📱 Emergency contacts notified",
        "arc_operator_alert":      "🚨 ARC operator alerted",
    }.get(event["action"], "ℹ️  Logged only")

    lines = [
        "",
        "┌─────────────────────────────────────────────────┐",
        f"│  ALERT  {level_emoji}  {event['alert_level'].upper():<10}                      │",
        f"│  ACTION      : {action_label:<34} │",
        "├─────────────────────────────────────────────────┤",
        f"│  Alert ID    : {event['alert_id'][:8]}...                     │",
        f"│  Badge       : {event['badge_id']:<34} │",
        "├─────────────────────────────────────────────────┤",
        f"│  TRANSCRIPTION                                  │",
        f"│  \"{event['transcription'][:100]}\"",
        "├─────────────────────────────────────────────────┤",
        f"│  Session started : {event['recording_start_time'][:19]}              │",
        f"│  Turn started    : {event['turn_start_time'][:19]}              │",
        "├─────────────────────────────────────────────────┤",
        f"│  REASON                                         │",
    ]

    # Word-wrap the reason across multiple lines at 49 chars
    reason = event["action_reason"]
    words  = reason.split()
    line   = "│  "
    for word in words:
        if len(line) + len(word) + 1 > 51:
            lines.append(f"{line:<51} │")
            line = "│  " + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(f"{line:<51} │")

    lines.append("└─────────────────────────────────────────────────┘")
    return "\n".join(lines)


def _subscriber_loop():
    while True:
        event = _alert_queue.get()
        if event is None:
            _alert_queue.task_done()
            break
            
        # Only print/process if it's a real alert level
        if event["alert_level"] in ["medium", "high"]:
            print(_format_message(event))
            
        _alert_queue.task_done()


def start_subscriber() -> threading.Thread:
    """Start the background subscriber. Call once at pipeline startup."""
    if _use_pubsub():
        # In Pub/Sub mode, run a standalone subscriber process/service.
        return threading.current_thread()

    global _subscriber_thread
    t = threading.Thread(target=_subscriber_loop, daemon=False)
    t.start()
    _subscriber_thread = t
    return t


def stop_subscriber():
    """Gracefully stop the subscriber by sending the sentinel."""
    if _use_pubsub():
        return

    global _subscriber_thread
    _alert_queue.put(None)
    if _subscriber_thread is not None:
        _subscriber_thread.join()
        _subscriber_thread = None


def subscribe_forever():
    """Blocking Pub/Sub subscriber loop for Docker/production-style deployments."""
    if not _use_pubsub():
        _subscriber_loop()
        return

    from google.cloud import pubsub_v1

    project_id, _, subscription_id = _pubsub_config()
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    def _callback(message):
        try:
            event = json.loads(message.data.decode("utf-8"))
            if event.get("alert_level") in ["medium", "high"]:
                print(_format_message(event))
            message.ack()
        except Exception:
            message.nack()

    streaming_future = subscriber.subscribe(subscription_path, callback=_callback)
    try:
        streaming_future.result()
    finally:
        subscriber.close()