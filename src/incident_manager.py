"""
incident_manager.py

Turn-by-turn incident logic for the eNO badge.

Alert levels (based on fused_score):
    very_low  : < 0.3   → no action
    low       : 0.3–0.6 → no action (logged only)
    medium    : 0.6–0.8 → send location + message to emergency contacts
    high      : 0.8+    → send info to downstream human operator (ARC)

Incident rules:
    - Each turn is evaluated independently against the current active action level
    - If incoming level == same or lower than active action → suppress, no new action
    - If incoming level is higher than active action → cancel current, trigger new action
    - If level drops to very_low or low → incident closes, active action clears
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# ALERT LEVEL DEFINITIONS
# ─────────────────────────────────────────────

# Numeric rank so we can compare levels easily
LEVEL_RANK = {
    "very_low": 0,
    "low":      1,
    "medium":   2,
    "high":     3,
}

def score_to_alert_level(fused_score: float) -> str:
    """Map fused_score to alert level string."""
    if fused_score >= 0.8:
        return "high"
    elif fused_score >= 0.6:
        return "medium"
    elif fused_score >= 0.3:
        return "low"
    else:
        return "very_low"


# ─────────────────────────────────────────────
# ACTIONS
# These are stubs — replace with real integrations
# (Twilio for SMS, ARC webhook, etc.)
# ─────────────────────────────────────────────

def action_medium(assessment: dict, incident_id: str) -> dict:
    """
    MEDIUM action: send location + message to emergency contacts.
    Stub — in production this hits Twilio SMS / push notification API.
    """
    payload = {
        "action":      "emergency_contact_alert",
        "incident_id": incident_id,
        "badge_id":    assessment["badge_id"],
        "timestamp":   assessment["timestamp"],
        "message":     (
            f"Safety alert from badge {assessment['badge_id']}. "
            f"Situation detected: {assessment['threat_type']} threat. "
            f"Tone: {assessment['tone']}. "
            f"Reason: {assessment['reasoning']}"
        ),
        "location":    "LIVE_LOCATION_STUB",   # replace with real GPS in production
        "transcript":  assessment["transcript"],
        "fused_score": assessment["fused_score"],
    }
    # TODO: POST to Twilio / push notification service
    return payload


def action_high(assessment: dict, incident_id: str) -> dict:
    """
    HIGH action: publish full alert to downstream human operator (ARC).
    Stub — in production this publishes to Pub/Sub topic → ARC dashboard.
    """
    payload = {
        "action":           "arc_operator_alert",
        "incident_id":      incident_id,
        "badge_id":         assessment["badge_id"],
        "timestamp":        assessment["timestamp"],
        "alert_level":      "high",
        "threat_type":      assessment["threat_type"],
        "directionality":   assessment["directionality"],
        "tone":             assessment["tone"],
        "tone_confidence":  assessment["tone_confidence"],
        "fused_score":      assessment["fused_score"],
        "score_breakdown":  assessment["score_breakdown"],
        "transcript":       assessment["transcript"],
        "reasoning":        assessment["reasoning"],
        "keyword_check":    assessment["keyword_check"],
        "location":         "LIVE_LOCATION_STUB",
        "latency_ms":       assessment["latency_ms"],
    }
    # TODO: publish to Pub/Sub / ARC webhook
    return payload


def cancel_action(active_action_level: str, incident_id: str):
    """
    Cancel the currently running action before escalating.
    Stub — in production this would cancel a pending Twilio call etc.
    """
    # TODO: call cancellation API for whatever is in flight


# ─────────────────────────────────────────────
# INCIDENT MANAGER
# ─────────────────────────────────────────────

class IncidentManager:
    """
    Stateful turn-by-turn incident tracker.

    Maintains one active incident at a time per badge.
    Compares each new turn's alert level against the current
    active action level and decides whether to act, suppress, or escalate.

    Usage:
        manager = IncidentManager()

        for turn in pipeline:
            assessment = assess_threat(...)
            result = manager.process(assessment)
            print(result)
    """

    def __init__(self):
        self.incident_id:          Optional[str]  = None   # current incident UUID
        self.active_action_level:  Optional[str]  = None   # level of last fired action
        self.incident_start:       Optional[str]  = None   # ISO timestamp
        self.turns_in_incident:    int            = 0      # turns since incident opened

    def _open_incident(self, timestamp: str):
        """Start a new incident."""
        self.incident_id         = str(uuid.uuid4())
        self.incident_start      = timestamp
        self.turns_in_incident   = 0
        self.active_action_level = None

    def _close_incident(self):
        """Clear incident state — situation has de-escalated."""
        self.incident_id         = None
        self.active_action_level = None
        self.incident_start      = None
        self.turns_in_incident   = 0

    def process(self, assessment: dict) -> dict:
        """
        Core decision function. Call once per turn with the assess_threat() output.

        Returns a result dict describing what was decided and why.
        """
        fused_score  = assessment["fused_score"]
        alert_level  = score_to_alert_level(fused_score)
        timestamp    = assessment["timestamp"]
        action_taken = None
        action_payload = None
        decision     = None

        # ── 1. Very low / low → no actionable threat ──────────────────────
        if alert_level in ("very_low", "low"):
            if self.incident_id:
                # Situation has de-escalated — close the incident
                self._close_incident()
                decision = "incident_closed"
            else:
                decision = "no_action"

        # ── 2. Medium or high → actionable ────────────────────────────────
        else:
            incoming_rank = LEVEL_RANK[alert_level]
            active_rank   = LEVEL_RANK.get(self.active_action_level, -1)

            # Open incident if none exists
            if not self.incident_id:
                self._open_incident(timestamp)

            self.turns_in_incident += 1

            # Same or lower level as active action → suppress
            if self.active_action_level and incoming_rank <= active_rank:
                decision = "suppressed"

            # Higher level than active → cancel current, escalate
            else:
                if self.active_action_level:
                    cancel_action(self.active_action_level, self.incident_id)

                # Fire the appropriate action
                if alert_level == "medium":
                    action_taken   = "emergency_contact_alert"
                    action_payload = action_medium(assessment, self.incident_id)

                elif alert_level == "high":
                    action_taken   = "arc_operator_alert"
                    action_payload = action_high(assessment, self.incident_id)

                self.active_action_level = alert_level
                decision = "action_fired"

        return {
            "turn_id":             assessment["turn_id"],
            "incident_id":         self.incident_id,
            "alert_level":         alert_level,
            "fused_score":         fused_score,
            "active_action_level": self.active_action_level,
            "decision":            decision,
            "turns_in_incident":   self.turns_in_incident,
            "action_taken":        action_taken,
            "action_payload":      action_payload,
            "timestamp":           timestamp,
        }