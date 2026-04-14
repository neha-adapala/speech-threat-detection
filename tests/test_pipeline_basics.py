import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from publisher import build_alert_event, publish_alert, _alert_queue  # noqa: E402


class TestPipelineBasics(unittest.TestCase):
    def setUp(self):
        while not _alert_queue.empty():
            try:
                _alert_queue.get_nowait()
                _alert_queue.task_done()
            except Exception:
                break

    def test_build_alert_event_schema(self):
        assessment = {
            "badge_id": "badge_001",
            "timestamp": "2026-04-14T12:00:10+00:00",
            "transcript": "please help me",
            "tone": "angry",
            "tone_confidence": 0.92,
            "tone_reasoning": "Raised voice and hostile language.",
            "keyword_check": {
                "keyword_match": True,
                "matched_phrase": "help me",
                "detected_threat_type": "indirect",
                "tier_score": 0.65,
            },
            "escalation_score": 0.4,
            "volume_score": 0.6,
            "audio_escalation_score": 0.3,
            "reasoning": "Potential harassment context.",
        }
        incident_result = {
            "alert_level": "high",
            "action_taken": "arc_operator_alert",
        }

        event = build_alert_event(
            assessment=assessment,
            incident_result=incident_result,
            recording_start_time="2026-04-14T12:00:00+00:00",
        )

        expected_keys = {
            "badge_id",
            "alert_id",
            "recording_start_time",
            "turn_start_time",
            "transcription",
            "alert_level",
            "action",
            "action_reason",
        }
        self.assertEqual(set(event.keys()), expected_keys)
        self.assertEqual(event["badge_id"], "badge_001")
        self.assertEqual(event["alert_level"], "high")
        self.assertEqual(event["action"], "arc_operator_alert")

    def test_publish_alert_filters_non_actionable(self):
        non_actionable_event = {
            "badge_id": "badge_001",
            "alert_id": "123",
            "recording_start_time": "2026-04-14T12:00:00+00:00",
            "turn_start_time": "2026-04-14T12:00:05+00:00",
            "transcription": "normal conversation",
            "alert_level": "low",
            "action": "logged_only",
            "action_reason": "No direct threat detected.",
        }
        actionable_event = {
            **non_actionable_event,
            "alert_id": "456",
            "alert_level": "high",
            "action": "arc_operator_alert",
        }

        publish_alert(non_actionable_event)
        self.assertTrue(_alert_queue.empty())

        publish_alert(actionable_event)
        queued = _alert_queue.get_nowait()
        self.assertEqual(queued["alert_id"], "456")
        _alert_queue.task_done()


if __name__ == "__main__":
    unittest.main()
