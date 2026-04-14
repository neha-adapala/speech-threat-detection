#!/usr/bin/env python3
"""
Simple CLI to run the eNO pipeline on sample audio.

Usage:
  python scripts/run_pipeline.py --audio /path/to/file.wav
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from threat_detection import (  # pylint: disable=import-error
        assess_threat,
    )
    from audio_processing import ingest  # pylint: disable=import-error
    from transcription import build_transcript_record  # pylint: disable=import-error
    from incident_manager import IncidentManager  # pylint: disable=import-error
    from publisher import (  # pylint: disable=import-error
        build_alert_event,
        publish_alert,
        start_subscriber,
        stop_subscriber,
    )
    import numpy as np
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Run eNO threat pipeline on sample audio.")
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input WAV file (sample audio).",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_subscriber()

    recording_start_time = datetime.now(timezone.utc).isoformat()
    turn_history = []
    rms_history = []
    manager = IncidentManager()

    for turn_audio, sr in ingest(str(audio_path)):
        audio_float = turn_audio.astype(np.float32)
        raw_rms = float(np.sqrt(np.mean(audio_float ** 2)))
        rms_history.append(raw_rms)

        record = build_transcript_record(turn_audio, sr)
        assessment = assess_threat(
            record,
            turn_history=turn_history,
            rms_history=rms_history,
        )

        incident_result = manager.process(assessment)
        if incident_result["decision"] == "action_fired":
            alert_event = build_alert_event(
                assessment,
                incident_result,
                recording_start_time,
            )
            publish_alert(alert_event)

        turn_history.append(assessment)

    stop_subscriber()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
