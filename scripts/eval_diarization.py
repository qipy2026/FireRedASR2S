#!/usr/bin/env python3
"""Compute DER between two RTTM files (requires ``pyannote.metrics``).

Usage:
    .venv/Scripts/python.exe scripts/eval_diarization.py ref.rttm hyp.rttm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_rttm", type=Path)
    parser.add_argument("hyp_rttm", type=Path)
    parser.add_argument("--uri", type=str, default="audio")
    args = parser.parse_args()
    try:
        from tests.utils import der
    except ImportError:
        print("tests.utils.der unavailable", file=sys.stderr)
        return 1
    try:
        score = der(str(args.ref_rttm), str(args.hyp_rttm), file_id=args.uri)
    except ImportError as e:
        print(f"pyannote.metrics required: {e}", file=sys.stderr)
        return 2
    print(f"DER={score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
