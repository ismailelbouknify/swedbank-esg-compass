"""Stopping a running analysis on the analyst's request.

The analysis runs in a background thread of this process, so a stop request is an in-memory flag the runner checks at
each stage change, at each discovery step and between questions; a running model call is interrupted at its next
token. A server restart clears the flags, which does not matter: a restart already ends every running analysis."""
from __future__ import annotations

import threading

STOPPED_MESSAGE = "Analysis stopped by the user."

_lock = threading.Lock()
_requested: set[int] = set()


class AnalysisStopped(Exception):
    """Raised inside the runner when the analyst asked to stop the analysis."""


def request(assessment_id: int) -> None:
    with _lock:
        _requested.add(assessment_id)


def requested(assessment_id: int) -> bool:
    with _lock:
        return assessment_id in _requested


def clear(assessment_id: int) -> None:
    with _lock:
        _requested.discard(assessment_id)
