#!/usr/bin/env python3
"""
Structured logging for the DMFT pipeline.

A thin wrapper around Python's `logging` that:
  - Tags every record with a phase label (analysis, dmft_1p, dca, vertex,
    bse, pairing, ac, csc, ...)
  - Emits one JSON-line per "physics event" (converged Σ, density matrix
    trace, Tc estimate, etc.) for downstream parsing
  - Mirrors human-readable output to stdout so existing log-tailing
    workflows keep working

Usage from a pipeline module:

    from dmft_logger import get_logger, log_event
    log = get_logger("dmft.dca", work_dir)
    log.info("Starting DCA loop with N_c=4")
    log_event(log, "dmft.density",
              n=2.45, mu=0.31, beta=40.0, n_orb=5)

The JSON-line records land in `<work_dir>/dmft_events.jsonl` so the
caller can grep / replay them without parsing the human log.
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}
_EVENT_PATHS: dict[str, str] = {}


def get_logger(
    name: str,
    work_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger with a stdout handler and (if work_dir given)
    a per-run file handler that mirrors all output to <work_dir>/dmft.log.

    Idempotent: repeated calls with the same name return the same logger.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False  # don't double-emit through the root logger

    # Clear any pre-existing handlers (test runs, repeated imports)
    for h in list(log.handlers):
        log.removeHandler(h)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Stdout handler
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    log.addHandler(stream)

    # File handler (per-run)
    if work_dir:
        try:
            os.makedirs(work_dir, exist_ok=True)
            log_path = os.path.join(work_dir, "dmft.log")
            file_h = logging.FileHandler(log_path, mode="a")
            file_h.setLevel(level)
            file_h.setFormatter(formatter)
            log.addHandler(file_h)
            _EVENT_PATHS[name] = os.path.join(work_dir, "dmft_events.jsonl")
        except OSError:
            pass

    _LOGGERS[name] = log
    return log


def log_event(
    log: logging.Logger,
    event: str,
    **fields,
) -> None:
    """
    Emit a structured event record.

    The event lands in both:
      - The human log (one-line summary at INFO level)
      - <work_dir>/dmft_events.jsonl (JSON record, one per line)

    Args:
        log:    a logger from get_logger()
        event:  short event name (e.g., "dmft.converged", "tc.estimated")
        fields: arbitrary key/value payload to log

    Convention for `event`:
        <subsystem>.<action> using lowercase + dot separator
        Examples: "dca.iter", "bse.singlet_attractive", "tc.estimated"
    """
    record = {
        "ts": time.time(),
        "event": event,
        "logger": log.name,
        **{k: _safe(v) for k, v in fields.items()},
    }

    # Human-readable one-liner
    summary = ", ".join(f"{k}={v}" for k, v in fields.items())
    log.info(f"event={event} {summary}")

    # JSON-line record
    path = _EVENT_PATHS.get(log.name)
    if path is None:
        # Look for any logger in the same name family that has a path
        for n, p in _EVENT_PATHS.items():
            if log.name.startswith(n.split(".")[0]):
                path = p
                break
    if path:
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


def _safe(value):
    """Convert any value into something JSON-serializable."""
    import numpy as np
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        if value.size > 16:
            return {
                "_array": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.min(np.abs(value))) if value.size else None,
                "max": float(np.max(np.abs(value))) if value.size else None,
            }
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)
