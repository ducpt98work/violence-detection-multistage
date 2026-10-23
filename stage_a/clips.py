"""From forwarded windows to Stage B invocations (Sec. III-A-5, Algorithm 1, lines 14-19).

Each forwarded window asks for the 16 analysis frames t-8 .. t+7 around its trigger. Forwarded
windows of the same group whose triggers fall within one open clip span are merged into a single
Stage B invocation, anchored at the earliest trigger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple


@dataclass
class PendingClip:
    clip_no: int
    anchor: int                          # analysis index of the earliest trigger
    lo: int                              # first analysis frame of the clip (anchor - before)
    hi: int                              # last analysis frame of the clip (anchor + after)
    members: Set[int] = field(default_factory=set)
    triggers: List[Tuple[int, Tuple[int, ...], float]] = field(default_factory=list)  # (aidx, members, score)
    payload: list = field(default_factory=list)   # caller data per trigger (e.g. WindowSample)

    @property
    def n_merged(self) -> int:
        return len(self.triggers)


class ClipMerger:
    def __init__(self, frames_before: int = 8, frames_after: int = 7):
        self.before = frames_before
        self.after = frames_after
        self.open: List[PendingClip] = []
        self._next = 0

    def add(self, aidx: int, members: Sequence[int], score: float = float("nan"),
            payload=None) -> Tuple[PendingClip, bool]:
        """Register a forwarded window. Returns (clip, created_new)."""
        mset = set(members)
        for clip in self.open:
            if clip.lo <= aidx <= clip.hi and clip.members & mset:
                clip.members |= mset
                clip.triggers.append((aidx, tuple(members), score))
                if payload is not None:
                    clip.payload.append(payload)
                return clip, False
        clip = PendingClip(self._next, aidx, aidx - self.before, aidx + self.after, mset,
                           [(aidx, tuple(members), score)], [payload] if payload is not None else [])
        self._next += 1
        self.open.append(clip)
        return clip, True

    def pop_ready(self, aidx: Optional[int] = None) -> List[PendingClip]:
        """Clips whose last frame (hi) has been reached; all of them when aidx is None."""
        ready = [c for c in self.open if aidx is None or c.hi <= aidx]
        self.open = [c for c in self.open if c not in ready]
        return ready


def count_invocations(triggers: Sequence[Tuple[int, Sequence[int]]], frames_before: int = 8,
                      frames_after: int = 7) -> int:
    """Number of Stage B clip evaluations one video costs, given its forwarded (aidx, members)."""
    merger = ClipMerger(frames_before, frames_after)
    n = 0
    for aidx, members in sorted(triggers, key=lambda t: t[0]):
        # clips that already ended can no longer absorb a trigger
        merger.pop_ready(aidx - 1)
        _, new = merger.add(aidx, members)
        n += int(new)
    return n
