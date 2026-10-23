"""Adaptive, perspective-aware proximity grouping (Sec. III-A-1, Eq. adaptive_threshold).

Persons i and j are linked when the distance between their box centres is below
T_adaptive = K * (H_i + H_j) / 2; groups are the connected components of the links,
so grouping is transitive across pairwise links.
"""
from __future__ import annotations

from typing import List

import numpy as np


def adaptive_groups(boxes: np.ndarray, k: float = 1.5) -> List[List[int]]:
    """boxes: (n, 4) x1, y1, x2, y2. Returns index lists, singletons included."""
    n = len(boxes)
    if n == 0:
        return []
    boxes = np.asarray(boxes, np.float64)
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    h = boxes[:, 3] - boxes[:, 1]
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    thr = k * (h[:, None] + h[None, :]) * 0.5
    linked = dist < thr

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if linked[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda c: c[0])
