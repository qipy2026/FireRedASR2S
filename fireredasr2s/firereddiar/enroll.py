# Copyright 2026 Xiaohongshu.
"""In-memory / JSON file speaker registry for optional enrollment matching.

**1:N 声纹匹配（开集）**：库内每条注册名对应一条 L2 归一化嵌入（多次 ``register`` 同名会滑动平均）。
推理时对查询向量与 **全部** 注册嵌入做余弦相似度，取 **argmax**；仅当最高分
``>= speaker_match_threshold`` 时返回该姓名，否则 ``None``（CAM++ SV 见
``embedder.ModelScopeCampplusEmbedder`` + ``production.CAMPLUS_SV_DEFAULT_MATCH_THRESHOLD``）。

**输出字段**（``FireRedAsr2System.process``）：整段 ``utterance_enrolled_speaker`` /
``utterance_enrolled_similarity``；逐句 ``enrolled_speaker`` / ``enrolled_similarity``（按句时间窗切 PCM
再提嵌入）。与 ``diar_speaker_id``（聚类 ID）独立，可同开。
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _l2n(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float64).ravel()
    n = float(np.linalg.norm(x)) + 1e-8
    return x / n


class SpeakerRegistry:
    """Maps display names to a single averaged L2-normalized embedding."""

    def __init__(self, path: str = ""):
        self.path = (path or "").strip()
        self._emb: dict[str, np.ndarray] = {}
        if self.path and os.path.isfile(self.path):
            self.load(self.path)

    def count(self) -> int:
        return len(self._emb)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._emb.clear()
        for k, vec in data.items():
            self._emb[str(k)] = _l2n(np.asarray(vec, dtype=np.float64))

    def save(self) -> None:
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {k: v.astype(float).tolist() for k, v in self._emb.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def register_vector(self, name: str, emb: np.ndarray) -> None:
        e = _l2n(emb)
        key = str(name)
        if key in self._emb:
            acc = self._emb[key] + e
            e = _l2n(acc)
        self._emb[key] = e

    def best_match(self, emb: np.ndarray, threshold: float) -> tuple[Optional[str], float]:
        if not self._emb:
            return None, 0.0
        e = _l2n(emb)
        best_n: Optional[str] = None
        best_s = -1.0
        for n, ref in self._emb.items():
            sim = float(np.dot(e, ref))
            if sim > best_s:
                best_s, best_n = sim, n
        if best_n is not None and best_s >= float(threshold):
            return best_n, best_s
        return None, best_s
