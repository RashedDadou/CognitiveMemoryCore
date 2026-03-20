from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import math

@dataclass
class MemoryChunk:
    content: str
    embedding: Optional[Any] = None
    priority: float = 1.0
    timestamp: float = 0.0
    source: str = "current"
    layer: int = -1

class CognitiveMemoryOrchestrator:
    """
    النواة الرئيسية التي تدمج:
    - LayeredWorkingMemory
    - CognitiveLayerPredictor
    - منطق memory_router
    تدير الطبقات + التنبؤ + قرارات الاسترجاع + reset متعدد المستويات
    """
    def __init__(self):
        # ---------------- Layered Working Memory ----------------
        self.layer_config = [
            {"layer": 0, "name": "عاجل",       "max": 3,  "min_pri": 0.90},
            {"layer": 1, "name": "مهم",        "max": 5,  "min_pri": 0.70},
            {"layer": 2, "name": "متوسط",     "max": 6,  "min_pri": 0.45},
            {"layer": 3, "name": "احتياطي",   "max": 8,  "min_pri": 0.20},
            {"layer": 4, "name": "أرشيف",      "max": 12, "min_pri": 0.00},
        ]
        self.layers: Dict[int, List[MemoryChunk]] = {i: [] for i in range(5)}
        self.chunk_index: Dict[str, MemoryChunk] = {}

        # ---------------- Cognitive Layer Predictor ----------------
        self.predictor = self._init_predictor()

        # ---------------- Reset states ----------------
        self.last_major_reset = time.time()
        self.last_minor_reset = time.time()

    def _init_predictor(self):
        class Predictor:
            def __init__(self):
                self.mode_base_order = {
                    "intuition": [0,1,2], "brainstorm": [0,1,3], "analytical": [0,1,2,3],
                    # ... باقي المودات كما في السابق
                }
                self.interest_weights = {
                    "high": [0.0,0.1,0.3,0.6,1.0],
                    "mid": [0.4,0.6,0.8,0.9,1.0],
                    "low": [0.8,0.9,1.0,1.0,1.0]
                }

            def predict(self, mode, interest, layer_state, layer_ages, user_emphasis=0.0):
                base = self.mode_base_order.get(mode, [0,1,2,3])
                key = "high" if interest > 0.82 else "mid" if interest > 0.45 else "low"
                w = self.interest_weights[key]

                scores = []
                for ly in range(5):
                    b_rank = base.index(ly) if ly in base else 999
                    b_score = 1.0 - b_rank * 0.15
                    i_mod = w[ly] if ly < len(w) else 1.0
                    occ = layer_state.get(ly, 0) / 10.0
                    occ_b = min(0.25, occ * 0.4)
                    age_p = 0.22 * (1 - math.exp(-0.45 * layer_ages.get(ly, 0)))
                    glob_p = 0.08 * ((time.time() - self.last_major_reset)/3600 / 4)
                    score = b_score * i_mod * (1 - age_p - glob_p) + occ_b + user_emphasis*0.3
                    scores.append((ly, max(0.05, score)))

                scores.sort(key=lambda x: x[1], reverse=True)
                return [ly for ly, _ in scores]

        return Predictor()

    # ──── إضافة / تحديث chunk ────────────────────────────────────────────────
    def add_or_update(self, content: str, priority: float = 1.0, **kwargs):
        now = time.time()
        layer = self._get_layer_for_priority(priority)

        if content in self.chunk_index:
            old = self.chunk_index[content]
            old_layer = old.layer
            self.layers[old_layer] = [c for c in self.layers[old_layer] if c.content != content]

        chunk = MemoryChunk(content=content, priority=priority, timestamp=now, layer=layer, **kwargs)
        self.layers[layer].append(chunk)
        self.chunk_index[content] = chunk

        self._enforce_capacity(layer)
        self.last_minor_reset = now  # أي تعديل = minor reset

    def _get_layer_for_priority(self, pri: float) -> int:
        for cfg in self.layer_config:
            if pri >= cfg["min_pri"]:
                return cfg["layer"]
        return 4

    def _enforce_capacity(self, layer: int):
        cfg = next(c for c in self.layer_config if c["layer"] == layer)
        chunks = self.layers[layer]
        if len(chunks) > cfg["max"]:
            chunks.sort(key=lambda c: (c.priority, -c.timestamp), reverse=True)
            to_move = chunks.pop()
            next_l = layer + 1
            if next_l < 5:
                to_move.layer = next_l
                self.layers[next_l].append(to_move)
                self._enforce_capacity(next_l)
            else:
                del self.chunk_index[to_move.content]

    # ──── التنبؤ بترتيب الطبقات ───────────────────────────────────────────────
    def get_predicted_layer_order(self, mode: str, interest_score: float, user_emphasis: float = 0.0) -> List[int]:
        layer_state = {ly: len(chunks) for ly, chunks in self.layers.items()}
        layer_ages = {}
        now = time.time()
        for ly, chunks in self.layers.items():
            if chunks:
                avg_age = sum((now - c.timestamp)/3600 for c in chunks) / len(chunks)
                layer_ages[ly] = avg_age

        global_age = (now - self.last_major_reset) / 3600

        return self.predictor.predict(mode, interest_score, layer_state, layer_ages, user_emphasis)

    # ──── منطق memory_router مدمج ──────────────────────────────────────────────
    def decide_retrieval_strategy(
        self,
        current_thought_emb,
        context_emb,
        mode: str,
        uncertainty: float,
        user_emphasis: float = 0.0,
        cosine_sim = lambda a,b: 0.7
    ) -> Dict:
        semantic = cosine_sim(current_thought_emb, context_emb)
        wm_count = sum(len(chunks) for chunks in self.layers.values())
        wm_pri_avg = sum(c.priority for ly in self.layers.values() for c in ly) / max(1, wm_count)
        wm_fresh = 1 - min(1, max((time.time()-c.timestamp)/3600 for ly in self.layers.values() for c in ly)/4)

        interest_raw = (
            0.32 * semantic +
            0.22 * (1 - uncertainty)*-1 +
            0.14 * (0.4 if mode in ["intuition","brainstorm"] else 0.85) +
            0.12 * wm_pri_avg +
            0.10 * wm_fresh +
            0.06 * user_emphasis
        )
        interest = 1 / (1 + math.exp(-(interest_raw * 5.5 - 2.4)))

        # قرار العمق (مختصر)
        if interest < 0.25: depth, max_i, strat = "surface", 12, "vector_fast"
        elif interest < 0.58: depth, max_i, strat = "mid", 100, "hybrid_rerank"
        elif interest < 0.84: depth, max_i, strat = "deep", 600, "graph_guided"
        else: depth, max_i, strat = "exhaustive", 3000, "recursive_full"

        return {
            "interest_score": round(interest, 3),
            "predicted_layers": self.get_predicted_layer_order(mode, interest, user_emphasis),
            "depth_level": depth,
            "max_items": max_i,
            "strategy": strat,
            "context_weight": round(max(0.15, 1.0 - interest * 0.88), 3),
            "retrieval_weight": round(min(0.95, interest * 1.10), 3),
        }

    # ──── أنواع الـ reset المختلفة ──────────────────────────────────────────────
    def reset(self, mode: str = "soft"):
        now = time.time()
        if mode == "full":
            # حذف كل شيء
            self.layers.clear()
            self.chunk_index.clear()
            self.last_major_reset = now
            self.last_minor_reset = now
        elif mode == "major":
            # حذف الطبقات 3 و 4 + تصفير الأولويات في 0-2
            for ly in [3,4]:
                for c in self.layers.get(ly, []):
                    del self.chunk_index[c.content]
                del self.layers[ly]
            for ly in [0,1,2]:
                for c in self.layers.get(ly, []):
                    c.priority *= 0.6
            self.last_major_reset = now
        elif mode == "minor":
            # خفض أولوية الكل بنسبة خفيفة + حذف القديم جدًا
            for ly, chunks in self.layers.items():
                for c in chunks:
                    c.priority *= 0.85
            self._enforce_capacity_all()
            self.last_minor_reset = now
        elif mode == "idle":
            # reset خفيف جدًا بعد خمول طويل
            if (now - self.last_minor_reset) > 3600*2:
                self.reset("minor")

    def _enforce_capacity_all(self):
        for ly in sorted(self.layers.keys()):
            self._enforce_capacity(ly)
