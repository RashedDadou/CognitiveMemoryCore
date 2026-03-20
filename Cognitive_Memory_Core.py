# Cognitive_Memory_Core.py


import torch
import math
from torch import torch as torch.nn
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass
from bisect import bisect_left


# افتراضي (يمكن تمريره من الخارج أو من الكور)
# قاموس مساعد لشدة الأنماط (يمكن تعديله)
MODE_INTENSITY = {
    "intuition": 0.4,
    "brainstorm": 0.45,
    "visual-spatial": 0.65,
    "analytical": 0.85,
    "calculation": 0.92,
    "episodic": 0.78,
    "semantic": 0.88,
}


@dataclass
class MemoryChunk:
    content: str
    embedding: Optional[Any] = None
    priority: float = 1.0
    timestamp: float = 0.0
    source: str = "current"
    layer: int = -1

# ────────────────────────────────────────────────────────────────
#          نظام الإدارة المؤقته
# ────────────────────────────────────────────────────────────────
class LayeredWorkingMemory:
    """الذاكرة المؤقتة مقسمة طبقات – تابعة للمحول الرئيسي"""

    def __init__(self, core):
        self.core = core
        self.layer_config = [
            {"layer": 0, "name": "عاجل",       "max": 3,  "min_pri": 0.90},
            {"layer": 1, "name": "مهم",        "max": 5,  "min_pri": 0.70},
            {"layer": 2, "name": "متوسط",     "max": 6,  "min_pri": 0.45},
            {"layer": 3, "name": "احتياطي",   "max": 8,  "min_pri": 0.20},
            {"layer": 4, "name": "أرشيف",      "max": 12, "min_pri": 0.00},
        ]
        self.layers: Dict[int, List[MemoryChunk]] = {i: [] for i in range(5)}
        self.chunk_index: Dict[str, MemoryChunk] = {}

    def add_or_update(
        self,
        content: str,
        embedding=None,
        priority: float = 1.0,
        source: str = "current",
        update_if_exists: bool = True
    ):
        now = time.time()

        # إذا موجود سابقاً ونريد تحديثه
        if update_if_exists and content in self.chunk_index:
            chunk = self.chunk_index[content]
            old_layer = chunk.layer
            chunk.priority = priority
            chunk.timestamp = now
            chunk.source = source
            chunk.embedding = embedding or chunk.embedding

            # إزالته من الطبقة القديمة
            self.layers[old_layer] = [c for c in self.layers[old_layer] if c.content != content]

            # وضعه في الطبقة الجديدة المناسبة
            new_layer = self._get_layer_for_priority(priority)
            chunk.layer = new_layer
            self.layers[new_layer].append(chunk)
            self._enforce_layer_capacity(new_layer)
            return

        # إضافة جديد
        new_layer = self._get_layer_for_priority(priority)
        chunk = MemoryChunk(
            content=content,
            embedding=embedding,
            priority=priority,
            timestamp=now,
            source=source,
            layer=new_layer
        )

        self.layers[new_layer].append(chunk)
        self.chunk_index[content] = chunk

        # ضمان عدم تجاوز السعة
        self._enforce_layer_capacity(new_layer)

    def _get_layer_for_priority(self, priority: float) -> int:
        """إرجاع رقم الطبقة المناسبة حسب قيمة الأولوية"""
        for cfg in self.layer_config:
            if priority >= cfg["min_priority"]:
                return cfg["layer"]
        return self.layer_config[-1]["layer"]  # آخر طبقة

    def get_layer_counts(self) -> Dict[int, int]:
        return {ly: len(chunks) for ly, chunks in self.layers.items()}

    def get_layer_ages(self) -> Dict[int, float]:
        now = time.time()
        ages = {}
        for ly, chunks in self.layers.items():
            if chunks:
                ages[ly] = sum((now - c.timestamp)/3600 for c in chunks) / len(chunks)
        return ages

    def _enforce_layer_capacity(self, layer: int):
        """إذا امتلأت الطبقة → دفع أضعف عنصر إلى الطبقة التالية أو حذف"""
        current_layer = self.layers[layer]
        cfg = next(c for c in self.layer_config if c["layer"] == layer)

        if len(current_layer) <= cfg["max_chunks"]:
            return

        # ترتيب تنازلي حسب الأولوية ثم الزمن (الأقدم أولاً)
        current_layer.sort(key=lambda c: (c.priority, -c.timestamp), reverse=True)

        # العنصر الأضعف (آخر واحد بعد الترتيب)
        to_move = current_layer.pop()

        next_layer = layer + 1
        if next_layer < len(self.layer_config):
            # دفع للطبقة التالية
            to_move.layer = next_layer
            self.layers[next_layer].append(to_move)
            # تكرار التحقق في الطبقة التالية (قد يتسلسل)
            self._enforce_layer_capacity(next_layer)
        else:
            # آخر طبقة → حذف
            if to_move.content in self.chunk_index:
                del self.chunk_index[to_move.content]

    def get_all_grouped(self) -> Dict[int, List[Dict]]:
        """إرجاع جميع الـ chunks مقسمة حسب الطبقة"""
        result = {}
        for layer_num, chunks in self.layers.items():
            result[layer_num] = [
                {
                    "content": c.content,
                    "priority": c.priority,
                    "layer": c.layer,
                    "age_hours": (time.time() - c.timestamp) / 3600,
                    "source": c.source
                }
                for c in sorted(chunks, key=lambda x: x.priority, reverse=True)
            ]
        return result

    def build_context(self, layer_order: List[int], max_chars: int = 6000) -> str:
        parts = []
        for ly in layer_order:
            for chunk in sorted(self.layers.get(ly, []), key=lambda c: c.priority, reverse=True):
                parts.append(f"[L{ly} pri:{chunk.priority:.2f}] {chunk.content}")
        text = "\n".join(parts)
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    def get_highest_priority_context(self, max_chars: int = 6000) -> str:
        """نص مختصر يركز على الطبقات العليا أولاً"""
        parts = []
        for layer_num in sorted(self.layers.keys()):
            for chunk in sorted(self.layers[layer_num], key=lambda c: c.priority, reverse=True):
                prefix = f"[L{layer_num} pri:{chunk.priority:.2f}] "
                parts.append(prefix + chunk.content)
        full_text = "\n".join(parts)
        return full_text[:max_chars] + ("..." if len(full_text) > max_chars else "")

    def compress_all(self, keep_layers_up_to: int = 2):
        """ضغط كامل: الاحتفاظ فقط بالطبقات حتى رقم معين"""
        for layer_num in list(self.layers.keys()):
            if layer_num > keep_layers_up_to:
                for chunk in self.layers[layer_num]:
                    if chunk.content in self.chunk_index:
                        del self.chunk_index[chunk.content]
                del self.layers[layer_num]

# ────────────────────────────────────────────────────────────────
#               نظام إستعادة البيانات
# ────────────────────────────────────────────────────────────────
class AdvancedMemoryRouter(MemoryRouter):
    """
    نسخة متقدمة تعتمد على embeddings + uncertainty + mode confidence
    تحتاج: sentence-transformers أو torch + نموذج صغير
    """

    def __init__(self, core, embedding_model=None):
        super().__init__(core)
        self.embedding_model = embedding_model  # sentence-transformers أو أي نموذج

    def decide(
        current_thought_emb: torch.Tensor | None,
        context_summary_emb: torch.Tensor | None,
        mode: str = "intuition",
        uncertainty: float = 0.0,           # 0 = واثق، 1 = مش واثق
        user_emphasis: float = 0.0,
        fallback_interest: float = 0.50,
    ) -> Dict:
        if current_thought_emb is None or context_summary_emb is None:
            # مسار احتياطي بسيط جدًا (بدون embeddings)
            interest_score = fallback_interest
            used_semantic = False
        else:
            # المسار الذكي
            semantic_match = torch.nn.functional.cosine_similarity(
                current_thought_emb.unsqueeze(0),
                context_summary_emb.unsqueeze(0)
            ).item()

            mode_intensity = MODE_WEIGHTS.get(mode, 0.50)

            # هنا نصحح الاتجاه: uncertainty عالي → interest عالي (نحتاج استرجاع أكتر)
            interest_raw = (
                0.42 * semantic_match +
                0.26 * uncertainty +                # ← التصحيح المهم
                0.14 * mode_intensity +
                0.10 * (0.8 if uncertainty > 0.6 else 1.0) +  # recency بديل مؤقت
                0.08 * user_emphasis
            )
            interest_score = torch.sigmoid(torch.tensor(interest_raw * 4.5 - 2.2)).item()
            used_semantic = True

        # تدرج العمق (نفس المنطق لكن نضبط الحدود شوية)
        if interest_score < 0.28:
            depth = "surface"
            max_items = 15
            extra = 0
            strat = "vector_fast"
        elif interest_score < 0.62:
            depth = "mid"
            max_items = 140
            extra = 1
            strat = "hybrid_rerank"
        elif interest_score < 0.89:
            depth = "deep"
            max_items = 750
            extra = 3
            strat = "graph_guided"
        else:
            depth = "exhaustive"
            max_items = 4000
            extra = 5
            strat = "recursive_full"

        return {
            "interest_score": round(interest_score, 3),
            "depth_level": depth,
            "max_initial_items": max_items,
            "extra_traversal_layers": extra,
            "retrieval_strategy": strat,
            "context_weight": round(max(0.12, 1.0 - interest_score * 0.92), 3),
            "retrieval_weight": round(min(0.96, interest_score * 1.12), 3),
            "allow_recursive_followup": interest_score > 0.84,
            "used_semantic_match": used_semantic,
        }

# ────────────────────────────────────────────────────────────────
#               النواة الرئيسية – الواجهة الوحيدة
# ────────────────────────────────────────────────────────────────
class CognitiveMemoryCore:
    """
    المحول الرئيسي الوحيد الذي يتعامل معه الكود الخارجي.
    يحتوي داخليًا على الثلاث وحدات ويدير التواصل بينها.
    """

    def __init__(self):
        self.core = core
        self.last_major_event = time.time()
        self.current_mode = "intuition"
        self.global_interest = 0.5
        self.topic_timestamps: Dict[str, list[float]] = {}  # topic → [ts1, ts2, ...]

        self.working_memory = LayeredWorkingMemory(self)
        self.layer_predictor = CognitiveLayerPredictor(self)
        self.retrieval_router = AdvancedMemoryRouter(self)

    # ──── واجهات عامة (اللي هيستخدمها الكود الخارجي) ───────────────────────

    def add(self, content: str, priority: float = 1.0, **kwargs):
        self.working_memory.add_or_update(content, priority, **kwargs)
        self._on_content_change()  # يمكن يحدث الـ predictor أو يعيد حساب

    def get_context_for_prompt(self, mode: str, interest_score: float) -> str:
        """
        الدالة الأساسية التي يستدعيها الـ LLM أو الـ agent
        ترجع نص السياق الحالي (من الطبقات المرتبة ديناميكيًا)
        """
        # 1. التنبؤ بترتيب الطبقات الحالي
        predicted_order = self.predictor.get_predicted_layer_order(
            mode=mode,
            interest_score=interest_score,
            current_layer_state=self.memory.get_layer_counts(),
            layer_ages=self.memory.get_layer_ages(),
            user_emphasis=0.0  # يمكن تمريرها كباراميتر لاحقًا
        )

        # 2. جمع النصوص من الطبقات حسب الترتيب المتوقع
        parts = []
        for layer_id in predicted_order:
            chunks = self.memory.layers.get(layer_id, [])
            for chunk in sorted(chunks, key=lambda c: c.priority, reverse=True):
                parts.append(f"[L{layer_id} pri:{chunk.priority:.2f}] {chunk.content}")

        context_text = "\n".join(parts)

        # 3. قرار الـ router: هل نكتفي بالسياق الحالي أم نسترجع أكثر؟
        router_decision = self.router.decide(
            mode=mode,
            interest_score=interest_score,
            wm_chunk_count=self.memory.total_chunks(),
            wm_avg_priority=self.memory.average_priority()
        )

        # إرجاع السياق + توصية الاسترجاع (إذا كان مطلوبًا)
        return {
            "context_text": context_text,
            "retrieval_needed": router_decision["retrieval_weight"] > 0.35,
            "recommended_depth": router_decision["depth_level"],
            "predicted_layers_used": predicted_order,
            "context_weight": router_decision["context_weight"]
        }

    def get_context(
        self,
        interest_override: Optional[float] = None,
        max_chars: Optional[int] = None,
        current_thought: Optional[str] = None,          # لاحقاً لـ embeddings أو emphasis
        user_emphasis: Optional[float] = None,          # يمكن تمريره من الخارج
        include_immediate: bool = True,                 # هل نضيف الـ short-term buffer؟
    ) -> Dict[str, Any]:
        """
        الواجهة الرئيسية لبناء السياق + قرار الاسترجاع
        Returns richer dict مع معلومات إضافية للـ monitoring والتطوير
        """
        now = time.time()

        # ─── 0. حساب الاهتمام الفعلي ───────────────────────────────────────
        interest = (
            interest_override
            if interest_override is not None
            else self.global_interest
        )

        # ─── 1. حساب user_emphasis إذا لم يُمرر ──────────────────────────────
        if user_emphasis is None:
            user_emphasis = self._estimate_user_emphasis(current_thought or "")
            # أو يمكن أن تأخذ قيمة افتراضية من الـ mode أو من الـ state

        # ─── 2. جمع البيانات اللازمة للتنبؤ بالطبقات ───────────────────────
        layer_counts = self.working_memory.get_layer_counts()
        layer_ages   = self.working_memory.get_layer_ages()
        global_age_h = (now - self.last_major_event) / 3600.0

        # ─── 3. التنبؤ بترتيب الطبقات (النسخة المحسنة) ──────────────────────
        layer_order = self.layer_predictor.predict_layer_order(
            current_mode     = self.current_mode,
            interest_score   = interest,
            current_layer_state = layer_counts,
            layer_ages       = layer_ages,
            user_emphasis    = user_emphasis,
            global_age_hours = global_age_h
        )

        # ─── 4. بناء نص السياق ───────────────────────────────────────────────
        # دعم تخصيص الطول حسب الـ mode أو الـ interest
        if max_chars is None:
            # مثال على تخصيص ديناميكي
            if interest > 0.75 or self.current_mode in ("analytical", "calculation"):
                max_chars = 11000
            else:
                max_chars = 6500

        context_text = self.working_memory.build_context(
            layer_order=layer_order,
            max_chars=max_chars
        )

        # ─── 5. (اختياري) إضافة immediate / short-term buffer ────────────────
        immediate_text = ""
        if include_immediate and hasattr(self, 'short_term_buffer') and self.short_term_buffer:
            immediate_text = self._build_immediate_context(max_chars=1200)

        if immediate_text:
            context_text = immediate_text + "\n\n" + context_text

        # ─── 6. قرار الاسترجاع ────────────────────────────────────────────────
        retrieval = self.retrieval_router.decide(
            mode              = self.current_mode,
            interest          = interest,
            uncertainty       = self._estimate_uncertainty() if hasattr(self, '_estimate_uncertainty') else 0.3,
            user_emphasis     = user_emphasis
        )

        # ─── 7. النتيجة النهائية (معلومات إضافية للـ observability) ─────────
        return {
            "context_text": context_text,
            "retrieval_needed": retrieval["retrieval_weight"] > 0.35,
            "retrieval_depth": retrieval["depth_level"],
            "used_layer_order": layer_order,
            "context_weight": retrieval["context_weight"],
            "retrieval_weight": retrieval["retrieval_weight"],

            # معلومات إضافية مفيدة
            "interest_used": round(interest, 3),
            "user_emphasis_used": round(user_emphasis, 2),
            "global_session_age_hours": round(global_age_h, 1),
            "layer_counts": layer_counts,
            "layer_ages_avg": {k: round(v, 1) for k,v in layer_ages.items()},
            "immediate_included": bool(immediate_text),
            "context_length_chars": len(context_text),
            "timestamp": now,
        }

    # ─── دوال مساعدة صغيرة (يمكن تطويرها لاحقاً) ────────────────────────────

    def _estimate_user_emphasis(self, text: str) -> float:
        """تقدير بسيط لتركيز المستخدم من النص الأخير"""
        if not text:
            return 0.0

        emphasis_keywords = ["مهم", "ضروري", "ركز", "افهم جيدًا", "بالتفصيل", "لا تنسى", "أكيد"]
        count = sum(1 for kw in emphasis_keywords if kw in text.lower())
        return min(1.0, count * 0.25 + 0.1 * (len(text) > 200))

    def _build_immediate_context(self, max_chars: int = 1200) -> str:
        """بناء سياق فوري من الـ short-term buffer (إذا وُجد)"""
        parts = []
        for chunk in sorted(self.short_term_buffer, key=lambda c: c.priority, reverse=True):
            parts.append(f"[immediate pri:{chunk.priority:.1f}] {chunk.content}")
        text = "\n".join(parts)
        return text[:max_chars] + "..." if len(text) > max_chars else text

    def _estimate_uncertainty(self) -> float:
        return 0.3  # placeholder – يمكن تطويره

    def set_mode(self, mode: str):
        self.current_mode = mode
        self._on_mode_change()

    def reset(self, mode: str = "minor"):
        """
        إعادة تهيئة الذاكرة بمستويات مختلفة

        Modes:
        - 'full':     حذف كامل لكل الطبقات والفهرس
        - 'major':    حذف الطبقات المنخفضة + تقليل أولوية العليا
        - 'minor':    تقليل خفيف للأولويات + تنظيف السعة
        - 'idle':     reset تلقائي خفيف بعد فترة خمول (افتراضيًا بعد ساعتين)
        """
        now = time.time()

        # وضع idle → تحقق من الخمول ثم نفذ minor
        if mode == "idle":
            if hasattr(self, 'last_minor_reset') and (now - self.last_minor_reset) > 7200:  # 2 ساعات
                return self.reset("minor")
            else:
                return  # لا حاجة لعمل شيء

        # وضع full
        if mode == "full":
            self.layers.clear() if hasattr(self, 'layers') else self.working_memory.layers.clear()
            self.chunk_index.clear() if hasattr(self, 'chunk_index') else self.working_memory.chunk_index.clear()

            # تحديث الطوابع الزمنية (يدعم النسختين)
            if hasattr(self, 'last_major_reset'):
                self.last_major_reset = now
                self.last_minor_reset = now
            if hasattr(self, 'last_major_event'):
                self.last_major_event = now
            return

        # وضع major
        if mode == "major":
            target_layers = self.layers if hasattr(self, 'layers') else self.working_memory.layers
            target_index  = self.chunk_index if hasattr(self, 'chunk_index') else self.working_memory.chunk_index

            for ly in [3, 4]:
                for c in target_layers.pop(ly, []):
                    target_index.pop(c.content, None)

            for ly in [0, 1, 2]:
                for c in target_layers.get(ly, []):
                    c.priority *= 0.60  # أو 0.55 حسب الرغبة

            if hasattr(self, 'last_major_reset'):
                self.last_major_reset = now
            if hasattr(self, 'last_major_event'):
                self.last_major_event = now
            return

        # وضع minor (الافتراضي)
        if mode == "minor":
            target_layers = self.layers if hasattr(self, 'layers') else self.working_memory.layers

            for chunks in target_layers.values():
                for c in chunks:
                    c.priority *= 0.85  # أو 0.80–0.90 حسب الاختبار

            # تنظيف السعة إذا كانت الدالة موجودة
            if hasattr(self, '_enforce_capacity_all'):
                self._enforce_capacity_all()
            elif hasattr(self, 'working_memory') and hasattr(self.working_memory, '_enforce_capacity'):
                for ly in list(target_layers.keys()):
                    self.working_memory._enforce_capacity(ly)

            # تحديث الطابع الزمني للـ minor
            if hasattr(self, 'last_minor_reset'):
                self.last_minor_reset = now
            # لو كان موجود last_major_event فقط، نحدثه أيضًا
            if hasattr(self, 'last_major_event'):
                self.last_major_event = now

    def _clear_low_layers(self):
        target = self.layers if hasattr(self, 'layers') else self.working_memory.layers
        idx    = self.chunk_index if hasattr(self, 'chunk_index') else self.working_memory.chunk_index
        for ly in [3,4]:
            for c in target.pop(ly, []):
                idx.pop(c.content, None)

    def _decay_priorities(self, factor: float = 0.85):
        target = self.layers if hasattr(self, 'layers') else self.working_memory.layers
        for chunks in target.values():
            for c in chunks:
                c.priority *= factor

    def _enforce_capacity_all(self):
        for ly in sorted(self.layers.keys()):
            self._enforce_capacity(ly)

    # ──── دوال داخلية يستدعيها الوحدات ──────────────────────────────────────

    def _on_content_change(self):
        # مثال: تحديث interest أو إشارة لإعادة حساب
        self.global_interest = min(1.0, self.global_interest + 0.08)

    def _on_mode_change(self):
        # ممكن نعدل الـ decay أو نعمل reset جزئي
        pass

    def compute_topic_recency_score(
        topic: str,
        topic_history_timestamps: list[float],      # كل ظهور سابق للموضوع (مرتب تصاعديًا)
        now: float = None,
        params: dict = None
    ) -> float:
        if now is None:
            now = time.time()

        if params is None:
            params = {
                'lambda_base': 0.075,           # decay أساسي
                'alpha_ewma': 0.12,             # smoothing factor لـ EWMA
                'max_age_hours_ewma': 120.0,    # نافذة EWMA فعالة (~5 أيام)
                'freq_log_base': 1.7,
                'freq_scale': 0.28,
                'min_recency': 0.04,
                'max_recency': 1.85,
                'core_decay_reduction': 0.50    # نسبة التخفيض للـ core topics (50% → decay أبطأ)
            }

        # ─── 1. الـ decay الأساسي (exponential) ────────────────────────────────
        age_hours = max(0.0, (now - (topic_history_timestamps[-1] if topic_history_timestamps else now)) / 3600)

        # تعديل lambda حسب أهمية الموضوع (الطريقة 1)
        if isinstance(params.get('core_topics'), set) and topic in params['core_topics']:
            effective_lambda = params['base_lambda'] * params['core_decay_factor']
        else:
            # أو استخدام قاموس إذا أردت دقة أكبر
            factor = params.get('core_decay_factors', {}).get(topic, 1.0)
            effective_lambda = params['base_lambda'] * factor

        base_decay = math.exp(-effective_lambda * age_hours)

        # ─── 2. قوة التكرار الحديث (EWMA frequency) ─────────────────────────────
        freq_strength = 0.0
        if topic_history_timestamps:
            freq_strength = compute_ewma_frequency(
                topic_history_timestamps,
                now=now,
                alpha=params['freq_alpha'],
                max_age_hours=params['freq_max_age_hours']
            )

        # تحويل التكرار إلى multiplier (log لتجنب الانفجار)
        freq_boost = params['freq_scale'] * math.log(params['freq_log_base'] + freq_strength)

        # ─── 3. النتيجة النهائية ────────────────────────────────────────────────
        score = base_decay * (1.0 + freq_boost)

        # clipping + floor لتجنب القيم السالبة أو الصغيرة جدًا
        score = max(params['min_score'], min(params['max_score'], score))

        return score

    def compute_ewma_frequency(
        timestamps: list[float],
        now: float,
        alpha: float = 0.10,
        max_age_hours: float = 168.0
    ) -> float:
        if not timestamps:
            return 0.0

        ewma = 0.0
        total_weight = 0.0

        for ts in reversed(timestamps):
            age_h = (now - ts) / 3600.0
            if age_h > max_age_hours:
                break
            w = math.exp(-alpha * age_h)
            ewma += w          # يمكن استخدام ewma += 1 * w إذا أردت العدد المرجح
            total_weight += w

        return ewma / total_weight if total_weight > 0 else 0.0

    # ──── Recency & Frequency helpers ────────────────────────────────────────
    def compute_recency_factor(
        self,
        age_hours: float,
        frequency_recent: int,
        params: dict = None
    ) -> float:
        if params is None:
            params = {
                'lambda_decay': 0.075,
                'freq_scale': 0.28,
                'freq_log_base': 1.7,
                'min_recency': 0.04,
                'max_recency': 1.85
            }

        decay = math.exp(-params['lambda_decay'] * max(0.0, age_hours))
        freq_boost = params['freq_scale'] * math.log(params['freq_log_base'] + max(0, frequency_recent))
        combined = decay * (1.0 + freq_boost)
        return max(params['min_recency'], min(params['max_recency'], combined))

    def get_frequency_recent_bisect(
        self,
        sorted_timestamps: list[float],
        now: float = None,
        window_hours: float = 48.0
    ) -> int:
        if now is None:
            now = time.time()
        cutoff = now - window_hours * 3600
        idx = bisect_left(sorted_timestamps, cutoff)
        return len(sorted_timestamps) - idx

    # مثال استخدام داخلي (يمكن استدعاؤها عند add أو reset أو priority update)
    def _boost_priority_by_recency(self, chunk: MemoryChunk, topic_timestamps: list[float]):
        now = time.time()
        age_h = (now - chunk.timestamp) / 3600
        freq = self.get_frequency_recent_bisect(sorted(topic_timestamps), now)
        recency_score = self.compute_recency_factor(age_h, freq)
        chunk.priority *= recency_score  # أو += بعض القيمة، حسب السياسة

class CognitiveLayerPredictor:
    """
    شبكة ربط ديناميكية بين جهد الفكري والطبقات
    تتنبأ بترتيب الطبقات قبل الطلب (Predictive Layer Ordering)
    """
    def __init__(self):
        self.core = core
        self.last_major_update = time.time()  # وقت آخر تغيير كبير (إضافة/حذف/أولوية)
        # خريطة أساسية: كل mode يفضل ترتيب معين للطبقات
        self.mode_base_order = {
            "intuition":      [0, 1, 2],      # يركز على العاجل + المهم
            "brainstorm":     [0, 1, 3],      # يسمح بالاحتياطي للإلهام
            "visual-spatial": [1, 0, 2],      # الطبقة 1 غالبًا أكثر فائدة في التخيل
            "analytical":     [0, 1, 2, 3],   # يبدأ بالأعلى ثم ينزل تدريجيًا
            "calculation":    [0, 1],         # يركز جدًا على العاجل والمهم فقط
            "episodic":       [2, 1, 3],      # يحتاج الطبقات المتوسطة والاحتياطي (الذاكرة السابقة)
            "semantic":       [1, 2, 3],      # يحتاج معرفة عامة (متوسطة وخلفية)
        }

        # معاملات تعديل حسب الـ interest_score
        self.interest_weights = {
            "high":   [0.0, 0.1, 0.3, 0.6, 1.0],   # interest > 0.8  → يسمح بعمق أكبر
            "mid":    [0.4, 0.6, 0.8, 0.9, 1.0],
            "low":    [0.8, 0.9, 1.0, 1.0, 1.0]    # interest منخفض → يركز على الطبقات العليا فقط
        }

    def predict_layer_order(
        self,
        current_mode: str,
        interest_score: float,                  # 0.0 → 1.0
        current_layer_state: Dict[int, int],    # {layer: count}
        layer_ages: Dict[int, float] = None,    # {layer: avg_age_hours} – اختياري
        user_emphasis: float = 0.0,
        global_age_hours: float = 0.0           # عمر الجلسة/السياق الكلي – اختياري
    ) -> List[int]:
        """
        ترتيب الطبقات المقترح (من الأعلى أولوية إلى الأقل)
        يرجع قائمة مثل [0, 1, 3, 2, 4]
        """

        # ─── الإعدادات الأساسية ─────────────────────────────────────────────
        base_order = self.mode_base_order.get(current_mode, [0, 1, 2, 3, 4])

        # تحديد مستوى الاهتمام
        if interest_score > 0.82:
            interest_key = "high"
        elif interest_score > 0.45:
            interest_key = "mid"
        else:
            interest_key = "low"

        weights = self.interest_weights[interest_key]

        # إذا لم يُمرر layer_ages، نستخدم قاموس فارغ (لا عقوبة عمر)
        layer_ages = layer_ages or {}

        layer_scores: List[Tuple[int, float]] = []

        for layer in range(5):
            # 1. الدرجة الأساسية من الـ mode
            base_rank = base_order.index(layer) if layer in base_order else 999
            base_score = 1.0 - (base_rank * 0.15)  # كلما أقرب للبداية → أعلى

            # 2. تعديل الاهتمام
            interest_modifier = weights[layer] if layer < len(weights) else 1.0

            # 3. مكافأة الاحتلال (لو الطبقة مليانة نسبيًا → قيمتها ترتفع)
            occupancy = current_layer_state.get(layer, 0) / 10.0
            occupancy_bonus = min(0.25, occupancy * 0.4)

            # 4. مكافأة التركيز من المستخدم (خاصة الطبقات العاجلة)
            emphasis_bonus = user_emphasis * 0.32 if layer <= 2 else user_emphasis * 0.10

            # 5. عقوبة التقادم (الجزء الأهم الجديد)
            age_penalty = 0.0
            if layer in layer_ages and layer_ages[layer] > 0.1:
                age_h = layer_ages[layer]
                # exponential decay ناعم
                age_penalty = 0.24 * (1 - math.exp(-0.42 * age_h))

            # عقوبة إضافية لو السياق كله قديم
            global_penalty = 0.09 * max(0.0, (global_age_hours - 1.0) / 5.0)

            # ─── الحساب النهائي ───────────────────────────────────────────────
            temporal_factor = 1.0 - (age_penalty + global_penalty)
            temporal_factor = max(0.35, temporal_factor)  # حد أدنى حتى لا تنهار الدرجة

            final_score = (
                base_score
                * interest_modifier
                * temporal_factor
                + occupancy_bonus
                + emphasis_bonus
            )

            final_score = max(0.05, final_score)  # لا نسمح بقيم سالبة أو ضعيفة جدًا

            layer_scores.append((layer, final_score))

        # ─── الترتيب النهائي ─────────────────────────────────────────────────
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_order = [layer for layer, _ in layer_scores]

        return predicted_order

    def get_recommended_layers(
        self,
        current_mode: str,
        interest_score: float,
        current_layer_state: Dict[int, int],
        max_layers_to_fetch: int = 3,
        user_emphasis: float = 0.0
    ) -> List[int]:
        """الدالة النهائية المباشرة للاستخدام"""
        order = self.predict_layer_order(
            current_mode, interest_score, current_layer_state, user_emphasis
        )
        return order[:max_layers_to_fetch]
