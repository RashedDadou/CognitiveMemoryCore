from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time
from collections import defaultdict

@dataclass
class MemoryChunk:
    content: str
    embedding: Optional[Any] = None
    priority: float = 1.0           # من 0.0 إلى 1.0
    timestamp: float = 0.0
    source: str = "current"
    layer: int = -1                 # سيتم تحديده تلقائياً

class LayeredWorkingMemory:
    """
    ذاكرة عمل مقسمة إلى طبقات مرقمة حسب الأولوية
    كل طبقة لها سعة محددة، وعند الامتلاء يتم دفع العنصر الأضعف للطبقة التالية
    """
    def __init__(self):
        # تعريف الطبقات (يمكن تعديلها بسهولة)
        self.layer_config = [
            {"layer": 0, "name": "عاجل / في التركيز",     "max_chunks": 3,  "min_priority": 0.90},
            {"layer": 1, "name": "مهم / نشط حالياً",       "max_chunks": 5,  "min_priority": 0.70},
            {"layer": 2, "name": "متوسط الأهمية",         "max_chunks": 6,  "min_priority": 0.45},
            {"layer": 3, "name": "احتياطي / خلفي",         "max_chunks": 8,  "min_priority": 0.20},
            {"layer": 4, "name": "أرشيف مؤقت",             "max_chunks": 12, "min_priority": 0.00},
        ]

        # تخزين الـ chunks حسب الطبقة
        self.layers: Dict[int, List[MemoryChunk]] = defaultdict(list)

        # لتسهيل البحث السريع عن chunk معين (اختياري)
        self.chunk_index: Dict[str, MemoryChunk] = {}

    def _get_layer_for_priority(self, priority: float) -> int:
        """إرجاع رقم الطبقة المناسبة حسب قيمة الأولوية"""
        for cfg in self.layer_config:
            if priority >= cfg["min_priority"]:
                return cfg["layer"]
        return self.layer_config[-1]["layer"]  # آخر طبقة

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
