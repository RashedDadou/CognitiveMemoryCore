from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from collections import deque

@dataclass
class MemoryChunk:
    content: str                        # النص أو الوصف
    embedding: Optional[Any] = None     # vector إذا موجود
    priority: float = 1.0               # أولوية داخل الـ buffer (1.0 = عالية)
    timestamp: float = 0.0              # وقت الإضافة / آخر تعديل
    source: str = "current"             # من أين جاء (user, thought, retrieval, ...)

class WorkingMemoryBuffer:
    """
    ذاكرة عمل مؤقتة محدودة الحجم
    تدير الـ chunks بالأولوية + الزمن + الضغط التلقائي
    """
    def __init__(self, max_chunks: int = 7, max_age_hours: float = 4.0):
        self.max_chunks = max_chunks
        self.max_age_hours = max_age_hours
        self.buffer: deque[MemoryChunk] = deque(maxlen=max_chunks)
        self.current_time = time.time

    def add(self, content: str, embedding=None, priority: float = 1.0, source: str = "current"):
        chunk = MemoryChunk(
            content=content,
            embedding=embedding,
            priority=priority,
            timestamp=self.current_time(),
            source=source
        )
        self.buffer.append(chunk)
        self._cleanup_old()

    def _cleanup_old(self):
        """حذف الـ chunks القديمة جدًا حتى لو لم يمتلئ الحد"""
        now = self.current_time()
        cutoff = now - self.max_age_hours * 3600
        while self.buffer and self.buffer[0].timestamp < cutoff:
            self.buffer.popleft()

    def compress(self, keep_top_n: int = 4):
        """ضغط الـ buffer: احتفظ بأعلى الأولويات فقط"""
        if len(self.buffer) <= keep_top_n:
            return
        sorted_chunks = sorted(self.buffer, key=lambda c: c.priority, reverse=True)
        self.buffer = deque(sorted_chunks[:keep_top_n], maxlen=self.max_chunks)

    def get_all(self) -> List[Dict]:
        return [
            {
                "content": c.content,
                "priority": c.priority,
                "age_hours": (self.current_time() - c.timestamp) / 3600,
                "source": c.source
            }
            for c in self.buffer
        ]

    def get_context_string(self, max_length_chars: int = 8000) -> str:
        """تحويل الـ buffer إلى نص واحد متماسك لإدخاله في LLM"""
        parts = []
        for chunk in sorted(self.buffer, key=lambda c: c.priority, reverse=True):
            parts.append(f"[pri:{chunk.priority:.1f}] {chunk.content}")
        text = "\n".join(parts)
        return text[:max_length_chars] + "..." if len(text) > max_length_chars else text
