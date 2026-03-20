class WorkingMemoryDispatcher:
    """
    المحول الثالث (الجامع)
    يربط بين:
    - memory_router (قرارات السياق vs الاسترجاع + العمق)
    - CognitiveLayerPredictor (ترتيب الطبقات ديناميكيًا)
    - LayeredWorkingMemory (الطبقات نفسها)

    هو الواجهة الوحيدة التي يتعامل معها باقي الكود
    """
    def __init__(self):
        self.memory = LayeredWorkingMemory()           # الطبقات الفعلية
        self.predictor = CognitiveLayerPredictor()     # التنبؤ بترتيب الطبقات
        self.router = MemoryRouter()                   # قرارات السياق vs الاسترجاع

    # ──── الوظائف الرئيسية التي يستدعيها الكود الخارجي ───────────────────────

    def add(self, content: str, priority: float = 1.0, **kwargs):
        """إضافة أو تحديث عنصر في الذاكرة المؤقتة"""
        self.memory.add_or_update(content, priority, **kwargs)

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

    def reset(self, mode: str = "minor"):
        """إعادة تهيئة حسب نوع الـ reset"""
        if mode == "full":
            self.memory.clear_all()
        elif mode == "major":
            self.memory.clear_lower_layers(keep_up_to_layer=2)
            self.memory.lower_priorities_all(0.6)
        elif mode == "minor":
            self.memory.lower_priorities_all(0.85)
            self.memory.cleanup_old_chunks()
        # يمكن إضافة أنواع أخرى هنا

    def update_priority(self, content: str, new_priority: float):
        """تحديث أولوية عنصر معين وإعادة ترتيبه في الطبقات"""
        if content in self.memory.chunk_index:
            chunk = self.memory.chunk_index[content]
            old_layer = chunk.layer
            chunk.priority = new_priority
            self.memory.move_chunk_to_layer(chunk, self.memory._get_layer_for_priority(new_priority))
