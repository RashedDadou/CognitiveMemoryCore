class CognitiveMemoryCore:
    """
    المحول الرئيسي الوحيد
    كل الوحدات الداخلية تابعة له ولا تُستدعى مباشرة من الخارج
    """

    def __init__(self):
        self.working_memory = LayeredWorkingMemory(self)     # تمرير self عشان يقدر يطلب من المحول
        self.layer_predictor = CognitiveLayerPredictor(self)
        self.retrieval_router = MemoryRouter(self)

        # حالة مشتركة يشاركها الجميع
        self.current_mode = "intuition"
        self.last_major_event = time.time()
        self.global_interest_score = 0.5

    # ──── واجهات عامة نظيفة (اللي هيتعامل معاها الكود الخارجي) ─────────────

    def add(self, content: str, priority: float = 1.0, **kwargs):
        self.working_memory.add_or_update(content, priority, **kwargs)
        self._on_content_change()  # يمكن يحدث الـ predictor أو يعيد حساب

    def get_context(self, custom_interest: Optional[float] = None) -> Dict:
        interest = custom_interest if custom_interest is not None else self.global_interest_score
        predicted_order = self.layer_predictor.predict_order(self.current_mode, interest)
        context_text = self.working_memory.build_context_from_order(predicted_order)
        retrieval_decision = self.retrieval_router.decide(interest, self.current_mode)
        return {
            "context_text": context_text,
            "retrieval_needed": retrieval_decision["retrieval_weight"] > 0.4,
            "predicted_layers": predicted_order,
            "retrieval_depth": retrieval_decision["depth_level"]
        }

    def set_mode(self, mode: str):
        self.current_mode = mode
        self._on_mode_change()

    def reset(self, mode: str = "minor"):
        if mode == "full":
            self.working_memory.clear()
        elif mode == "major":
            self.working_memory.clear_lower_layers()
            self.working_memory.decay_priorities_all(0.65)
        # ... باقي الأنواع
        self.last_major_event = time.time()

    # ──── دوال داخلية يستدعيها الوحدات الفرعية ────────────────────────────────

    def _on_content_change(self):
        # هنا ممكن نحدث الـ global_interest_score أو نعمل re-prioritization خفيف
        pass

    def _on_mode_change(self):
        # ممكن نعدل الـ decay أو نعمل reset جزئي
        pass
