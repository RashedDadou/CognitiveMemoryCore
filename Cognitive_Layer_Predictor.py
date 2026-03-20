class CognitiveLayerPredictor:
    def __init__(self):
        # ... نفس التهيئة السابقة
        self.last_major_update = time.time()  # وقت آخر تغيير كبير (إضافة/حذف/أولوية)

    def predict_layer_order(
        self,
        current_mode: str,
        interest_score: float,
        current_layer_state: Dict[int, int],   # {layer: count}
        layer_ages: Dict[int, float],          # جديد: متوسط عمر الطبقة بالساعات
        user_emphasis: float = 0.0,
        global_age_hours: float = 0.0          # عمر الجلسة أو من آخر reset
    ) -> List[int]:
        """
        layer_ages: {0: 0.3, 1: 1.2, 2: 4.5, ...} متوسط عمر الـ chunks في كل طبقة
        global_age_hours: عمر السياق الكلي (اختياري)
        """

        base_order = self.mode_base_order.get(current_mode, [0, 1, 2, 3])

        # تحديد مستوى الاهتمام
        if interest_score > 0.82:
            interest_key = "high"
        elif interest_score > 0.45:
            interest_key = "mid"
        else:
            interest_key = "low"

        weights = self.interest_weights[interest_key]

        layer_scores = []

        for layer in range(5):
            base_rank = base_order.index(layer) if layer in base_order else 999
            base_score = 1.0 - (base_rank * 0.15)

            interest_modifier = weights[layer] if layer < len(weights) else 1.0

            occupancy = current_layer_state.get(layer, 0) / 10.0
            occupancy_bonus = min(0.25, occupancy * 0.4)

            emphasis_bonus = user_emphasis * 0.3 if layer <= 2 else 0.0

            # ─── الجزء الجديد: Temporal Decay على الطبقة ────────────────────────
            age_penalty = 0.0
            if layer in layer_ages:
                layer_age = layer_ages[layer]
                # exponential decay: كلما كبر عمر الطبقة → تنخفض درجتها
                age_penalty = 0.22 * (1 - math.exp(-0.45 * layer_age))

            # عقوبة إضافية لو السياق كله قديم
            global_penalty = 0.08 * (global_age_hours / 4.0) if global_age_hours > 1.0 else 0.0

            final_score = (
                base_score *
                interest_modifier *
                (1 - age_penalty - global_penalty) +
                occupancy_bonus +
                emphasis_bonus
            )

            # لا نسمح بقيمة سالبة
            final_score = max(0.05, final_score)

            layer_scores.append((layer, final_score))

        # ترتيب تنازلي
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_order = [layer for layer, _ in layer_scores]

        return predicted_order
