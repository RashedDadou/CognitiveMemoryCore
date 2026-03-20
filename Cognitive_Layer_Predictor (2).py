from typing import List, Dict, Tuple, Optional
import time

class CognitiveLayerPredictor:
    """
    شبكة ربط ديناميكية بين جهد الفكري والطبقات
    تتنبأ بترتيب الطبقات قبل الطلب (Predictive Layer Ordering)
    """
    def __init__(self):
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
        interest_score: float,           # من 0.0 إلى 1.0
        current_layer_state: Dict[int, int],   # عدد الـ chunks في كل طبقة حاليًا {0:3, 1:4, ...}
        user_emphasis: float = 0.0
    ) -> List[int]:
        """
        الدالة التنبؤية الرئيسية
        ترجع ترتيب الطبقات المفضل (مثلاً [0, 2, 1, 3])
        """

        # 1. الترتيب الأساسي حسب الـ mode
        base_order = self.mode_base_order.get(current_mode, [0, 1, 2, 3])

        # 2. تحديد مستوى الاهتمام
        if interest_score > 0.82:
            interest_key = "high"
        elif interest_score > 0.45:
            interest_key = "mid"
        else:
            interest_key = "low"

        weights = self.interest_weights[interest_key]

        # 3. حساب درجة تفضيل لكل طبقة (score)
        layer_scores: List[Tuple[int, float]] = []

        for layer in range(5):  # 0 إلى 4
            # أساس الدرجة = الترتيب في الـ base_order
            base_rank = base_order.index(layer) if layer in base_order else 999
            base_score = 1.0 - (base_rank * 0.15)

            # تعديل حسب الاهتمام
            interest_modifier = weights[layer] if layer < len(weights) else 1.0

            # تعديل حسب حالة الطبقة الحالية (لو الطبقة مليانة → قيمتها ترتفع)
            occupancy = current_layer_state.get(layer, 0) / 10.0  # افتراضي
            occupancy_bonus = min(0.25, occupancy * 0.4)

            # تعديل حسب user_emphasis
            emphasis_bonus = user_emphasis * 0.3 if layer <= 2 else 0.0

            final_score = base_score * interest_modifier + occupancy_bonus + emphasis_bonus

            layer_scores.append((layer, final_score))

        # 4. ترتيب الطبقات تنازليًا حسب الـ score
        layer_scores.sort(key=lambda x: x[1], reverse=True)

        predicted_order = [layer for layer, score in layer_scores]

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
