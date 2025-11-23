import pickle
import numpy as np
import pandas as pd
import sys, types

from domain.domain import ReviewRequest, ReviewResponse
from service.text_preproc import _clean_batch

# Phần này dùng để lấy hàm clean_batch từ text_preproc mà không bị lỗi 
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "_clean_batch", _clean_batch)
import __main__      
# gắn tên hàm vào __main__ để pickle tìm thấy
__main__._clean_batch = _clean_batch
LABEL_TEXT = {0: "negative", 1: "positive"}  # đổi nếu mapping khác

class ReviewService():
    def __init__(self):
        self.path_model = "artifacts/textclf2.pkl"
        self.model = self.load_artifact(self.path_model)
    def load_artifact(self, path_to_artifact):
        '''Load a prediction artifact from a pickle file'''
        with open(path_to_artifact, "rb") as f:
            artifact = pickle.load(f)
        return artifact 
    def preprocess_input(self, request: ReviewRequest):
        return request.as_list()
    # lấy cột positive ra để dễ xác định, tránh nhầm với negative
    def _positive_col(self, positive_label=1) -> int:
        # lấy từng bước trong pipeline ra và lấy tên model
        clf = getattr(self.model, "named_steps", {}).get("model", None) or self.model.steps[-1][1]
        # lấy class nếu có
        classes_ = getattr(clf, "classes_", None)
        if classes_ is None: return 1
        if positive_label in classes_: return int(np.where(classes_ == positive_label)[0][0])
        if "positive" in classes_:    return int(np.where(classes_ == "positive")[0][0])
        return 1

    def predict_table(self, data):
    # chuẩn hoá X như trước...
        X = [data] if isinstance(data, str) else (data.as_list() if isinstance(data, ReviewRequest) else list(data))

        y = self.model.predict(X)
        # map về nhãn chữ, bỏ hẳn pred_id
        if np.issubdtype(np.array(y).dtype, np.number):
            labels = [LABEL_TEXT.get(int(i), str(i)) for i in y]
        else:
            labels = [str(i) for i in y]

        scores = None
        if hasattr(self.model, "predict_proba"):
            idx = self._positive_col(1)
            scores = self.model.predict_proba(X)[:, idx].astype(float)
        elif hasattr(self.model, "decision_function"):
            scores = np.ravel(self.model.decision_function(X)).astype(float)

        return pd.DataFrame({"review": X, "pred": labels, "score": scores})
    def predict_stt_pred_score(self, stt_and_text: list[tuple[str, str]], threshold: float = 0.5) -> pd.DataFrame:
        """
        Input : list[(stt, text)]
        Output: DataFrame gồm: stt, pred, score
        - pred  : 'positive' hoặc 'negative' theo threshold
        - score : p(positive) từ predict_proba (LogisticRegression, v.v.)
        """
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                "Model không có predict_proba. Hãy dùng classifier hỗ trợ proba (vd: LogisticRegression)."
            )

        stts  = [s for s, _ in stt_and_text]
        texts = [t for _, t in stt_and_text]

        idx_pos = self._positive_col(1)  # tìm cột 'positive' trong classes_
        proba   = self.model.predict_proba(texts).astype(float)
        p_pos   = proba[:, idx_pos]

        preds = np.where(p_pos >= threshold, "positive", "negative")

        return pd.DataFrame({
            "stt":   stts,
            "pred":  preds,
            "score": p_pos,     # chỉ 1 score: xác suất positive
        })
