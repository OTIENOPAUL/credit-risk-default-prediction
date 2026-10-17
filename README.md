# Enterprise Credit Risk Scoring & Default Prediction Engine

An end-to-end machine learning pipeline and predictive underwriting engine built using the **Home Credit Default Risk** dataset. This project addresses the dual operational challenges of high non-performing loan (NPL) rates and the exclusion of "thin-file" retail borrowers by replacing legacy rule-based scorecards with a high-performance gradient-boosting model.

---

## 📊 Executive Summary & Key Metrics

* **Business Objective:** Transition from manual, lagging credit reviews to a real-time, data-driven underwriting engine that predicts probability of default (PD) while minimizing false negatives and credit losses.
* **Algorithm:** Regularized LightGBM (Gradient Boosting Decision Trees) with native categorical encoding and early stopping.
* **Key Validation Metrics:**
  * **ROC-AUC:** `0.7651`
  * **Gini Coefficient:** `0.5301`
  * **PR-AUC:** `0.2454`
* **Regulatory Compliance:** Audited via SHAP (SHapley Additive exPlanations) to provide local feature attribution and human-understandable adverse action reason codes.
