# DECIDE-X: Applied Intelligence for High-Stakes Credit Decisions

DECIDE-X is a production-grade Credit Risk Intelligence platform designed for senior decision-makers and ML auditors. It bridges the gap between raw algorithmic output and defensible business intelligence.

---

## � What problem DECIDE-X solves
Most credit risk systems are **black boxes**. Even if they are accurate, they fail to provide the *why* behind a decision, often hiding systemic biases or failing to signal when they are uncertain. This leads to:
- **Trust Erosion**: Stakeholders cannot verify the logic of "Approved" or "Denied" verdicts.
- **Compliance Risk**: Lack of auditable fairness metrics and human-readable justifications.
- **Strategic Blindness**: Inability to simulate "What-If" scenarios to understand the model's decision boundaries.

## ⚖️ Why accuracy alone is not enough
A model with 95% accuracy can still be a liability.
- **Hidden Bias**: A highly accurate model might be relying on proxy variables for protected attributes.
- **False Confidence**: Systems that don't quantify uncertainty (OOD) will confidently make wrong decisions.
- **Static Rigidity**: Accuracy doesn't tell you how a model's logic changes under stress or synthetic demographic shifts.

---

## 🛡️ System Capabilities

### 1. Risk Prediction (Calibrated)
Unlike standard ML outputs, DECIDE-X provides **Calibrated Risk Scores**. Every prediction is clamped to a realistic `[0.01, 0.99]` range, acknowledging the probabilistic nature of credit risk and avoiding the "100% certainty" fallacy common in immature systems.

### 2. Explainability (SHAPley Multi-Tone)
We utilize a multi-layered explainability engine:
- **SHAPley Value Vectors**: Precise quantification of feature contribution to the final risk score.
- **Narrative Intelligence**: An NLP layer that translates SHAP values into three distinct tones: **Executive**, **Technical**, and **Simple**, allowing for stakeholder-specific briefings.

### 3. Uncertainty Quantification
The platform signals **Internal Confidence** based on model stability and input manifold similarity. If a profile falls into an "Out-of-Distribution" (OOD) zone, the system triggers a **Protocol Fault** or a **Human-in-the-Loop** flag, preventing automated errors on edge cases.

### 4. Algorithmic Fairness (Quantitative)
Fairness is not a badge; it's a metric. DECIDE-X audits every scan for:
- **Demographic Parity Difference (DPD)**
- **Equal Opportunity Difference (EOD)**
- **Treatment Equality**
We display raw numerical deltas to ensure objective transparency.

### 5. Stress Testing (Simulator)
The integrated **Intelligence Simulator** allows auditors to manipulate core variables (e.g., Loan Amount, Income) in real-time. This "What-If" exploration reveals the model's sensitivity and the exact "Path to Reconstitution" for denied applicants.

### 6. Audit Logs (Governance-Ready)
Every scan is recorded in a high-density **Decision Audit Log**, formatted for ISO_27001 compliance. It tracks version IDs, uncertainty levels, and the exact vector of feature contributions used for the verdict.

---

## � Demo Video
> [!NOTE]
> View the full system in action here: [Intelligence Command Center Demo](file:///Users/macbook/.gemini/antigravity/brain/d33bc128-4a1a-4451-b4f0-864e260f377a/walkthrough.md)

---

## ⚠️ Limitations
- **Synthetic Foundation**: Currently trained on an anonymized synthetic credit dataset; requires real-world calibration for specific lending domains.
- **Feature Sparsity**: Limited to 13 core features; production systems typically leverage hundreds of longitudinal data points.
- **Static Thresholds**: Fairness thresholds are currently global; multi-region deployments would require localized parity targets.

## 🚀 Future Improvements
- **Quantile Regression**: Moving from mean probability to full interval estimation for more robust risk-pricing.
- **Adversarial Robustness Testing**: Automated generation of "adversarial borrowers" to test the model's security boundaries.
- **Online Drift Detection**: Integrated feedback loops to detect concept drift in real-time and trigger automated retraining pipelines.

---
*Developed by the Deepmind Applied AI Team (Antigravity)*
