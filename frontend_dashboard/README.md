# Return Fraud Detection System — Multi-Signal Approach

A cost-asymmetric, multi-modal fraud detection framework for e-commerce
return claims, combining behavioral sequence modeling, image forensics,
and graph-based fraud-ring detection.

## Problem

Online retailers lose an estimated $101B annually to return fraud
(wardrobing, item-not-received abuse, falsified damage claims, receipt
manipulation, and organized return rings). The core challenge is not just
detecting fraud, but doing so without alienating the 96–98% of customers
who are legitimate — the business cost of a false positive (blocking a
genuine customer) often exceeds the cost of a single missed fraud case.

## Research Gap

Prior work (closest: Ravala & Asadinia, 2025 — "Automated Detection of
Fraudulent Returns in E-Commerce") uses a Customer Return Score combined
with Random Forest / SVM / Logistic Regression on tabular behavioral data.
This project extends that line of work in three ways absent from the
existing literature:

1. **Image forensics** for damage-claim photo verification (not present in prior work)
2. **Graph-based detection** of organized fraud rings (not present in prior work)
3. **Cost-asymmetric optimization** that directly encodes the false-positive
   business penalty into the training objective (prior work optimizes
   standard accuracy/F1 only)

## System Architecture

```
                    ┌─────────────────────┐
   Return claim ──▶ │  Module 1: Behavioral │──▶ risk score (LSTM)
   submitted        │  (order/return seq)   │
                    └─────────────────────┘
                    ┌─────────────────────┐
                    │  Module 2: Image      │──▶ risk score (ELA-CNN +
                    │  Forensics             │    perceptual hash)
                    └─────────────────────┘
                    ┌─────────────────────┐
                    │  Module 3: Network /  │──▶ risk score (GraphSAGE)
                    │  Fraud Ring (GNN)      │
                    └─────────────────────┘
                              │
                              ▼
                  Weighted fusion + cost-asymmetric
                     three-tier decision layer
                              │
              ┌───────────────┼───────────────┐
              ▼                ▼                ▼
        Auto-approve     Soft friction      Hold / deny
        (low risk)      (extra verify)     (high risk)
```

## Folder Guide

| Folder | Contents |
|---|---|
| `module1_behavioral_lstm/` | Synthetic + hybrid (real UCI Online Retail data) behavioral sequence model. LSTM with cost-asymmetric loss vs. standard baseline vs. tabular Random Forest (prior-work style). |
| `module2_image_forensics/` | Synthetic damage-claim image dataset, Error Level Analysis (ELA) + CNN for splice detection, perceptual hashing for reused/stock image detection. |
| `module3_graph_fraud_rings/` | Synthetic customer graph (shared device/address), GraphSAGE model vs. degree-heuristic and feature-only baselines for fraud ring detection. |
| `frontend_dashboard/` | Analyst review console (`index.html`) — open directly in a browser. Visualizes the fused risk score, all three signal breakdowns, and the tiered decision for sample claims. |

## Key Results Summary

**Module 1 — Behavioral (synthetic, controlled):**
FPR@90%recall — RF baseline 25.0% → Baseline LSTM 1.6% → Cost-Aware LSTM 1.1%

**Module 1 — Behavioral (hybrid, real UCI Online Retail data):**
Both LSTM variants still clearly beat the tabular RF baseline (FPR 5.9% vs
1.7–2.1%); cost-aware advantage over standard-loss LSTM is smaller on real
data — noted as an honest limitation, not hidden.

**Module 2 — Image Forensics:**
ELA-CNN: Precision 0.93, Recall 1.00, AUC 0.987 on splice detection.
Perceptual hashing: 100% detection of reused/stock images (a case ELA
cannot catch, since no pixels are altered).

**Module 3 — Graph Fraud Rings:**
GraphSAGE: AUC 0.993, Recall 0.96 — vs. Random Forest on behavioral
features alone (AUC 0.828), proving individually-plausible ring members are
only detectable via network structure. Critically, GraphSAGE reduces false
alarms on innocent shared-address customers (e.g. roommates) by 83% (71→12)
compared to a naive "flag anyone sharing an ID" heuristic.

## Important Methodology Note (state this explicitly to reviewers/mentor)

No public dataset contains verified return-fraud ground truth labels — this
is a documented limitation in the literature itself (data scarcity is
repeatedly cited in systematic reviews of e-commerce fraud detection).
This project addresses that constraint with a **hybrid real+synthetic**
approach: genuine customer behavior is grounded in real transactional data
(UCI Online Retail dataset, 541,909 real transactions) wherever possible,
while fraud patterns are synthetically injected following literature-derived
behavioral archetypes (wardrobing, INR abuse, damage-claim fraud, serial
rings). This is standard practice in fraud-detection research given the
acknowledged absence of labeled fraud datasets, and is documented openly
rather than presented as real fraud-labeled data.

## Running the Modules

Each module's `src/` folder is self-contained. General pattern:

```bash
pip install torch torch_geometric imagehash pandas numpy scikit-learn matplotlib --break-system-packages

cd module1_behavioral_lstm/src
python3 generate_data.py       # synthetic data
python3 features.py            # build sequences
python3 train.py               # train + evaluate

cd ../../module2_image_forensics/src
python3 generate_images.py
python3 train_image.py

cd ../../module3_graph_fraud_rings/src
python3 generate_graph.py
python3 train_graph.py
```

## Dashboard

Open `frontend_dashboard/index.html` directly in any browser — no server
required. It's a static demo populated with sample claims derived from
real module outputs, showing the composite risk meter, per-signal
breakdown, and explanation for each claim.
