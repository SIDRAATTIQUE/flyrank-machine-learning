# Refresh / Content Opportunity Scoring
### A Machine Learning Approach to Content Prioritization

**Author:** Sidra Attique  
**Organization:** FlyRank ML Internship  
**Repository:** https://github.com/SIDRAATTIQUE/flyrank-machine-learning  
**Date:** 2026  

---

## Abstract

Content teams managing large websites face an impossible task: 
manually reviewing thousands of pages to decide which need updating. 
This paper presents a machine learning system that scores each page 
by its probability of being a refresh opportunity, based on observed 
search performance signals. Using a Balanced Random Forest trained on 
119,340 pages across 42 websites, the model achieves F1 = 0.419 — 
a 2.7x improvement over a hand-coded rule baseline (F1 = 0.155). 
The key finding is that click-through rate change is a stronger 
refresh signal than impression volume — a pattern a human rule 
would not discover. The system is a triage tool only; human 
editorial judgment is required before any page is updated.

---

## 1. Introduction

### 1.1 The Problem

Search engine optimization requires continuous content maintenance. 
Pages that once ranked well gradually lose visibility as:
- Competitor content improves
- Search intent shifts
- Information becomes outdated
- Algorithm updates change ranking factors

With 119,340 pages across 42 client websites, manual review 
is completely infeasible for a content team of any practical size.

### 1.2 The Current Approach

Most content teams use simple rules:
- Refresh pages older than 12 months
- Refresh pages that lost more than X% of traffic

These rules fail because:
- Age is not a reliable signal — some old pages perform excellently
- Single-metric thresholds miss multi-dimensional patterns
- They require manual tuning for each client

### 1.3 What This Paper Contributes

1. A reproducible ML pipeline for content refresh prioritization
2. Evidence that CTR change outperforms impression volume as a signal
3. Honest evaluation including a deliberate leakage experiment
4. A clear action framework for editorial teams

---

## 2. Data

### 2.1 Source

FlyRank ML Internship Dataset  
HuggingFace: https://huggingface.co/datasets/FlyRank/internship-warehouse  
Table: fact_content_daily_performance  
Development window: February 1 – March 31, 2026  
Sealed test month: June 2026 (not used in this work)

### 2.2 Scale

| Dimension | Value |
|-----------|-------|
| Total pages | 119,340 |
| Total clients | 42 |
| Training rows | 75,002 |
| Test rows | 36,492 |
| Training clients | 27 |
| Test clients | 10 (unseen) |

### 2.3 Privacy and Safety

All client and page identifiers are pseudonymized:
- Real URLs replaced by content_hash_id
- Client names replaced by client_hash_id
- No client-identifying information appears in any output

### 2.4 Features Used

| Feature | Definition |
|---------|-----------|
| ctr_change | CTR in March minus CTR in February |
| pos_volatility | Standard deviation of average position over Feb-March |
| pos_change | Average position in March minus average position in February |
| imp_prev30 | Total impressions in February |
| days_of_data | Count of distinct dates with data |

### 2.5 Leakage Controls

| Risk | How Addressed |
|------|--------------|
| Label-derived fields | trend_direction and trend_pct excluded |
| Future data | clk_last30 (March clicks) tested and removed |
| Client leakage | Grouped split by client_hash_id |
| Sealed data | June 2026 never loaded |

The deliberate leakage test is worth noting: when clk_last30 
was included as a feature, accuracy inflated to 0.789. 
When removed, accuracy dropped to 0.564 — the honest number.

---

## 3. Target Definition

A page is labeled as a refresh opportunity (is_declining = 1) 
if it lost more than 20% of its impressions from February to 
March 2026:

    is_declining = (imp_last30 < 0.8 * imp_prev30)

This is a proxy label, not a direct observation of content quality. 
The threshold of 20% was chosen to balance class sizes and align 
with editorial intuition about meaningful decline.

Base rate: 25.2% of pages are declining (74.8% stable).

---

## 4. Baseline

Before building a model, we established a rule-based baseline 
that represents what a content team might do manually.

### 4.1 The Rule

Flag a page if it lost more than 20% of impressions AND 
its position volatility was stable (std less than 2.0):

    is_refresh_candidate = (
        imp_last30 < 0.8 * imp_prev30
    ) AND (
        pos_volatility < 2.0
    )

### 4.2 Baseline Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.765 |
| Precision | 1.000 |
| Recall | 0.084 |
| F1 | 0.155 |

### 4.3 Interpretation

The baseline is extremely conservative. It achieves perfect 
precision — when it flags a page, it is always right. But it 
finds only 8.4% of actual declining pages. A content team 
using this rule would miss 91.6% of refresh opportunities.

---

## 5. Model

### 5.1 Method

Balanced Random Forest Classifier using scikit-learn:

    RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

### 5.2 Why Random Forest

- Captures non-linear interactions between features
- class_weight='balanced' handles the 75/25 class imbalance
- More honest than Gradient Boosting with only 27 training clients
- Feature importance is interpretable via permutation

### 5.3 Evaluation Design

Split: GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  
Groups: client_hash_id

Grouped by client so the model is tested on completely unseen 
websites — proving generalization, not memorization.

---

## 6. Results

### 6.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Always predict majority | 0.748 | — | — | — |
| Rule baseline | 0.765 | 1.000 | 0.084 | 0.155 |
| RF Unbalanced | 0.743 | 0.677 | 0.002 | 0.004 |
| RF Balanced | 0.564 | 0.318 | 0.612 | 0.419 |

### 6.2 Key Result

The Balanced RF achieves F1 = 0.419 — a 2.7x lift over the 
rule baseline (F1 = 0.155).

Note that accuracy drops from 0.765 to 0.564. This is expected 
and correct. The model intentionally flags more pages 
(higher recall) at the cost of some false positives. 
For a triage system, missing a declining page is more 
costly than flagging a stable one.

### 6.3 Error Analysis

| Error Type | Count | Pattern |
|-----------|-------|---------|
| False Positives | 12,280 | Pages with high pos_volatility and zero ctr_change |
| False Negatives | 3,635 | Pages with subtle, moderate decline signals |

---

## 7. Feature Importance

Permutation importance measured on the grouped test set:

| Feature | Importance | Interpretation |
|---------|-----------|---------------|
| ctr_change | +0.072 | Strongest signal |
| pos_volatility | +0.048 | Second strongest |
| pos_change | +0.008 | Weak positive signal |
| imp_prev30 | -0.023 | Adds noise |
| days_of_data | -0.024 | Not predictive |

### 7.1 Surprising Finding

CTR change outperforms impression volume as a refresh signal.

Content teams typically monitor traffic volume (impressions). 
The model reveals that engagement rate change (CTR) is more 
predictive of content decline. A page losing clicks faster 
than impressions is a page where users are seeing it but 
choosing not to engage — a stronger signal of content 
relevance problems than raw traffic loss.

### 7.2 Negative Finding

The unbalanced RF achieved F1 = 0.004 — nearly useless. 
This confirms that model architecture alone cannot compensate 
for improper handling of class imbalance. This is a 
reproducible and documented finding.

---

## 8. Limitations

1. Single time window (February-March 2026) only
2. Correlation only — cannot prove refresh recovers traffic
3. No content quality signal — search visibility metrics only
4. 42 clients — may not generalize to all industries
5. Proxy label — impression decline is not identical to 
   content staleness
6. Sealed June 2026 test month not yet evaluated

---

## 9. Recommendations

### 9.1 Priority Tiers

| Tier | Score | Action |
|------|-------|--------|
| HIGH_PRIORITY | >= 0.5 | Review this sprint |
| MONITOR | < 0.5 | Standard monitoring |

### 9.2 Editorial Workflow

1. Export ranked list (highest score first)
2. Assign top 20-50 pages to editors
3. For each page ask:
   - Is information outdated?
   - Are there broken links?
   - Is this a business-priority page?
4. Log decision: refresh / monitor / deprioritize
5. Tag refreshed pages with update date
6. After 30-60 days measure traffic impact
7. Feed outcomes back to model

### 9.3 Important Caveat

Precision = 0.318 means approximately 2 out of 3 flagged 
pages may not need refresh. Human review is mandatory. 
This is a triage tool, not an automation system.

---

## 10. Conclusion

This paper presents a content refresh scoring system that 
identifies declining pages with 2.7x the effectiveness of 
a hand-coded rule. The central finding — that CTR change 
is a stronger signal than impression volume — provides 
actionable guidance for content teams beyond the model itself.

The system is honest about its limitations: it is a 
decision-support tool, not a replacement for editorial 
judgment. With precision of 0.318, human review remains 
essential before any content action.

Future work should:
- Remove imp_prev30 and days_of_data (negative importance)
- Test on sealed June 2026 data
- Validate refresh impact with a controlled experiment
- Extend to additional client verticals

---

## Reproducibility

Random seed: 42 (used in all experiments)  
Repository: https://github.com/SIDRAATTIQUE/flyrank-machine-learning  

Run order:
1. w01_research_question.ipynb
2. w02_ml_task_framing.ipynb
3. w03_data_contract.ipynb
4. w04_baseline_score.ipynb
5. w05_model.ipynb
6. w06_validation_audit.ipynb
7. w07_action_playbook.ipynb
8. capstone.ipynb

---

## References

FlyRank ML Internship Dataset  
https://huggingface.co/datasets/FlyRank/internship-warehouse

scikit-learn: Machine Learning in Python  
Pedregosa et al., JMLR 12, pp. 2825-2830, 2011

Breiman, L. Random Forests. Machine Learning 45, 5-32, 2001

---

*Built on the FlyRank ML Internship Dataset*  
*https://flyrank.ai*
