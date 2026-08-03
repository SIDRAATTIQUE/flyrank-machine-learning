# Refresh / Content Opportunity Scoring
### A Machine Learning Approach to Content Prioritization

**Author:** Sidra Attique  
**Organization:** FlyRank ML Internship  
**Repository:** https://github.com/SIDRAATTIQUE/flyrank-machine-learning  
**Date:** 2026  

---

## Abstract

FlyRank manages content strategy for dozens of client websites. 
Content teams face a real operational problem: with thousands of 
pages across multiple clients, deciding which pages need updating 
is done manually — slowly, inconsistently, and at high cost.

This paper presents a machine learning system built on FlyRank's 
own search performance data that scores each page by its 
probability of being a refresh opportunity. Using a Balanced 
Random Forest trained on 119,340 pages across 42 FlyRank client 
websites, the model achieves F1 = 0.419 — a 2.7x improvement 
over a hand-coded rule baseline (F1 = 0.155).

The key finding is that click-through rate change is a stronger 
refresh signal than impression volume — a pattern invisible to 
manual review and not captured by any existing FlyRank rule. 
The system is a triage tool only; human editorial judgment is 
required before any page is updated.

---

## 1. Introduction

### 1.1 The FlyRank Content Problem

FlyRank helps client businesses grow their search visibility. 
A core part of that work is content: creating, optimizing, and 
maintaining pages that rank well in search results.

The problem is scale. A content team managing 40+ client websites 
cannot manually review every page every month. The current approach 
relies on:
- Manual spot checks driven by client requests
- Simple rules like flagging pages older than 12 months
- Traffic reports that show what happened but not what to do

This means declining pages are often missed until traffic loss 
becomes severe. By the time a page is flagged for refresh, it may 
have lost months of search visibility — directly impacting client 
results and FlyRank's value delivery.

### 1.2 Why This Is Hard

Simple rules fail because content decline is multi-dimensional:
- An old page can still rank excellently if it is accurate and 
  authoritative
- A new page can decline quickly if it was poorly targeted
- Impression volume alone does not distinguish a page that is 
  declining from one that is stable at a lower traffic level
- Position volatility, CTR trends, and volume changes interact 
  in ways a single threshold cannot capture

With 119,340 pages across 42 client websites, manual review 
is completely infeasible for a content team of any practical size.

### 1.3 What This Paper Contributes

This paper presents a data-driven solution built entirely on 
FlyRank's own search performance signals:

1. A reproducible ML pipeline for content refresh prioritization
2. Evidence that CTR change outperforms impression volume as a signal
3. Honest evaluation including a deliberate leakage experiment
4. A clear action framework for FlyRank editorial teams
5. A ranked output that tells editors exactly where to look first

The system does not replace editorial judgment. It focuses it.

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
| Future data | clk_last30 tested and removed |
| Client leakage | Grouped split by client_hash_id |
| Sealed data | June 2026 never loaded |

The deliberate leakage test: when clk_last30 was included, 
accuracy inflated to 0.789. When removed, accuracy dropped 
to 0.564 — the honest number.

---

## 3. Target Definition

A page is labeled as a refresh opportunity (is_declining = 1) 
if it lost more than 20% of its impressions from February to 
March 2026:

    is_declining = (imp_last30 < 0.8 * imp_prev30)

This is a proxy label, not a direct observation of content quality.

Base rate: 25.2% of pages are declining (74.8% stable).

---

## 4. Baseline

### 4.1 The Rule

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

The baseline finds only 8.4% of actual declining pages. 
A FlyRank content team using this rule would miss 91.6% 
of refresh opportunities every month.

---

## 5. Model

### 5.1 Method

    RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

### 5.2 Evaluation Design

Split: GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  
Groups: client_hash_id  

Grouped by client — tested on completely unseen websites.

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

F1 = 0.419 — a 2.7x lift over the rule baseline (F1 = 0.155).

### 6.3 Error Analysis

| Error Type | Count | Pattern |
|-----------|-------|---------|
| False Positives | 12,280 | High pos_volatility, zero ctr_change |
| False Negatives | 3,635 | Subtle moderate decline signals |

---

## 7. Feature Importance

| Feature | Importance | Interpretation |
|---------|-----------|---------------|
| ctr_change | +0.072 | Strongest signal |
| pos_volatility | +0.048 | Second strongest |
| pos_change | +0.008 | Weak positive |
| imp_prev30 | -0.023 | Adds noise |
| days_of_data | -0.024 | Not predictive |

### 7.1 The Key Finding for FlyRank

CTR change outperforms impression volume as a refresh signal.

FlyRank content teams currently monitor traffic volume. 
The model reveals that engagement rate change is more predictive 
of content decline. A page losing clicks faster than impressions 
means users see it but choose not to engage — a stronger signal 
of content relevance problems than raw traffic loss.

This finding changes how FlyRank should triage content work.

---

## 8. Limitations

1. Single time window (February-March 2026) only
2. Correlation only — cannot prove refresh recovers traffic
3. No content quality signal — search visibility metrics only
4. 42 clients — may not generalize to all industries
5. Proxy label — impression decline is not identical to staleness
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
3. For each page ask: Is information outdated? Broken links? Business priority?
4. Log decision: refresh / monitor / deprioritize
5. Tag refreshed pages with update date
6. After 30-60 days measure traffic impact
7. Feed outcomes back to model

### 9.3 Caveat

Precision = 0.318 means 2 out of 3 flagged pages may not need 
refresh. Human review is mandatory. This is a triage tool, 
not an automation system.

---

## 10. Conclusion

This paper presents a content refresh scoring system that 
identifies declining FlyRank client pages with 2.7x the 
effectiveness of a hand-coded rule.

The central finding — that CTR change is a stronger signal 
than impression volume — provides actionable guidance for 
FlyRank content teams beyond the model itself. Teams should 
monitor engagement rate trends, not just traffic volume.

Future work:
- Remove imp_prev30 and days_of_data (negative importance)
- Test on sealed June 2026 data
- Validate refresh impact with a controlled experiment
- Extend to additional client verticals

---

## Reproducibility

Random seed: 42  
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
