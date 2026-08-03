
# ML-12: Demo Outline + Shareable Cuts

---

## 5-Minute Demo Outline

### Slide 1 — The Question (30 seconds)
**Problem:** FlyRank manages 119,340 content pages across 42 client 
websites. Which pages need refreshing? Manual review is impossible at 
this scale. Can search performance signals tell us automatically?

### Slide 2 — The Method (60 seconds)
**Approach:**
- Data: FlyRank search performance signals (Feb-March 2026)
- Features: CTR change, position volatility, position change
- Model: Balanced Random Forest (class_weight=balanced)
- Split: Grouped by client — tested on 10 completely unseen websites
- Key decision: class_weight=balanced to handle 75/25 imbalance

### Slide 3 — One Chart (60 seconds)
**Feature Importance Chart**
- ctr_change: +0.072 (strongest signal)
- pos_volatility: +0.048 (second strongest)
- imp_prev30: -0.023 (hurts the model — surprising)

**The insight:** CTR change beats impression volume.
Content teams watch traffic. They should watch engagement rate.

### Slide 4 — One Honest Result (90 seconds)
**Results Table:**

| Model | F1 |
|-------|----|
| Rule baseline | 0.155 |
| RF Unbalanced | 0.004 |
| RF Balanced | 0.419 |

**What this means:**
- 2.7x improvement over the manual rule
- Precision = 0.318 — 2 in 3 flags need human review
- This is a triage tool, not an automation system
- The unbalanced RF (F1=0.004) was an honest failure — documented

### Slide 5 — One Recommendation (60 seconds)
**For FlyRank content teams:**
1. Use score >= 0.5 as HIGH PRIORITY tier
2. Assign top 20-50 pages to editors each sprint
3. Start monitoring CTR change, not just impressions
4. After 30-60 days, measure traffic impact of refreshed pages
5. Feed outcomes back to improve the model

**Bottom line:** The model focuses editorial attention. 
It does not replace editorial judgment.

---

## Two Shareable Cuts

---

### Cut 1: LinkedIn / Social Post (Methodology Focus)

---

Built a content refresh scoring system during my ML internship 
at FlyRank.

The problem: 119,340 pages across 42 websites. Which ones need 
updating? Manual review at that scale is impossible.

What I built: A Balanced Random Forest that scores each page by 
its probability of being a refresh opportunity, using only search 
performance signals (no content scraped, no private data).

The surprising finding: CTR change is a stronger refresh signal 
than impression volume. Content teams watch traffic. They should 
watch engagement rate change.

The honest result: F1 = 0.419 vs a rule baseline of F1 = 0.155. 
2.7x improvement. Precision = 0.318 — human review is still 
mandatory. This is a triage tool, not an automation system.

Built on real data. Evaluated on unseen clients. 
All findings are directional and decision-support only.

#MachineLearning #SEO #ContentStrategy #MLInternship #DataScience

---

### Cut 2: Employer-Facing Summary (3 Sentences)

---

During my ML internship at FlyRank, I built a content refresh 
prioritization system trained on 119,340 pages of real search 
performance data across 42 client websites, using a Balanced 
Random Forest with grouped cross-validation to prevent 
client-level data leakage.

The model achieved F1 = 0.419 on completely unseen clients — 
a 2.7x improvement over a hand-coded rule baseline — and 
identified that click-through rate change is a stronger 
predictor of content decline than impression volume, a finding 
that was not captured by any existing manual process.

I handled class imbalance, deliberate leakage testing, 
proxy label design, and pseudonymized data safely, 
and delivered a ranked output with a clear editorial 
action framework tied to real business decisions.

---
