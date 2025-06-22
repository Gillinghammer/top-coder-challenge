# Legacy Reimbursement Engine – Analyst Checklist

Use this as a one-page tracker. Tick each box when the item is **fully** done.

---

## 0. Kick-off
- [ ] Read `README.md`, `PRD.md`, and `INTERVIEWS.md` 📖
- [ ] Clone / pull latest repo & create feature branch `analysis-<name>`
- [ ] Install Python ≥3.9 and create a dedicated virtual-env
- [ ] Copy `run.sh.template` ➜ `run.sh` (leave placeholder call for now)

## 1. Workspace & Tooling
- [ ] Add `requirements.txt` with pinned libs: `pandas`, `numpy`, `scikit-learn`, `lightgbm` (or `xgboost`)
- [ ] Implement `predict.py` skeleton that accepts 3 args and echoes `0.00` (sanity)
- [ ] Build a local test harness mirroring `eval.sh` for rapid loops

## 2. Exploratory Data Analysis (EDA)
- [ ] Load `public_cases.json` into notebook / script
- [ ] Compute derived metrics:
  - [ ] `miles_per_day`
  - [ ] `spend_per_day`
  - [ ] log transforms & indicator flags (e.g. `days==5`)
- [ ] Visualise relationships:
  - [ ] Scatter/box plots vs reimbursement for each raw & derived feature
  - [ ] Heat-maps for `miles × receipts`, colour = reimbursement
- [ ] Fit a naïve baseline formula and plot residuals to hunt patterns

## 3. Hypothesis Testing (from interviews)
- [ ] Check $100 per-diem & 5-day bonus pattern
- [ ] Identify mileage tier break-points (≈100 mi, 300 mi, 600 mi…)
- [ ] Investigate efficiency bonus (180-220 mi/day sweet spot)
- [ ] Confirm receipts diminishing returns & low-spend penalty
- [ ] Search for duplicated inputs with differing outputs (randomness?)

## 4. Feature Engineering
- [ ] Draft piece-wise functions for miles and receipts
- [ ] One-hot trip length 1…10 (+ flag `days==5`)
- [ ] Add interaction terms: `miles_per_day × days`, `spend_per_day × days`
- [ ] Evaluate feature importance via quick GBDT fit; prune unhelpful ones

## 5. Modelling Strategy
### Track A – Rule-based (optional)
- [ ] Hand-build formula matching strongest patterns
- [ ] Test against public set; log MAE & exact-match count

### Track B – Machine Learning (primary)
- [ ] Train/val split (80/20 stratified by `trip_duration_days`)
- [ ] Fit LightGBM/XGBoost with MAE objective
- [ ] Tune depth, leaves, LR until val MAE ≤ $0.20
- [ ] k-fold CV (5×) to ensure stability (std <10 % of mean)

### Hybrid
- [ ] Optionally feed rule-based prediction as an extra feature and re-train

## 6. Export & Integration
- [ ] Save final booster → `model.json` (text, <100 KB)
- [ ] Update `predict.py` to load model once and answer stdin queries
- [ ] `run.sh` should exec `python predict.py "$@"`
- [ ] Round prediction to 2 decimals before print

## 7. Private-set Risk Mitigation
- [ ] Re-run full CV; monitor train vs CV gap
- [ ] Apply input jitter tests; expect <1 % output change
- [ ] Simplify model / add L2 if signs of overfitting

## 8. Performance & Compliance
- [ ] Time `./run.sh 5 250 150.75` (<0.05 s)
- [ ] Ensure dependency list is minimal; vendor lock-free packages if needed
- [ ] Remove large notebooks & data from commit history

## 9. Final Evaluation & Submission
- [ ] Pass `./eval.sh` with ≥990/1000 exact matches, avg error ≤ $0.05
- [ ] Generate `private_results.txt` via `generate_results.sh`
- [ ] Push branch, open PR; add reviewer `arjun-krishna1`
- [ ] Complete Google submission form with final score

---
**Done?** Merge PR ➜ celebrate 🎉 