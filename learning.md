# Black Box Reimbursement - Learning Log

This document tracks the evolution of our understanding and modeling approach for the legacy reimbursement system.

---

## Iteration 1: Broad-Strokes XGBoost Model

### 1. Initial Approach

*   **Hypothesis:** The reimbursement amount is a complex but learnable function of the three inputs (`days`, `miles`, `receipts`) and their simple derivatives.
*   **Methodology:**
    1.  Perform a comprehensive Exploratory Data Analysis (EDA) to find basic correlations and patterns.
    2.  Engineer a wide range of general-purpose features:
        *   Derived metrics (`miles_per_day`, `spend_per_day`).
        *   Log and polynomial transforms to capture non-linearity.
        *   Simple interaction terms (`days * miles`, etc.).
        *   One-hot encodings for trip duration.
        *   Basic piecewise splits for `miles` and `receipts`.
    3.  Train an XGBoost model, a powerful gradient-boosting algorithm capable of capturing complex interactions automatically.
    4.  Use cross-validation to ensure the model was robust.

### 2. Results & Observations

*   **Performance:** The model performed poorly on the evaluation set.
    *   **Exact Matches:** 0.2%
    *   **Average Error:** ~$35.17
    *   **Score:** ~3617
*   **Key Failure:** The model successfully learned the general trends (e.g., more days/miles/receipts generally leads to higher reimbursement) but failed spectacularly on cases governed by specific, unwritten "business rules".
*   **High-Error Case Analysis:** The cases with the largest errors (like #684 and #711) were those where the model predicted a high reimbursement, but the actual was much lower. This points to **punitive rules** or **caps** that the general-purpose features did not capture.

### 3. Key Learnings & Course Correction

The initial "let the machine figure it out" approach was insufficient because the system is not just complex, it's **idiosyncratic**. It's a collection of specific rules layered on top of each other over time. The interview data is not just "color," it's the **blueprint for the actual algorithm**.

**Moving forward, we must transition from generic feature engineering to explicit, hypothesis-driven feature engineering based on the interviews.**

*   **Insight 1: Penalties are Crucial.** The system punishes certain behaviors. The largest errors come from the model not knowing when to *reduce* the reimbursement. We must explicitly model:
    *   **The Low-Receipt Penalty:** Submitting `receipts < $50` is worse than submitting zero.
    *   **The "Vacation" Penalty:** A combination of long trips (`>8 days`) and high daily spending.

*   **Insight 2: Bonuses are Conditional.** The "5-day bonus" isn't for every 5-day trip. It's a reward for a specific *type* of 5-day trip: high efficiency and modest spending. This requires a multi-condition feature.

*   **Insight 3: The System has Quirks.** The `.49` or `.99` cent rounding "bug" is a real, implementable rule that can help capture small variations and improve the exact match score.

### 4. Plan for Iteration 2

1.  **Codify Interview Rules as Features:** Create new, highly specific features in `build_model.py` that directly represent the rules learned from the interviews.
2.  **Retrain the Model:** Use the same XGBoost algorithm but with the new, more intelligent feature set.
3.  **Validate Against High-Error Cases:** Confirm that the new features correctly predict the lower outcomes for the previously identified high-error cases.
4.  **Update `learning.md`** with the results of the next iteration.

---

## Iteration 2: Hypothesis-Driven Feature Engineering

### 1. Approach

*   **Hypothesis:** The largest source of error in the previous model was its failure to account for specific, punitive "business rules" hinted at in the interviews. By explicitly modeling these rules as features, we can significantly improve accuracy.
*   **Methodology:**
    1.  Reviewed the `INTERVIEWS.md` file to extract concrete, testable rules.
    2.  Engineered a new set of features in `build_model.py` and `predict.py` to represent these rules directly:
        *   `bonus_5_day_sweet_spot`: A flag for the conditional 5-day trip bonus.
        *   `penalty_vacation`: A flag for long trips with high daily spending.
        *   `penalty_low_receipts`: A flag for submitting receipts between $0 and $50.
        *   `efficiency_score`: A Gaussian feature to model the "bell curve" of the efficiency bonus.
        *   `bonus_rounding_bug`: A flag for receipt cents ending in .49 or .99.
        *   `penalty_overspending`: Flags for violating trip-length-based daily spending caps.
    3.  Retrained the same XGBoost model architecture with this new, more intelligent feature set.

### 2. Results & Observations

*   **Performance:** The new model was a significant improvement across all metrics.
    *   **Exact Matches:** 1.2% (up from 0.2%)
    *   **Close Matches:** 28.4% (up from 19.7%)
    *   **Average Error:** ~$25.01 (down from ~$35.17)
    *   **Score:** ~2599 (down from ~3617)
*   **Key Success:** The new features correctly identified cases where reimbursement should be lower than a simple model would predict. The maximum error was cut by more than half (from ~$1073 to ~$437).
*   **New High-Error Case Analysis:** The new list of top errors is different. We are no longer just wildly over-predicting. Case #626, for example, is a significant *under-prediction* (`Expected: $1180.63, Got: $743.16`). This suggests there are scenarios that trigger **unaccounted-for bonuses** or have less severe penalties than our model assumes.

### 3. Key Learnings & Course Correction

Explicitly modeling the interview rules was highly effective. The model is no longer naive. However, it still lacks a complete understanding of the system's nuances.

*   **Insight 1: Under-prediction is a New Clue.** The model is now sometimes too punitive. The rules we've added (like `penalty_overspending`) might be too aggressive or might have exceptions. Case #626 (14 days, low mileage, low receipts) was penalized, but the actual reimbursement was high. This could indicate a base per-diem that is much stronger and less dependent on receipts/miles for very long trips.
*   **Insight 2: The "Base" Calculation is Still a Mystery.** While we've added features for bonuses and penalties, the core calculation (`days * X + miles * Y + receipts * Z`) is still being approximated by the model. The interaction terms are doing a lot of the work, but there might be a more explicit formula we can derive. For example, maybe the per-diem isn't a flat $100 but changes based on trip length (e.g., `$120` for days 1-3, `$100` for 4-7, `$80` for 8+).
*   **Insight 3: Feature Interactions Need Refinement.** The model's top features are now the interaction terms (`days_x_receipts`, `miles_x_receipts`). This confirms Kevin's primary theory. Our next step should be to explore more complex and targeted interactions.

### 4. Plan for Iteration 3

1.  **Analyze Under-prediction Cases:** Specifically investigate cases like #626. What do long, low-mileage, low-receipt trips have in common? Hypothesize a "long-stay bonus" or a different base-pay structure for extended trips.
2.  **Refine the Base Rate Model:** Instead of one-hot encoding every day, try creating features for trip duration *brackets* (e.g., `is_short_trip`, `is_medium_trip`, `is_long_trip`) and interact those with the core inputs. This might help the model learn different base rates for different trip types.
3.  **Hyperparameter Tuning:** Now that the feature set is more stable and intelligent, perform a more rigorous hyperparameter search (e.g., using GridSearchCV or RandomizedSearchCV) on the XGBoost model to find the optimal `max_depth`, `learning_rate`, and `n_estimators`. The current parameters are a good starting point, but they may not be optimal.
4.  **Ensemble Modeling (Optional):** If a single model can't capture all the quirks, consider a simple ensemble. For example, train one model that specializes in trips under 8 days and another for trips 8 days and over, and use a rule to combine their predictions. This can sometimes capture fundamentally different logic paths in the legacy system.
5.  **Update `learning.md`** with the results.

---

## Iteration 3: Premature Optimization

### 1. Approach
*   **Hypothesis:** With a decent feature set in place, the model could be further improved by adding more complex features to address under-predictions and by performing a broad search for better hyperparameters.
*   **Methodology:**
    1.  Added new features to `feature_engineering.py` specifically to address the under-prediction on long trips (`bonus_long_stay` and trip duration brackets).
    2.  Replaced the simple training function in `build_model.py` with a `GridSearchCV` to automatically find the best combination of `n_estimators`, `max_depth`, `learning_rate`, etc.

### 2. Results & Observations
*   **Performance:** The model's performance **regressed**. The average error increased from ~$25 to ~$34.
*   **Key Failure:** We tried to do two things at once (add new features and tune hyperparameters) without validating each change independently. The new features were likely not specified correctly, and the hyperparameter search may have found a model that was more general but less adapted to the specific quirks of this dataset.

### 3. Key Learnings & Course Correction
*   **Validate a Trustworthy Test Harness:** The slow `eval.sh` script was hindering development. We built and validated `fast_eval.py`, confirming it produces consistent results, which enables a much faster and more scientific iteration loop. This was the most important outcome of this phase.
*   **Don't Optimize Prematurely:** The `GridSearchCV` is a powerful tool, but it's best used as a final polishing step once the feature set is stable and proven. Using it too early, on a flawed feature set, did not lead to better results.
*   **Re-establish the Baseline:** We reverted the codebase to the simpler, more effective Iteration 2 model. This "champion" model now serves as the reliable baseline against which all new ideas must be compared.

---

## Iteration 4: Surgical Error Analysis

### 1. Approach
*   **Hypothesis:** Instead of guessing what features to add next, we can use the model's biggest mistakes to guide us. By systematically analyzing the cases with the highest error (the residuals), we can find patterns and create features that directly address those specific weaknesses.
*   **Methodology:**
    1.  Created a new `error_analysis.py` script to load the best model (Iteration 2) and analyze its errors.
    2.  The analysis revealed the model was weakest on **(a)** very long-mileage trips and **(b)** trips with very low receipt amounts.
    3.  We tested two new, highly specific features, one at a time, using our `fast_eval.py` loop:
        *   A `bonus_high_mileage` feature.
        *   A calculated `penalty_low_receipts_amount`.

### 2. Results & Observations
*   **Performance:** This approach was highly successful, yielding our best model to date.
    *   **Average Error:** Dropped from ~$25.01 to **$22.70**.
    *   **Exact Matches:** Increased from 12 to **24**.
    *   **Close Matches:** Increased from 284 to **335**.
*   **Key Success:** The iterative, "one change at a time" approach, validated by the fast evaluation script, proved to be a robust and effective strategy. We are no longer guessing; we are surgically improving the model based on data.

### 3. Key Learnings & Course Correction
*   **The "Residual-First" Method Works:** Analyzing the errors is the most direct path to understanding a model's deficiencies.
*   **Calculated Features are Powerful:** The `penalty_low_receipts_amount` feature, which calculates a penalty instead of just being a binary flag, was more effective. This implies the legacy system contains nuanced calculations, not just on/off rules.

### 4. Plan for Iteration 5
1.  **Analyze the *New* Errors:** Run `error_analysis.py` on our current best model (from Iteration 4) to identify the next frontier of problems.
2.  **Formulate New Hypotheses:** Examine the new top-offending cases and look for common themes. Is there a cap on total reimbursement? Is there a different logic path for trips with zero mileage?
3.  **Continue Surgical Implementation:** Add one new feature at a time and use `fast_eval.py` to confirm its impact before committing to it.
4.  **Final Polish:** Once the feature engineering is complete, perform one final `GridSearchCV` to squeeze out any remaining performance gains.
5.  **Update `learning.md`**.

---

## Iteration 5: The "Two-Path" Hypothesis

### 1. Approach
*   **Hypothesis:** The system uses two fundamentally different calculation paths: a "standard model" for most trips and a "long-stay / per-diem" model for long trips with low expenses.
*   **Methodology:**
    1.  Based on the error analysis of the Iteration 4 model, we identified that long, low-activity trips were still major outliers.
    2.  Created a new feature `is_long_stay_path` to flag these trips.
    3.  Interacted this flag with the core `days` and `receipts` features, hoping the model would learn a different set of coefficients for these cases.

### 2. Results & Observations
*   **Performance:** This approach **failed to improve the model**. The average error increased from ~$22.70 to ~$28.04.
*   **Key Failure:** While the hypothesis was sound, the implementation was likely too simplistic. A single binary flag and two interaction terms were not enough for the model to learn a completely separate calculation logic. The negative interactions from the main features likely overpowered the new, more specific features.

### 3. Key Learnings & Course Correction
*   **Modeling Alternate Logic Paths is Hard:** Capturing distinct, mutually exclusive calculation paths within a single regression model is non-trivial. The model will always try to find a smooth fit across all data points, and a few outlier cases might not be enough to carve out a separate logic branch without more explicit guidance.
*   **Our Best Model is from Iteration 4:** We have now confirmed through experimentation that the model with the refined `bonus_high_mileage` and `penalty_low_receipts_amount` features is our reigning champion. This is our new, reliable baseline.
*   **The Next Hypothesis Must Be Precise:** The remaining errors are concentrated in very specific scenarios. Our next move should not be a broad architectural change (like the "two-path" model) but another surgical feature that addresses a specific, observable pattern in the errors. The error analysis of our best model shows that high-mileage trips with low receipts are a key problem area.

### 4. Plan for Iteration 6
1.  **Formulate a Nuanced Hypothesis:** The `bonus_high_mileage` feature is too aggressive on trips with low receipts. We need to create a feature that *dampens* this bonus when receipts are low.
2.  **Implement a "Dampening" Interaction:** Create a new feature that combines the high-mileage bonus with the receipt amount (e.g., `bonus_high_mileage * log_receipts`).
3.  **Test Surgically:** Add only this feature to our best model (Iteration 4) and use `fast_eval.py` to confirm its impact.
4.  **Consider Advanced Techniques:** If surgical feature engineering reaches its limit, investigate more advanced machine learning techniques specifically suited for this type of problem.
5.  **Update `learning.md`**.

---

## Iteration 6: Explainability-Driven Tuning

### 1. Approach
*   **Hypothesis:** The model's worst over-predictions were caused by the `bonus_high_mileage` feature being too aggressive on trips with low associated receipts. The model needed to learn that high mileage without significant expenses is suspicious.
*   **Methodology:**
    1.  Used the **SHAP** library to analyze the specific feature contributions for our worst-performing cases.
    2.  The SHAP "force plot" for the worst over-prediction visually confirmed that `bonus_high_mileage` was applying a massive, unchecked bonus.
    3.  Implemented a "dampening" feature by modifying the bonus calculation to be `(miles - 1000) * factor * log(receipts)`. This scales the bonus by the magnitude of the receipts, effectively reducing it when expenses are low.

### 2. Results & Observations
*   **Performance:** This was our most successful iteration yet, achieving the best scores across the board.
    *   **Average Error:** Dropped from ~$25.01 to a new low of **$21.42**.
    *   **Exact Matches:** Increased from 24 to **35**.
    *   **Close Matches:** Increased from 335 to **380**.
*   **Key Success:** Using an explainability tool like SHAP allowed us to move beyond analyzing *what* the errors were to understanding *why* they were happening. This allowed for a far more precise and effective feature change.

### 3. Key Learnings & Course Correction
*   **Explainable AI (XAI) is a High-Value Tool:** For complex, rule-based systems like this, understanding the "why" behind a prediction is more valuable than just measuring the error. SHAP proved indispensable.
*   **The Final Frontier:** The SHAP plot for our worst *under-prediction* (Case #625) clearly shows the model is still applying standard penalties too aggressively on long trips with low activity, overpowering the simple effect of the trip's duration.

### 4. Plan for Iteration 7
1.  **Formulate a Stronger "Long-Stay" Hypothesis:** The system seems to have a default, generous per-diem for legitimate long-stay trips that our model is not capturing. We need to be more forceful in counteracting the penalties the model naturally wants to apply.
2.  **Implement a Per-Diem Bonus Feature:** Create a new feature that applies a strong, daily bonus *only* when the "long-stay, low-activity" conditions are met. This is a more direct way of modeling the "alternate calculation path" we hypothesized earlier.
3.  **Test and Finalize:** Run the fast evaluation loop. If this feature provides a significant improvement, it may be our final model. If not, we can be confident we have reached the reasonable limit of this modeling approach.
4.  **Update `learning.md`**.

---

## Iteration 7: Deeper SHAP Analysis

### 1. Approach
*   **Hypothesis:** Our best model (Iteration 6) still had significant errors on specific outlier cases. By re-running the SHAP analysis on this new best model, we could uncover the precise reasons for the remaining failures.
*   **Methodology:**
    1.  Generated new SHAP summary and force plots for the Iteration 6 model.
    2.  Closely examined the force plots for the new worst under-prediction (Case #625) and worst over-prediction (Case #114).

### 2. Results & Observations
*   **Worst Under-Prediction Insight (Case #625):** The plot clearly showed that even with our `bonus_long_stay_per_diem` feature, the combined negative pressure from multiple low-activity features was still overwhelming it. The bonus was directionally correct but not nearly strong enough.
*   **Worst Over-Prediction Insight (Case #114):** The plot showed that for trips with extremely high receipts, the raw receipt value was pushing the prediction far too high. Our simple binary `penalty_overspending` flag was visible but too weak to counteract this. It was not a true "soft cap."

### 3. Key Learnings & Course Correction
*   **Confirming the Two Final Problems:** The SHAP analysis pinpointed the two exact features that represent the final frontier for improvement: the bonus for long-stays and the penalty for overspending.
*   **Magnitude Matters:** It's not enough to have a feature that points in the right direction; its magnitude must be appropriate to fight against other competing features. Our current implementations for these two rules are too timid.

### 4. Plan for Iteration 8 (Final)
1.  **Strengthen the "Long-Stay" Bonus:** In `feature_engineering.py`, significantly increase the multiplier on the `bonus_long_stay_per_diem` feature to ensure it has the power to override the standard low-activity penalties.
2.  **Strengthen the "Overspending" Penalty:** Replace the binary `penalty_overspending` flag with a calculated penalty that scales with the amount overspent, creating a more effective soft cap.
3.  **Final Evaluation:** Run the `fast_eval.py` script to measure the impact of these two final, surgical changes. If successful, this will be our submission model.
4.  **Update `learning.md`**.

---

## Final Pivot: From Approximation to Algorithmic Discovery

### The Insight: We Are Modeling the Wrong Thing
Our work with XGBoost and SHAP has been a masterclass in behavioral modeling. We have built an exceptional *impersonator* of the legacy system. However, the `PRD` clearly states this is a 60-year-old system. It is not a statistical model; it is a **deterministic program**—a series of hard-coded `if/else` branches and simple arithmetic formulas.

No matter how sophisticated our regression model is, it is, by its nature, an approximator. It will always have a small amount of error because it is trying to find a smooth curve to fit a system that is likely not smooth at all. This is why our "close matches" are high, but our "exact matches" have plateaued at a low number.

To get a perfect score, we must stop impersonating the system and start **discovering its source code.**

### The New Strategy: Symbolic Regression
The correct tool for this new goal is **Symbolic Regression**. Unlike traditional regression, which fits coefficients to a pre-defined formula, symbolic regression uses genetic programming to "evolve" mathematical expressions from basic building blocks (`+`, `-`, `*`, `/`, `sqrt`, variables, constants) to find the formula that best fits the data.

Its output is not a model file; it's a human-readable equation.

### The "Formula Hunter" Plan
Our new mission is to find the original formulas.

1.  **Segment the Data:** Our "Router" concept was correct. We will use it to peel off distinct calculation paths. The "long-stay, low-activity" trips are the first and most obvious segment.
2.  **Discover the Simple Formulas:** For the `long_stay` segment, the formula is likely trivial (e.g., `days * 110 - 50`). We can find this with a simple linear regression.
3.  **Unleash Symbolic Regression on the Core:** For the main `standard` segment, we will use a symbolic regression library (`gplearn`) to hunt for the core reimbursement formula. All of our engineered features will be given to the algorithm as potential components of the final equation.
4.  **Reconstruct the Program:** The final `predict.py` will be a clean, fast, deterministic script with zero machine learning models. It will be a Python implementation of the decision tree and formulas we discover.

This is the final, most advanced step. It treats the problem as a reverse-engineering puzzle, not a data science problem, which is what the situation truly calls for. This is our path to a perfect score.

### Plan for the Next Implementation
1.  **Build the Infrastructure:** Refactor the scripts to support loading multiple models from a `models/` directory.
2.  **Define the First Branch:** Based on our last SHAP analysis, our first rule in the router will be to identify "long-stay, low-activity" trips.
3.  **Train Two Models:** Train the `long_stay_model` and the `standard_model` on their respective data slices.
4.  **Evaluate the Hybrid System:** Use `fast_eval.py` to measure the performance of our new, multi-model system.
5.  **Update `learning.md`**. 