# AmEx Profitability Engine 🏦  
### American Express Campus Challenge 2026 — Strategy Track, Round 1

> A deterministic, explainable profitability framework for ranking 500,000 Premier Cardmembers using issuer-style P&L logic, boundary analysis, and weight-only optimization against a strong 90% teacher prediction.

---

## The Problem

American Express Campus Challenge Round 1 asks teams to identify the **top 20% most profitable Premier Cardmembers** from a dataset of 500,000 customers.

The challenge has three major constraints:

- No true training labels are provided.
- The evaluation is based on overlap with the actual top 20%.
- Only existing variables can be used; customer ID cannot be used in the scoring equation.

So the main task is not normal supervised machine learning. It is a **rank-ordering problem** where the 80th percentile boundary matters the most.

---

## Our Approach

We built the solution in two stages:

1. **First-principles P&L scoring engine**
2. **Weight-only optimization using a 90% accurate binary teacher prediction**

The idea was to start with a business-driven profitability equation and then analyze where customers swap around the top-20% boundary.

```text
Raw Dataset + 90% Binary Prediction File
        |
        v
ID Mapping and Validation
        |
        v
Original P&L Score Generation
        |
        v
Top-20 Boundary Comparison
        |
        v
False Positive / False Negative Swap Analysis
        |
        v
Coordinate-Descent Weight Optimization
        |
        v
Final Continuous Profitability Prediction
        |
        v
Official Excel Template Export
```

---

## Base Profitability Equation

The original scoring engine models annual issuer profitability as:

```text
Profit =
    Discount Revenue
  + Revolve Interest Income
  + Supplementary Card Revenue
  + Annual Fee
  - Reward Point Liability
  - Benefit Utilization Cost
  - Expected Credit Loss
  - Servicing / Cancellation Cost
  - Fixed Operating Cost
```

The base formula used:

```python
discount_rev = (
    0.026 * f6
  + 0.026 * f9
  + 0.025 * f10
  + 0.024 * f8
  + 0.022 * f7
)

revolve_nii = 0.24 * f1

W_SUPP = 0.00
supp_rev = W_SUPP * max(f19 - 1, 0)

base_annual_fee = 750.0

earned_points = 5*f6 + 2*f9 + max(f7, 0) + f8 + f10
reward_cost = 0.003 * earned_points

benefit_cost = f14 + f16 + 30*f13 + 15*f15

ead = f1 + 0.15 * (f6 + f7 + f8 + f9 + f10)
ecl_cost = 0.75 * f11 * ead

call_cost = 200*f2 + 550*f3

final_profit =
    discount_rev
  + revolve_nii
  + supp_rev
  + base_annual_fee
  - reward_cost
  - benefit_cost
  - ecl_cost
  - call_cost
  - 120.0
```

The submitted prediction is the **continuous profitability score**, not a binary 0/1 flag.

---

## Variables Used

| Variable Group | Features | Role |
|---|---|---|
| Spend Revenue | `f6`, `f7`, `f8`, `f9`, `f10` | Category-level discount/interchange revenue |
| Revolve Revenue | `f1` | Interest income from revolving balance |
| Risk | `f11` | Expected credit loss penalty |
| Benefits | `f13`, `f14`, `f15`, `f16` | Lounge, airline credit, cab, entertainment cost |
| Rewards | `f4`, `f21`, spend variables | Reward point liability |
| Servicing / Churn | `f2`, `f3` | Cancellation and collection-call penalties |
| Relationship Depth | `f19`, `f20` | Supplementary and active card relationship |
| Mapping Only | `id` | Used only to map predictions to template, not used in scoring |

---

## Data Preprocessing

The preprocessing was kept simple and business-consistent:

```text
Missing spend vectors        → filled with 0
Missing rewards/benefits     → filled with 0
Missing risk score           → filled with median
Missing active card count    → filled with 1
Benefit usage variables      → capped at business limits
```

Business caps applied:

```python
f13 = clip(f13, upper=9)
f14 = clip(f14, upper=250)
f15 = clip(f15, upper=200)
f16 = clip(f16, upper=280)
```

---

## Teacher-Based Optimization

After generating a strong 90% binary prediction file, we used it as a **teacher signal**.

The 90-score file contained:

```text
ID | Prediction
```

where `Prediction` was binary:

```text
1 = predicted top 20%
0 = not top 20%
```

We mapped this file back to the original dataset by `ID`.

Then we compared:

```text
Original P&L top 20%
vs
90-score teacher top 20%
```

This created four groups:

| Segment | Meaning |
|---|---|
| TP | Original P&L selected and teacher also selected |
| FP | Original P&L selected but teacher rejected |
| FN | Teacher selected but original P&L missed |
| TN | Both rejected |

The most important groups were:

```text
FP = customers wrongly promoted by original formula
FN = customers missed by original formula
```

The optimization goal was:

```text
Promote FN-like customers
Demote FP-like customers
```

---

## Weight-Only Optimization

We did not change the structure of the formula.  
Only the weights were allowed to change.

The optimized weights were found using coordinate descent:

```text
For each weight:
    Try multiple candidate values
    Recalculate profitability score
    Extract top 20%
    Compare overlap with 90-score teacher
    Keep the weight only if overlap improves
```

Weights optimized:

```text
w_f6
w_f9
w_f10
w_f8
w_f7
w_revolve
w_supp
w_reward
w_f14
w_f16
w_f13
w_f15
w_ecl
w_f2
w_f3
```

Fixed constants:

```text
annual_fee = 750.0
fixed_cost = 120.0
```

---

## Validation and Diagnostics

The solution prints several checks before exporting:

- Row count verification
- Unique ID check
- Missing prediction check
- Teacher positive count check
- Base overlap with teacher
- Optimized overlap with teacher
- Old false positives vs new false positives
- Old false negatives vs new false negatives
- Top-20 feature lift diagnostics
- Weight comparison table
- MD5 fingerprint of final prediction vector

This avoids uploading stale or wrong files.

---

## Final Output

The final output is an Excel file with two sheets:

```text
Predictions
Profitability Framework
```

The `Predictions` sheet contains:

```text
ID | Prediction
```

where `Prediction` is the final continuous optimized profitability score.

The `Profitability Framework` sheet explains:

- variables used
- equation logic
- coefficient derivation
- feature transformations
- business assumptions
- validation method
- shortcomings

---

## Important Shortcoming

This method optimizes against the **90-score teacher file**, not against the true hidden Amex labels.

So improvement in teacher overlap does not guarantee a leaderboard jump from 90% to 93%+.

The real limitation is:

```text
We do not know which 10% of the teacher file is wrong.
```

The weight search can only infer likely corrections from boundary swaps.  
A final leaderboard submission is required to confirm whether the optimized weights truly improve hidden accuracy.

---

## Why This Approach Makes Business Sense

The final solution is explainable and issuer-style.

It does not use black-box ID leakage.  
It does not alter row order.  
It does not add external data.  
It preserves a real profitability equation.

The key improvement is that it learns which parts of the P&L equation were overweighted or underweighted near the 80th percentile decision boundary.

---

## Project Structure

```text
amex_profitability_engine/
│
├── notebooks/
│   └── amex_weight_optimization.ipynb
│
├── data/
│   ├── original_dataset.csv
│   └── teacher_90_prediction.xlsx
│
├── output/
│   └── final_optimized_submission.xlsx
│
├── docs/
│   └── framework_notes.md
│
└── README.md
```

---

## How to Run

Run the notebook block by block:

```text
Block 1 → Upload dataset and 90-score prediction file
Block 2 → Run weight-only optimization and export final template
```

Required packages:

```bash
pip install pandas numpy openpyxl xlsxwriter
```

---

## Tech Stack

```text
Python
pandas
NumPy
openpyxl
XlsxWriter
Google Colab
```

---

## Author

**Abhinav Kumar**  
IIT Kanpur  
American Express Campus Challenge 2026 — Strategy Track
