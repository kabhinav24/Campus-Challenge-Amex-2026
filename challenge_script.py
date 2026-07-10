from google.colab import files
from IPython.display import FileLink, display

import os
import time
import hashlib
import pandas as pd
import numpy as np
import subprocess
import sys

try:
    import xlsxwriter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "XlsxWriter"])
    import xlsxwriter

print("=== BLOCK 2: WEIGHT-ONLY OPTIMIZATION STARTED ===")


# 0. Safety checks from Block 1


assert "df" in globals(), "Run Block 1 first. df is missing."
assert "pred90" in globals(), "Run Block 1 first. pred90 is missing."
assert "id" in df.columns, "Dataset must contain id column."
assert "ID" in pred90.columns, "90 prediction file must contain ID column."
assert "Prediction" in pred90.columns, "90 prediction file must contain Prediction column."

df = df.copy()
df["id"] = df["id"].astype(int)

teacher = pred90[["ID", "Prediction"]].copy()
teacher["ID"] = teacher["ID"].astype(int)
teacher["Prediction"] = pd.to_numeric(teacher["Prediction"], errors="coerce")

assert teacher["Prediction"].isna().sum() == 0, "90 prediction file has non-numeric or missing Prediction values."

unique_teacher_values = sorted(teacher["Prediction"].unique())
print("Teacher unique values:", unique_teacher_values)

assert set(unique_teacher_values).issubset({0, 1, 0.0, 1.0}), "90 prediction file must be binary 0/1."

df = df.merge(
    teacher.rename(columns={"ID": "id", "Prediction": "Teacher_Target"}),
    on="id",
    how="left"
)

assert df["Teacher_Target"].isna().sum() == 0, "Some dataset IDs are missing in 90 prediction file."

df["Teacher_Target"] = df["Teacher_Target"].astype(int)

TOP_N = int(0.20 * len(df))

print("Rows:", len(df))
print("TOP_N:", TOP_N)
print("Teacher positives:", df["Teacher_Target"].sum())
print("Teacher negatives:", (df["Teacher_Target"] == 0).sum())

assert df["Teacher_Target"].sum() == TOP_N, "Teacher file must have exactly top 20% positives."


# 1. preprocessing for P&L formula


d = df.copy()

print("[INFO] Imputing missing vectors...")

for c in ["f6", "f7", "f8", "f9", "f10"]:
    d[c] = d[c].fillna(0.0)

for c in ["f1", "f2", "f3", "f4", "f12", "f13", "f14", "f15",
          "f16", "f17", "f18", "f19", "f21", "f22", "f23"]:
    d[c] = d[c].fillna(0.0)

d["f11"] = d["f11"].fillna(df["f11"].median())
d["f20"] = d["f20"].fillna(1.0)

# Business caps formula
d["f13"] = d["f13"].clip(upper=9)
d["f14"] = d["f14"].clip(upper=250)
d["f15"] = d["f15"].clip(upper=200)
d["f16"] = d["f16"].clip(upper=280)

print("Remaining missing values:", d.isna().sum().sum())


# 2. Vectorized base features


print("[INFO] Building P&L vectors...")

f1, f2, f3   = d["f1"].values, d["f2"].values, d["f3"].values
f6, f7, f8   = d["f6"].values, d["f7"].values, d["f8"].values
f9, f10, f11 = d["f9"].values, d["f10"].values, d["f11"].values
f13, f14     = d["f13"].values, d["f14"].values
f15, f16     = d["f15"].values, d["f16"].values
f19          = d["f19"].values

f7_pos = np.maximum(f7, 0.0)
spend_sum = f6 + f7 + f8 + f9 + f10
earned_points = 5 * f6 + 2 * f9 + f7_pos + f8 + f10
ead = f1 + 0.15 * spend_sum
supp_units = np.maximum(f19 - 1, 0)

teacher_y = d["Teacher_Target"].values.astype(bool)


# 3. Base weights from original formula

BASE_WEIGHTS = {
    "w_f6": 0.026,
    "w_f9": 0.026,
    "w_f10": 0.025,
    "w_f8": 0.024,
    "w_f7": 0.022,

    "w_revolve": 0.24,
    "w_supp": 0.00,

    "annual_fee": 750.0,

    "w_reward": 0.003,

    "w_f14": 1.0,
    "w_f16": 1.0,
    "w_f13": 30.0,
    "w_f15": 15.0,

    "w_ecl": 0.75,

    "w_f2": 200.0,
    "w_f3": 550.0,

    "fixed_cost": 120.0,
}

def compute_score(W):
    score = (
        W["w_f6"] * f6
        + W["w_f9"] * f9
        + W["w_f10"] * f10
        + W["w_f8"] * f8
        + W["w_f7"] * f7

        + W["w_revolve"] * f1
        + W["w_supp"] * supp_units

        + W["annual_fee"]

        - W["w_reward"] * earned_points

        - W["w_f14"] * f14
        - W["w_f16"] * f16
        - W["w_f13"] * f13
        - W["w_f15"] * f15

        - W["w_ecl"] * f11 * ead

        - W["w_f2"] * f2
        - W["w_f3"] * f3

        - W["fixed_cost"]
    )

    return score

def top_mask_from_score(score):
    score = np.asarray(score)
    idx = np.argpartition(score, -TOP_N)[-TOP_N:]
    mask = np.zeros(len(score), dtype=bool)
    mask[idx] = True
    return mask

def teacher_overlap(score):
    m = top_mask_from_score(score)
    return (m & teacher_y).sum() / TOP_N

base_score = compute_score(BASE_WEIGHTS)
base_top = top_mask_from_score(base_score)
base_overlap = teacher_overlap(base_score)

print("\n===== BASELINE =====")
print("Base overlap with 90 teacher:", base_overlap)
print("Base top20 count:", base_top.sum())

# 4. Swap analysis: why customers move around top 20 boundary


print("\n===== SWAP ANALYSIS =====")

d["base_score"] = base_score
d["base_top"] = base_top.astype(int)

d["segment"] = "TN_teacher0_base0"
d.loc[base_top & teacher_y, "segment"] = "TP_teacher1_base1"
d.loc[base_top & (~teacher_y), "segment"] = "FP_base1_teacher0"
d.loc[(~base_top) & teacher_y, "segment"] = "FN_base0_teacher1"

print(d["segment"].value_counts())

component_df = pd.DataFrame({
    "ID": d["id"].values,
    "Teacher_Target": d["Teacher_Target"].values,
    "segment": d["segment"].values,

    "score_disc_f6": BASE_WEIGHTS["w_f6"] * f6,
    "score_disc_f9": BASE_WEIGHTS["w_f9"] * f9,
    "score_disc_f10": BASE_WEIGHTS["w_f10"] * f10,
    "score_disc_f8": BASE_WEIGHTS["w_f8"] * f8,
    "score_disc_f7": BASE_WEIGHTS["w_f7"] * f7,
    "score_revolve": BASE_WEIGHTS["w_revolve"] * f1,
    "score_supp": BASE_WEIGHTS["w_supp"] * supp_units,
    "cost_reward": -BASE_WEIGHTS["w_reward"] * earned_points,
    "cost_f14": -BASE_WEIGHTS["w_f14"] * f14,
    "cost_f16": -BASE_WEIGHTS["w_f16"] * f16,
    "cost_f13": -BASE_WEIGHTS["w_f13"] * f13,
    "cost_f15": -BASE_WEIGHTS["w_f15"] * f15,
    "cost_ecl": -BASE_WEIGHTS["w_ecl"] * f11 * ead,
    "cost_f2": -BASE_WEIGHTS["w_f2"] * f2,
    "cost_f3": -BASE_WEIGHTS["w_f3"] * f3,
})

component_cols = [
    "score_disc_f6", "score_disc_f9", "score_disc_f10", "score_disc_f8", "score_disc_f7",
    "score_revolve", "score_supp", "cost_reward",
    "cost_f14", "cost_f16", "cost_f13", "cost_f15",
    "cost_ecl", "cost_f2", "cost_f3"
]

swap_component_report = component_df.groupby("segment")[component_cols].mean().T
display(swap_component_report)

raw_cols = [
    "f1", "f2", "f3", "f6", "f7", "f8", "f9", "f10",
    "f11", "f13", "f14", "f15", "f16", "f19", "f21"
]

swap_raw_report = d.groupby("segment")[raw_cols].mean().T
display(swap_raw_report)

if "FP_base1_teacher0" in d["segment"].unique() and "FN_base0_teacher1" in d["segment"].unique():
    fp = d["segment"] == "FP_base1_teacher0"
    fn = d["segment"] == "FN_base0_teacher1"

    direction_rows = []

    for c in raw_cols:
        fp_mean = d.loc[fp, c].mean()
        fn_mean = d.loc[fn, c].mean()
        direction_rows.append([c, fp_mean, fn_mean, fn_mean - fp_mean])

    direction_report = pd.DataFrame(
        direction_rows,
        columns=["feature", "wrong_top_FP_mean", "missed_top_FN_mean", "FN_minus_FP"]
    ).sort_values("FN_minus_FP", ascending=False)

    print("Features higher in missed teacher positives than wrong base positives:")
    display(direction_report)
else:
    print("No FP/FN swap against teacher. Base formula may already reproduce the 90 file.")


# 5. Weight-only coordinate descent optimization


print("\n===== WEIGHT-ONLY OPTIMIZATION =====")

def objective(W):
    # Maximize top20 overlap with 90-score teacher file
    return teacher_overlap(compute_score(W))

search_keys = [
    "w_f6", "w_f9", "w_f10", "w_f8", "w_f7",
    "w_revolve", "w_supp",
    "w_reward",
    "w_f14", "w_f16", "w_f13", "w_f15",
    "w_ecl", "w_f2", "w_f3"
]

BOUNDS = {
    "w_f6": (0.002, 0.080),
    "w_f9": (0.002, 0.080),
    "w_f10": (0.002, 0.080),
    "w_f8": (0.002, 0.080),
    "w_f7": (0.002, 0.080),

    "w_revolve": (0.02, 0.80),
    "w_supp": (0.00, 100.00),

    "w_reward": (0.0002, 0.020),

    "w_f14": (0.05, 5.00),
    "w_f16": (0.05, 5.00),
    "w_f13": (2.00, 100.00),
    "w_f15": (2.00, 80.00),

    "w_ecl": (0.05, 3.00),
    "w_f2": (5.00, 800.00),
    "w_f3": (20.00, 1500.00),
}

def candidate_values(key, current):
    low, high = BOUNDS[key]

    if current == 0:
        vals = np.array([0, 5, 10, 20, 35, 43.75, 60, 80, 100], dtype=float)
    else:
        factors = np.array([
            0.35, 0.50, 0.65, 0.80, 0.90, 0.97,
            1.00,
            1.03, 1.10, 1.25, 1.50, 2.00, 3.00
        ])
        vals = current * factors

    vals = np.clip(vals, low, high)
    vals = np.unique(np.round(vals, 10))
    return vals

best_W = BASE_WEIGHTS.copy()
best_score = compute_score(best_W)
best_overlap = teacher_overlap(best_score)

history = []

print("Starting overlap:", best_overlap)

N_ROUNDS = 4

start_time = time.time()

for round_i in range(N_ROUNDS):
    print(f"\n----- ROUND {round_i + 1}/{N_ROUNDS} -----")

    improved = False

    for key in search_keys:
        current = best_W[key]

        local_best_W = best_W.copy()
        local_best_score = best_score.copy()
        local_best_overlap = best_overlap

        for val in candidate_values(key, current):
            trial_W = best_W.copy()
            trial_W[key] = float(val)

            trial_score = compute_score(trial_W)
            trial_overlap = teacher_overlap(trial_score)

            if trial_overlap > local_best_overlap:
                local_best_W = trial_W.copy()
                local_best_score = trial_score.copy()
                local_best_overlap = trial_overlap

        if local_best_overlap > best_overlap:
            print(
                f"{key}: {best_W[key]} -> {local_best_W[key]} | "
                f"overlap {best_overlap:.6f} -> {local_best_overlap:.6f}"
            )

            best_W = local_best_W.copy()
            best_score = local_best_score.copy()
            best_overlap = local_best_overlap
            improved = True

        history.append([round_i + 1, key, best_W[key], best_overlap])

    if not improved:
        print("No improvement in this round. Stopping early.")
        break

history_df = pd.DataFrame(history, columns=["round", "weight", "value", "teacher_overlap"])

best_top = top_mask_from_score(best_score)

print("\n===== OPTIMIZED RESULT =====")
print("Base overlap:", base_overlap)
print("Optimized overlap:", best_overlap)
print("Optimized top20 count:", best_top.sum())
print("Runtime minutes:", (time.time() - start_time) / 60)

print("\n===== OPTIMIZED WEIGHTS =====")
for k, v in best_W.items():
    print(f"{k}: {v}")

weight_compare = []

for k in best_W:
    base_val = BASE_WEIGHTS[k]
    opt_val = best_W[k]

    if base_val != 0:
        ratio = opt_val / base_val
    else:
        ratio = np.nan

    weight_compare.append([k, base_val, opt_val, opt_val - base_val, ratio])

weight_compare_df = pd.DataFrame(
    weight_compare,
    columns=["weight", "base", "optimized", "delta", "ratio"]
)

display(weight_compare_df)


# 6. Final optimized diagnostics


d["optimized_score"] = best_score
d["optimized_prediction"] = best_top.astype(int)

print("\n===== OPTIMIZED TOP20 FEATURE LIFT =====")

rows = []

for c in raw_cols:
    overall = d[c].mean()
    top_mean = d.loc[best_top, c].mean()
    lift = top_mean / (overall + 1e-9)
    rows.append([c, overall, top_mean, lift])

optimized_diag = pd.DataFrame(
    rows,
    columns=["feature", "overall_mean", "optimized_top20_mean", "lift"]
).sort_values("lift", ascending=False)

display(optimized_diag)

# Movement summary
old_fp = base_top & (~teacher_y)
old_fn = (~base_top) & teacher_y

new_fp = best_top & (~teacher_y)
new_fn = (~best_top) & teacher_y

print("\n===== ERROR MOVEMENT VS 90 TEACHER =====")
print("Old false positives:", old_fp.sum())
print("Old false negatives:", old_fn.sum())
print("New false positives:", new_fp.sum())
print("New false negatives:", new_fn.sum())
print("Top20 overlap old vs optimized:", (base_top & best_top).sum() / TOP_N)

# Fingerprint
score_bytes = pd.Series(best_score).round(8).astype(str).str.cat(sep="|").encode()
score_md5 = hashlib.md5(score_bytes).hexdigest()

print("Optimized prediction MD5:", score_md5)

# 7. Upload blank template and fill output

print("\nUpload official blank Amex submission template XLSX")
uploaded_template = files.upload()
TEMPLATE_PATH = list(uploaded_template.keys())[0]

pred_template = pd.read_excel(TEMPLATE_PATH, sheet_name="Predictions")
framework_template = pd.read_excel(TEMPLATE_PATH, sheet_name="Profitability Framework")

assert "ID" in pred_template.columns, "Predictions sheet must contain ID"
assert "Prediction" in pred_template.columns, "Predictions sheet must contain Prediction"
assert "Section" in framework_template.columns, "Profitability Framework sheet must contain Section"
assert "Response" in framework_template.columns, "Profitability Framework sheet must contain Response"

optimized_predictions = pd.DataFrame({
    "ID": d["id"].astype(int),
    "Prediction": best_score
}).sort_values("ID").reset_index(drop=True)

pred_template["ID"] = pred_template["ID"].astype(int)

if len(pred_template) > 0:
    final_prediction_sheet = pred_template[["ID"]].merge(
        optimized_predictions,
        on="ID",
        how="left"
    )
else:
    final_prediction_sheet = optimized_predictions.copy()

missing = final_prediction_sheet["Prediction"].isna().sum()
print("Missing mapped predictions:", missing)
assert missing == 0, "Some template IDs are missing from optimized predictions."

optimized_weights_text = "\n".join([f"{k}: {v}" for k, v in best_W.items()])

framework_answers = {
    "Variables Used": (
        "The model uses the existing behavioral variables from the dataset. Category spend variables estimate discount/interchange revenue, "
        "average revolving balance estimates interest income, rewards and benefit variables estimate issuer costs, risk score estimates expected credit loss, "
        "and cancellation or collection calls estimate servicing penalties. ID is used only for mapping into the template and is not used in scoring."
    ),

    "Profitability Equation": (
        "The submitted score is an optimized continuous P&L score: Profit = category discount revenue + revolve interest income + supplementary account revenue "
        "+ annual fee - reward point liability - benefit utilization cost - expected credit loss - servicing call penalties - fixed cost. "
        "Only the coefficients of the same original formula are optimized; the structure of the equation is unchanged."
    ),

    "Prediction Logic": (
        "Customers are ranked by optimized annual profitability. The optimization compares the original formula top 20 percent with the 90-score teacher file, "
        "studies accounts that swap around the 80th percentile boundary, and adjusts only the formula weights to promote profiles similar to missed teacher positives "
        "while demoting profiles similar to false positives."
    ),

    "Variable Selection Logic": (
        "f6, f7, f8, f9, and f10 are spend revenue drivers. f1 is the revolve revenue driver. f19 contributes supplementary card relationship value. "
        "Earned points, f13, f14, f15, and f16 are cost drivers. f11 combined with exposure estimates credit loss. f2 and f3 represent cancellation and collection servicing penalties."
    ),

    "Coefficient/Weight Derivation": (
        "Weights are selected through coordinate-descent search around the original first-principles P&L equation. Each coefficient is tested one at a time, "
        "and accepted only when it improves top-20 overlap against the 90-score teacher file. The optimized weights are:\n"
        + optimized_weights_text
    ),

    "Feature Transformations": (
        "Missing spend and activity vectors are imputed with zero, missing risk score is filled with the median, and missing active charge card count is set to one. "
        "Benefit usage variables are capped using the same business caps as the original solution. The final prediction remains a continuous profitability score."
    ),

    "Business Logic": (
        "Premier Card profitability is modeled as issuer revenue minus issuer costs. Strong spend, controlled revolve, and relationship depth increase profitability, "
        "while reward accrual, benefit usage, expected credit loss, cancellation calls, and collection calls reduce profitability."
    ),

    "Assumptions": (
        "The 90-score file is treated as a strong but imperfect teacher signal. Remaining mistakes are assumed to be concentrated near the 80th percentile decision boundary. "
        "The optimization therefore focuses on boundary movement rather than changing the entire dataset structure."
    ),

    "Validation Approach": (
        "Validation is performed through top-20 overlap with the 90-score teacher, old-vs-new top20 overlap, false-positive/false-negative movement, "
        "feature lift diagnostics, and MD5 fingerprinting of the final prediction vector."
    ),

    "Additional Notes (Optional)": (
        "Shortcoming: this method does not use true Amex hidden labels, so a 93+ leaderboard score cannot be guaranteed. "
        "If the 90-score teacher is already exactly reproduced by the original formula, there is no additional signal to learn from the teacher file alone. "
        "The next real improvement requires leaderboard testing of carefully controlled weight variants."
    )
}

final_framework_sheet = framework_template.copy()

for i in range(len(final_framework_sheet)):
    section = str(final_framework_sheet.loc[i, "Section"]).strip()
    final_framework_sheet.loc[i, "Response"] = framework_answers.get(
        section,
        "This section is addressed using the optimized deterministic P&L framework with weight-only tuning."
    )

assert final_framework_sheet["Response"].isna().sum() == 0


# 8. Export final Excel


TEAM_NAME = "Strategist"
timestamp = time.strftime("%H%M%S")

OUT_FILE = f"2026_File1_{TEAM_NAME}_OptimizedWeights_{score_md5[:8]}_{timestamp}.xlsx"
OUT_PATH = f"/content/{OUT_FILE}"

with pd.ExcelWriter(
    OUT_PATH,
    engine="xlsxwriter",
    engine_kwargs={"options": {"strings_to_urls": False}}
) as writer:

    final_prediction_sheet[["ID", "Prediction"]].to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    final_framework_sheet[["Section", "Response"]].to_excel(
        writer,
        sheet_name="Profitability Framework",
        index=False
    )

    workbook = writer.book

    header_fmt = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1,
        "align": "center",
        "valign": "vcenter"
    })

    text_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top",
        "border": 1
    })

    num_fmt = workbook.add_format({
        "num_format": "0.000000",
        "border": 1
    })

    pred_ws = writer.sheets["Predictions"]
    pred_ws.set_column("A:A", 18)
    pred_ws.set_column("B:B", 20, num_fmt)
    pred_ws.write(0, 0, "ID", header_fmt)
    pred_ws.write(0, 1, "Prediction", header_fmt)

    fw_ws = writer.sheets["Profitability Framework"]
    fw_ws.set_column("A:A", 34, text_fmt)
    fw_ws.set_column("B:B", 120, text_fmt)
    fw_ws.write(0, 0, "Section", header_fmt)
    fw_ws.write(0, 1, "Response", header_fmt)

    for r in range(1, len(final_framework_sheet) + 1):
        fw_ws.set_row(r, 95)

print("\n===== FINAL FILE SAVED =====")
print("Saved:", OUT_PATH)
print("File size MB:", os.path.getsize(OUT_PATH) / (1024 * 1024))
print("Final MD5:", score_md5)
print("Final teacher overlap:", best_overlap)

verify_pred = pd.read_excel(OUT_PATH, sheet_name="Predictions")
verify_fw = pd.read_excel(OUT_PATH, sheet_name="Profitability Framework")

assert verify_pred.columns.tolist() == ["ID", "Prediction"]
assert verify_fw.columns.tolist() == ["Section", "Response"]
assert len(verify_pred) == len(final_prediction_sheet)
assert verify_pred["Prediction"].isna().sum() == 0
assert verify_fw["Response"].isna().sum() == 0

print("Verification successful.")
print("Rows:", len(verify_pred))
print("Prediction min:", verify_pred["Prediction"].min())
print("Prediction max:", verify_pred["Prediction"].max())

time.sleep(1)
files.download(OUT_PATH)

print("If download does not appear, click this:")
display(FileLink(OUT_PATH))
