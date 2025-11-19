import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

FRAMES = [
    "sexual stigma and transmission routes",
    "racial disparities and stigmatising name",
    "global relations",
    "public health failure",
    "epidemic preparedness and surveillance",
    "human-interest stories",
    "broader health issues",
]

# Paths – adjust as needed
model = "NB"  # Change as needed: "NB", "BERT", "DeBERTa"
TRUE_PATH = "labelled_frames_test.csv"
RUN_PRED_PATH = f"all_run3/{model}_predictions.csv"
OG_RUN_PRED_PATH = f"{model}_test.csv"

# --- LOAD ---

df_true = pd.read_csv(TRUE_PATH)
pred1 = pd.read_csv(RUN_PRED_PATH)
pred2 = pd.read_csv(OG_RUN_PRED_PATH)

# Only keep id + frame columns in predictions
cols_keep = ["stories_id"] + FRAMES
pred1 = pred1[cols_keep].copy()
pred2 = pred2[cols_keep].copy()

# Rename frame cols to distinguish runs
pred1 = pred1.rename(columns={f: f"{f}_run" for f in FRAMES})
pred2 = pred2.rename(columns={f: f"{f}_OG_run" for f in FRAMES})

# Merge on stories_id
merged = (
    df_true[cols_keep]
    .merge(pred1, on="stories_id")
    .merge(pred2, on="stories_id")
)

print("Merged shape:", merged.shape)


# --- METRIC COMPARISON FUNCTION ---

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

rows = []

for frame in FRAMES:
    y_true = merged[frame].values
    y_pred_run = merged[f"{frame}_run"].values
    y_pred_OG_run = merged[f"{frame}_OG_run"].values

    m1 = compute_metrics(y_true, y_pred_run)
    m2 = compute_metrics(y_true, y_pred_OG_run)

    row = {"frame": frame}
    for metric in ["accuracy", "precision", "recall", "f1", "kappa"]:
        row[f"{metric}_run"] = m1[metric]
        row[f"{metric}_OG_run"] = m2[metric]
        row[f"{metric}_delta"] = m2[metric] - m1[metric]
    rows.append(row)

instance_compare_df = pd.DataFrame(rows).sort_values("frame")
print(instance_compare_df)

instance_compare_df.to_csv(f"{model}_run_comparison_instance_level.csv", index=False)
print(f"Saved instance-level comparison to {model}_run_comparison_instance_level.csv")