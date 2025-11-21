import pickle
import pprint
import pandas as pd


SAVE_DIR = "all_run3"  

files = ["NB", "BERT", "DeBERTa"]

for name in files:
    print("\n\n========================", name, "========================")
    try:
        data = pickle.load(open(f"{SAVE_DIR}/{name}_stats_and_preds.pkl", "rb"))
        print("Test Stats:\n", data["test_stats"])
        print("\nValidation Stats:", data.get("val_binary"))
    except FileNotFoundError:
        print("File not found")


df_frames_test = pd.read_csv('labelled_frames_test.csv')

frames = df_frames_test.columns[-7:]

cols_for_x = df_frames_test.columns[:-7]
X_test = df_frames_test[cols_for_x]


def make_csv(model_name, pickle_file, X_test):
    print(f"Processing {model_name}...")
    data = pickle.load(open(pickle_file, "rb"))
    preds = data["test_binary"]
    df = X_test.copy()
    for frame in frames:
        df[frame] = preds[frame]

    out_path = f"{SAVE_DIR}/{model_name}_predictions.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

make_csv("NB", f"{SAVE_DIR}/NB_stats_and_preds.pkl", X_test)
make_csv("BERT", f"{SAVE_DIR}/BERT_stats_and_preds.pkl", X_test)
make_csv("DeBERTa", f"{SAVE_DIR}/DeBERTa_stats_and_preds.pkl", X_test)
