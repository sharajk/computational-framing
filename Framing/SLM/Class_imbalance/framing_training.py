import pandas as pd
from classifiers import *
import accelerate
import gc
import torch
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, average_precision_score

# script to train framing models and output labels on the test set.
parser = argparse.ArgumentParser(
                    prog='FramingTraining',
                    description='Trains Framing Classifiers and Outputs Labels',
                    epilog='good luck')
parser.add_argument('-n', '--n_train', help='number of training examples to use')
parser.add_argument('-d', '--dir_for_files', help='directory for files saved from this run')
parser.add_argument('-b', '--bert_skip', help='skip BERT training and only train DEBERTa', action="store_true")
parser.add_argument('-o', '--overlap_train_val', help='overlap training and validation sets', action="store_true")

args = parser.parse_args()
N_TRAIN = int(args.n_train)
OVERLAP_TRAIN_VAL = args.overlap_train_val
SAVE_DIR = args.dir_for_files
if args.bert_skip:
    ONLY_TRAIN_DEBERTA = True
else:
    ONLY_TRAIN_DEBERTA = False

try:
    os.mkdir(SAVE_DIR)
except FileExistsError as e:
    print(e)

df_frames_test = pd.read_csv('labelled_frames_test.csv')
df_frames_train = pd.read_csv('labelled_frames_train.csv')

gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.7"

frames = df_frames_test.columns[-7:]
print(df_frames_test.columns)
print(frames)

cols_for_x = ['title', 'text']
X_test_NB = df_frames_test['text']
X_test_BERT = df_frames_test[cols_for_x]
y_test = df_frames_test[frames]

# don't overlap train and val sets
if N_TRAIN < len(df_frames_train) and not(OVERLAP_TRAIN_VAL):
    X_train_BERT, X_val_BERT, y_train, y_val = train_test_split(
        df_frames_train[cols_for_x], 
        df_frames_train[frames], 
        test_size=1.0 - (N_TRAIN / len(df_frames_train)),
        random_state=5,
    )
    X_train_NB = X_train_BERT['text']
    X_val_NB = X_val_BERT['text']
elif N_TRAIN == len(df_frames_train):
    # completely overlap train and val sets; use full train set for training & for validation.
    X_train_BERT = df_frames_train[cols_for_x]
    X_val_BERT = df_frames_train[cols_for_x]
    y_train = df_frames_train[frames]
    y_val = df_frames_train[frames]
    X_train_NB = df_frames_train['text']
    X_val_NB = df_frames_train['text']
else:
    # partial random overlap of train and val sets
    X_train_BERT, _, y_train, _ = train_test_split(
        df_frames_train[cols_for_x], 
        df_frames_train[frames], 
        test_size=1.0 - (N_TRAIN / len(df_frames_train)),
        random_state=5,
    )
    X_val_BERT, _, y_val, _ = train_test_split(
        df_frames_train[cols_for_x], 
        df_frames_train[frames], 
        test_size=1.0 - (N_TRAIN / len(df_frames_train)),
        random_state=42,
    )
    X_train_NB = X_train_BERT['text']
    X_val_NB = X_val_BERT['text']

def make_stats_and_preds(X_test, X_val, y_test, clf, frames, thresholds=None):
    """
    Given test and validation data, along with a classifier, return predictions 
    and performance metrics for the classifier on the provided data.

    Inputs:
    X_test: test data (DataFrame or Series, depending on clf)
    X_val: validation data
    y_test: ground truth frames for X_test (DataFrame with one col per frame)
    clf: classifier object (MultiFrameClassifier)
    frames: list of frame names
    thresholds: optional dict[frame -> threshold]. If None, use clf.predict().

    Returns:
    dict with performance metrics for the test data and labels for the validation data.
    """
    # Always keep continuous scores (probs)
    val_continuous  = clf.frac_predict(X_val)   # dict[frame -> probs]
    test_continuous = clf.frac_predict(X_test)  # dict[frame -> probs]

    if thresholds is None:
        # Old behaviour: rely on classifier defaults
        val_binary  = clf.predict(X_val)
        test_binary = clf.predict(X_test)

        # and use the existing accuracy_pr_kappa if you like
        test_stats = clf.accuracy_pr_kappa(X_test, y_test)

    else:
        # Use tuned thresholds per frame
        val_binary  = {
            f: (val_continuous[f]  >= thresholds[f]).astype(int) for f in frames
        }
        test_binary = {
            f: (test_continuous[f] >= thresholds[f]).astype(int) for f in frames
        }

        # Compute metrics manually so they're aligned with the thresholds
        test_stats = {}
        for f in frames:
            y_true = y_test[f].values.astype(int)
            y_pred = test_binary[f]

            acc   = accuracy_score(y_true, y_pred)
            prec  = precision_score(y_true, y_pred, zero_division=0)
            rec   = recall_score(y_true, y_pred, zero_division=0)
            f1    = f1_score(y_true, y_pred, zero_division=0)
            kappa = cohen_kappa_score(y_true, y_pred)

            test_stats[f] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "kappa": kappa,
                "threshold": float(thresholds[f]),
            }

    return {
        "test_stats": test_stats,
        "val_binary": val_binary,
        "val_continuous": val_continuous,
        "test_binary": test_binary,
        "test_continuous": test_continuous,
    }


def eval_with_thresholds(test_probs, y_test, thresholds, frames):
    metrics = {}
    for frame in frames:
        thresholds_frame = thresholds[frame]
        probs = test_probs[frame]
        y_true = y_test[frame].values.astype(int)
        y_pred = (probs >= thresholds_frame).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)

        metrics[frame] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "kappa": kappa,
            "threshold": thresholds_frame,
        }
    return metrics

def tune_thresholds_per_frame(clf, X_val, y_val, frames, primary="f1"):
    thresholds = {}
    for frame in frames:
        probs = clf.frame_classifiers[frame].frac_predict(X_val)
        y_true = y_val[frame].values.astype(int)

        # If val labels are trivial, use 0.5
        pos_count = y_true.sum()
        n = len(y_true)
        if pos_count == 0 or pos_count == n:
            thresholds[frame] = 0.5
            continue

        # use quantiles of the prob distribution rather than a fixed grid
        # this focuses search where the model actually places mass
        q_grid = np.linspace(0.05, 0.95, 19)
        cand_thresholds = np.unique(np.quantile(probs, q_grid))

        best_t, best = 0.5, -1.0
        nondeg_found = False

        for t in cand_thresholds:
            preds = (probs >= t).astype(int)

            if preds.min() != preds.max():
                nondeg_found = True

            if primary == "f1":
                score = f1_score(y_true, preds, average="binary", zero_division=0)
            else:
                score = average_precision_score(y_true, probs)

            if score > best:
                best, best_t = score, t

        if not nondeg_found:
            # model is effectively constant on val; choose a prevalence-matching fallback
            pos_rate = y_true.mean()
            best_t = float(np.quantile(probs, 1 - pos_rate))

        print(f"\n[Frame: {frame}]")
        print("  y_val positives:", pos_count, "out of", n)
        print("  probs min/max/mean:", probs.min(), probs.max(), probs.mean())
        print("  chosen threshold:", best_t, "(fallback nondeg:", not nondeg_found, ")")

        thresholds[frame] = float(best_t)

    return thresholds

nb_stats_and_preds = None
bert_stats_and_preds = None
deberta_stats_and_preds = None

# ========== 1) Train NB (bag-of-words) ==========

clf_nb = MultiFrameClassifier(frames, clf_type='NB')
clf_nb.fit(X_train_NB, y_train)
nb_thresholds = tune_thresholds_per_frame(clf_nb, X_val_NB, y_val, frames)
nb_stats_and_preds = make_stats_and_preds(
    X_test_NB, X_val_NB, y_test, clf_nb, frames, thresholds=nb_thresholds
)

gc.collect()
torch.cuda.empty_cache()

# ========== 2) Train BERT ==========
if not ONLY_TRAIN_DEBERTA:
    clf_bert = MultiFrameClassifier(frames, clf_type='BERT', run_save_dir=SAVE_DIR, device='cpu')
    clf_bert.fit(X_train_BERT, y_train, X_val_BERT, y_val)
    bert_thresholds = tune_thresholds_per_frame(clf_bert, X_val_BERT, y_val, frames)
    bert_stats_and_preds = make_stats_and_preds(
        X_test_BERT, X_val_BERT, y_test, clf_bert, frames, thresholds=bert_thresholds
    )
else:
    clf_bert = None
    bert_stats_and_preds = None

gc.collect()
torch.cuda.empty_cache()

# ========== 3) Train DeBERTa ==========
clf_deberta = MultiFrameClassifier(frames, clf_type='DeBERTa', run_save_dir=SAVE_DIR, device='cpu')
clf_deberta.fit(X_train_BERT, y_train, X_val_BERT, y_val)

deberta_thresholds = tune_thresholds_per_frame(clf_deberta, X_val_BERT, y_val, frames)

deberta_stats_and_preds = make_stats_and_preds(
    X_test_BERT, X_val_BERT, y_test, clf_deberta, frames, thresholds=deberta_thresholds
)

# ========== 4) Save everything ==========
for name, res in [
    ("NB", nb_stats_and_preds),
    ("BERT", bert_stats_and_preds),
    ("DeBERTa", deberta_stats_and_preds),
]:
    if res is not None:
        pickle.dump(
            res,
            open(f"{SAVE_DIR}/{name}_stats_and_preds.pkl", "wb"),
        )
