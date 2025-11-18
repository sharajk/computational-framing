import pandas as pd
from classifiers import *
import accelerate
import gc
import torch
import os
import argparse
import pickle

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

def make_stats_and_preds(X_test, X_val, y_test, clf):
    """
    Given test and validation data, along with a classifier, return predictions and performance metrics for the classifier on the provided data.

    Inputs:
    X_test: test data; dataframe
    X_val: validation data; dataframe
    y_test: ground truth frames for X_test.
    clf: classifier object.

    Returns:
    dict with performance metrics for the test data and labels for the validation data.
    """
    test_stats = clf.accuracy_pr_kappa(X_test, y_test)
    val_binary = clf.predict(X_val)
    val_continuous = clf.frac_predict(X_val)
    test_binary = clf.predict(X_test)
    test_continuous = clf.frac_predict(X_test)

    return {
        'test_stats': test_stats,
        'val_binary': val_binary,
        'val_continuous': val_continuous,
        'test_binary': test_binary,
        'test_continuous': test_continuous,
    }

# train NB regardless.
clf_nb = MultiFrameClassifier(frames, clf_type='NB')
clf_nb.fit(X_train_NB, y_train)
nb_stats_and_preds = make_stats_and_preds(X_test_NB, X_val_NB, y_test, clf_nb)

# train BERT unless we skip it
clf_bert = MultiFrameClassifier(frames, clf_type='BERT', run_save_dir=SAVE_DIR, device='cuda')
if ONLY_TRAIN_DEBERTA:
    for frame, clf in clf_bert.frame_classifiers.items():
        clf.load_best()
else:
    clf_bert.fit(X_train_BERT, y_train, X_val_BERT, y_val)
bert_stats_and_preds = make_stats_and_preds(X_test_BERT, X_val_BERT, y_test, clf_bert)

# train DeBERTa
clf_deberta = MultiFrameClassifier(frames, clf_type='DeBERTa', run_save_dir=SAVE_DIR, device='cuda')
clf_deberta.fit(X_train_BERT, y_train, X_val_BERT, y_val)
deberta_stats_and_preds = make_stats_and_preds(X_test_BERT, X_val_BERT, y_test, clf_deberta)

# pickle stats & predictions
for name, res in [('NB', nb_stats_and_preds), ('BERT', bert_stats_and_preds), ('DeBERTa', deberta_stats_and_preds)]:
    pickle.dump(res, open('{}/{}_stats_and_preds.pkl'.format(SAVE_DIR, name), 'wb'))