import numpy as np
import pandas as pd
from text_utils import load_data, combine_ngrams, timer # timer.records
from ngrams import ngram2dic, preprocess, train_language_models
from ngrams import predict_language
from uroman import Uroman
import pickle
import matplotlib.pyplot as plt
from collections import Counter

from ngrams import build_indexer_matrix, predict_language3_fast

# 1. LOAD DATA FROM TXT FILES
with timer('loading_data'):
    X_train = load_data('./WiLI-2018_dataset/x_train.txt')
    y_train = load_data('./WiLI-2018_dataset/y_train.txt')
    X_test = load_data('./WiLI-2018_dataset/x_test.txt')
    y_test = load_data('./WiLI-2018_dataset/y_test.txt')

# 2. ROMANIZING THE DATA ===================================================================================
with timer('loading_romanizer'):
    romanizer = Uroman()
# 2.1 ROMANIZING X_train ===================================================================================
with timer("creating X_train_romanized"):
    X_train_romanized = []
    for text in X_train:
        try:
            X_train_romanized.append(romanizer.romanize_string(text))
        except Exception as e:
            X_train_romanized.append(f"__ERROR__:{type(e).__name__}:{e}")
with timer("saving X_train_romanized"):
    with open("dataset_romanized/X_train_romanized.txt", "w") as fhand:
        for item in X_train_romanized:
            fhand.write(item + '\n')
# 2.2 ROMANIZING X_test ====================================================================================
with timer("creating X_test_romanized"):
    X_test_romanized = []
    for text in X_test:
        try:
            X_test_romanized.append(romanizer.romanize_string(text))
        except Exception as e:
            X_test_romanized.append(f"__ERROR__:{type(e).__name__}:{e}")
with timer("saving X_test_romanized"):
    with open("dataset_romanized/X_test_romanized.txt", "w") as fhand:
        for item in X_test_romanized:
            fhand.write(item + '\n')
# ==========================================================================================================
# TR: TRain -> columns: text_TR, target_language_TR, romanized_TR, text_ngrams_TR, romanized_ngrams_TR
# TE: TEst  -> columns: text_TE, target_language_TE, romanized_TE, text_ngrams_TE, romanized_ngrams_TE
# 3. CREATNG TRAINING DF ======================================================================================
with timer('loading_data'):
    X_train = load_data('./WiLI-2018_dataset/x_train.txt')
    y_train = load_data('./WiLI-2018_dataset/y_train.txt')
    X_train_romanized = load_data("dataset_romanized/X_train_romanized.txt", strip=False)
# retry creating df with romanized file
with timer('train_df_created'):
    df_train = pd.DataFrame({
        "text_TR": X_train,
        "target_language_TR": y_train,
        "romanized_TR": X_train_romanized})
with timer('train_text_ngrams'):
    df_train.insert(3, "text_ngrams_TR", df_train.text_TR.apply(lambda text: ngram2dic(preprocess(text), 1,5)))
with timer('train_romanized_ngrams'):
    df_train.insert(4, "romanized_ngrams_TR", df_train.romanized_TR.apply(lambda text: ngram2dic(preprocess(text), 1,5)))
# ==========================================================================================================
# 4. COMBINING NGRAMS FROM TRAINING DATA FOR TRAINING =========================================================        
with timer('text combine ngrams'):
    lang_text_all_ngrams = combine_ngrams(df_train, "text_ngrams_TR")
with timer('romanized combine ngrams'):    
    lang_romanized_all_ngrams = combine_ngrams(df_train, "romanized_ngrams_TR")
# ==========================================================================================================
# 5. TRAINING MODEL ===========================================================================================
# TODO: NEEDS TO BE REDESIGNED TO ALLOW FOR N=1 OR M% TO BE REMOVED
with timer("creating_models"):
    models_text = train_language_models(lang_text_all_ngrams, alpha=0.1)
    models_romanized = train_language_models(lang_romanized_all_ngrams, alpha=0.1)
# ==========================================================================================================
# 6. BUILDING SPARSE MATRIX WITH DELTA (delta = logp - unk)
with timer("calculating_unknown_scores_models_text"):
    unknown_scores = {
        lang: models_text[lang].unknown_log_prob
        for lang in models_text
    }

with timer('precalc cost'):
    langs = sorted(unknown_scores.keys())
    M, ngram_idx, lang_idx = build_indexer_matrix(models_text, langs)
    unk_arr = np.array([unknown_scores[l] for l in langs])

# TESTING
with timer('loading testing data'):
    X_test = load_data('./WiLI-2018_dataset/x_test.txt')
    y_test = load_data('./WiLI-2018_dataset/y_test.txt')
    X_test_romanized = load_data("dataset_romanized/X_test_romanized.txt", strip=False)

with timer('predictions using predict_language3_fast'):
    predictions = [
        predict_language3_fast(M, ngram_idx, langs, unk_arr, ngram2dic(preprocess(text), 1, 5))
        for text in X_test
    ]

predictions_max = [max(prediction, key=prediction.get) for prediction in predictions]
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_max, sample_weight=None))

# ROMANZIED TEST


# === PREIDCT_LANGUAGE3() + REMOVE COUNT 1 FROM TRAINING
# TODO: 1.during romanization, there are some observations that start with __ERROR__. These need to be removed
# TODO: 2. removing lowest count=1
# TODO: 3. removing a percentage of observations

