import pandas as pd
from text_utils import load_data, combine_ngrams
from ngrams import ngram2dic, string2ngram, preprocess, train_language_models, predict_language
from uroman import Uroman
from time import perf_counter
from contextlib import contextmanager
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

name_time_df = []
@contextmanager
def timer(name):
    start = perf_counter()
    yield
    end = perf_counter()
    time = end - start
    print(f"{name}: {time:.4f} s")
    name_time_df.append({"name": name,
                         "time": round(time, 2)})

# LOAD DATA FROM TXT FILES
with timer('loading_data'):
    X_train = load_data('./WiLI-2018_dataset/x_train.txt')
    y_train = load_data('./WiLI-2018_dataset/y_train.txt')
    X_test = load_data('./WiLI-2018_dataset/x_test.txt')
    y_test = load_data('./WiLI-2018_dataset/y_test.txt')

# ROMANIZING DATA ==========================================================================================
with timer('loading_romanizer'):
    romanizer = Uroman()

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
# ==========================================================================================================
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
# CREATNG TRAINING DF ======================================================================================
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
# COMBINING NGRAMS FROM TRAINING DATA FOR TRAINING =========================================================
# === combining new try
with timer('text combine ngrams'):
    lang_text_all_ngrams = {}
    for lang, group in df_train.groupby("target_language_TR"):
        total = Counter()
        for d in group["text_ngrams_TR"]:
            total.update(d)
        lang_text_all_ngrams[lang] = total

with timer('romanized combine ngrams'):
    lang_romanized_all_ngrams = {}
    for lang, group in df_train.groupby("target_language_TR"):
        total = Counter()
        for d in group["romanized_ngrams_TR"]:
            total.update(d)
        lang_romanized_all_ngrams[lang] = total

# ==========================================================================================================
# TRAINING MODEL ===========================================================================================
with timer("creating_models"):
    models_text = train_language_models(lang_text_all_ngrams, alpha=0.1)
    models_romanized = train_language_models(lang_romanized_all_ngrams, alpha=0.1)
# ==========================================================================================================


def predict_language3(indexer_delta, unknown_scores, ngrams, output='scores'):
    """indexer delta is required, which is generated using val - models_text[lang].unknown_log_prob"""
    N = sum(ngrams.values())
    scores = {
        lang: N * unk
        for lang, unk in unknown_scores.items()
        }
    for ng, count in ngrams.items():
        for lang, delta in indexer_delta.get(ng, []):
            scores[lang] += count * delta
    # NORMALIZING BY LENGTH
    for lang in scores:
        scores[lang] /= N
    return scores

with timer('creating_indexer_delta'):
    indexer_delta = {}
    for lang in models_text:
        for key, val in models_text[lang].log_probs.items():
            if key in indexer_delta:
                indexer_delta[key].append((lang, val - models_text[lang].unknown_log_prob))
            else:
                indexer_delta[key] = [(lang, val - models_text[lang].unknown_log_prob)]


with timer("calculating_unkonwn_scores_models_text"):
    unknown_scores = {
        lang: models_text[lang].unknown_log_prob
        for lang in models_text
    }

# SAVE INDEXER_DELTA AND UNKNOWN_SCORES
with timer("saving indexer_delta.pkl"):
    with open('indexer_delta.pkl', "wb") as file:
        pickle.dump(indexer_delta, file)
with timer("saving unknown_scores.pkl"):
    with open('unknown_scores.pkl', "wb") as file:
        pickle.dump(unknown_scores, file)


# LOAD THE INDEXER_DELTA AND UNKNWON_SCORES
with timer('loading indexer_delta.pkl'):
    with open('indexer_delta.pkl', "rb") as file:
        indexer_delta = pickle.load(file)
with timer('loading unknown_scores.pkl'):
    with open('unknown_scores.pkl', "rb") as file:
        unknown_scores = pickle.load(file)


# TODO: CREATE TESTING DATA FROM BOTH .TXT FILES
with timer('loading testing data'):
    X_test = load_data('./WiLI-2018_dataset/x_test.txt')
    y_test = load_data('./WiLI-2018_dataset/y_test.txt')
    X_test_romanized = load_data("dataset_romanized/X_test_romanized.txt", strip=False)


# remember no need for test df, just convert to ngrams and test
with timer('text_ngrams_list'):
    predictions = [predict_language3(indexer_delta, unknown_scores, ngram2dic(preprocess(text), 1,5)) for text in X_test[:1000]]

with timer('the rest'):
    predictions_max = [max(prediction, key=prediction.get) for prediction in predictions]
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, predictions_max, sample_weight=None)


# new predict langauge function
from collections import defaultdict

def predict_language4(indexer_delta, unknown_scores, ngrams):
    N = sum(ngrams.values())
    accum = defaultdict(float)
    for ng, count in ngrams.items():
        pairs = indexer_delta.get(ng)
        if pairs:
            for lang, delta in pairs:
                accum[lang] += count * delta
    return {lang: unk + accum[lang] / N for lang, unk in unknown_scores.items()}


with timer('text_ngrams_list'):
    predictions = [predict_language4(indexer_delta, unknown_scores, ngram2dic(preprocess(text), 1,5)) for text in X_test[:1000]]


def predict_language5(indexer_delta, unknown_scores, ngrams):
    N = sum(ngrams.values())
    accum = defaultdict(float)
    for ng in ngrams.keys() & indexer_delta.keys():
        count = ngrams[ng]
        for lang, delta in indexer_delta[ng]:
            accum[lang] += count * delta
    return {lang: unk + accum[lang] / N for lang, unk in unknown_scores.items()}

with timer('text_ngrams_list'):
    predictions = [predict_language5(indexer_delta, unknown_scores, ngram2dic(preprocess(text), 1,5)) for text in X_test[:1000]]


# claude
import numpy as np
from scipy.sparse import csr_matrix

def build_indexer_matrix(indexer_delta, langs):
    lang_idx = {lang: i for i, lang in enumerate(langs)}
    ngram_list = list(indexer_delta.keys())
    ngram_idx = {ng: i for i, ng in enumerate(ngram_list)}

    rows, cols, data = [], [], []
    for ng, pairs in indexer_delta.items():
        r = ngram_idx[ng]
        for lang, delta in pairs:
            rows.append(r)
            cols.append(lang_idx[lang])
            data.append(delta)

    M = csr_matrix((data, (rows, cols)), shape=(len(ngram_list), len(langs)))
    return M, ngram_idx, lang_idx

with timer('precalc cost'):
    langs = sorted(unknown_scores.keys())
    M, ngram_idx, lang_idx = build_indexer_matrix(indexer_delta, langs)
    unk_arr = np.array([unknown_scores[l] for l in langs])


def predict_language3_fast(M, ngram_idx, langs, unk_arr, ngrams):
    N = sum(ngrams.values())

    rows, data = [], []
    for ng, count in ngrams.items():
        i = ngram_idx.get(ng)
        if i is not None:
            rows.append(i)
            data.append(count)

    v = csr_matrix((data, ([0] * len(rows), rows)), shape=(1, M.shape[0]))
    accum = v @ M                      # 1 x num_langs sparse result
    scores = unk_arr + np.asarray(accum.todense()).ravel() / N
    return dict(zip(langs, scores))

with timer('fast_version'):
    predictions = [
        predict_language3_fast(M, ngram_idx, langs, unk_arr, ngram2dic(preprocess(text), 1, 5))
        for text in X_test
    ]

predictions_max = [max(prediction, key=prediction.get) for prediction in predictions]
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions_max, sample_weight=None)


with timer('fast_version_romanized'):
    predictions_r = [
        predict_language3_fast(M, ngram_idx, langs, unk_arr, ngram2dic(preprocess(text), 1, 5))
        for text in X_test_romanized
    ]

predictions_max_r = [max(prediction, key=prediction.get) for prediction in predictions_r]
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions_max_r, sample_weight=None)



# ROMANZIED TEST
with timer('creating_indexer_delta_r'):
    indexer_delta_r = {}
    for lang in models_romanized:
        for key, val in models_romanized[lang].log_probs.items():
            if key in indexer_delta_r:
                indexer_delta_r[key].append((lang, val - models_romanized[lang].unknown_log_prob))
            else:
                indexer_delta_r[key] = [(lang, val - models_romanized[lang].unknown_log_prob)]


with timer("calculating_unkonwn_scores_models_text_r"):
    unknown_scores_r = {
        lang: models_romanized[lang].unknown_log_prob
        for lang in models_romanized
    }


with timer('precalc cost_r'):
    langs = sorted(unknown_scores_r.keys())
    M, ngram_idx, lang_idx = build_indexer_matrix(indexer_delta_r, langs)
    unk_arr = np.array([unknown_scores_r[l] for l in langs])
    
with timer('fast_version_r'):
    predictions_r = [
        predict_language3_fast(M, ngram_idx, langs, unk_arr, ngram2dic(preprocess(text), 1, 5))
        for text in X_test_romanized
    ]
    
predictions_max_r = [max(prediction, key=prediction.get) for prediction in predictions_r]
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions_max_r, sample_weight=None)

# === PREIDCT_LANGUAGE3() + REMOVE COUNT 1 FROM TRAINING
# TODO: this is a bad implementation, so it changes the object directl (log_probs), need ot create a copy first
langs = []
total_ngrams = []
removed_ngrams = []
def remove_min(lang, obj):
    total_ngrams.append(len(obj.log_probs))
    lowest = min(obj.log_probs.values())
    n_removed = sum(
        lp == lowest
        for lp in obj.log_probs.values()
    )
    removed_ngrams.append(n_removed)
    obj.log_probs = {
        ng: lp
        for ng, lp in obj.log_probs.items()
        if not math.isclose(lp, lowest)
    }
    langs.append(lang)
    return obj

adjusted_models_text = {}
for lang, model in models_text.items():
    new_mod = remove_min(lang, model)
    adjusted_models_text[lang] = new_mod


# note: during romanization, there are some observations that start with __ERROR__. These need to be removed
# try removing a percentage of observations


with open("models_text_logp.pkl", "rb") as fhand:
    models_text = pickle.load(fhand)
    
with open("models_text.pkl", "rb") as fhand:
    models_text_count = pickle.load(fhand)
    
    
all_lens = []
for model in models_text:
    all_lens.append((model,(len(models_text[model].log_probs))))
sorted_all_lens = sorted(all_lens, key=lambda x: x[1], reverse=False)
lan, height = zip(*sorted_all_lens)

plt.figure(figsize=(52, 4))
plt.bar(x=lan, height=height, width=0.5)
plt.show()
