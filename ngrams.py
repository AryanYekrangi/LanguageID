import numpy as np
import math
from collections import Counter
import re
import regex
from scipy.sparse import csr_matrix

def preprocess(string):
    string = regex.sub(r'[\p{P}\p{S}\s]+', '_', string)
    string = re.sub(r'_+', '_', string)
    return string


def string2ngram(string, n):
    string = '\x02' * max(n-1, 1) + string + '\x03' * max(n-1, 1)
    return [string[i:i+n] for i in range(len(string)-n+1)]


def ngram2dic(text, min_n, max_n):
    dic = {}
    for n in range(min_n, max_n + 1):
        for ng in string2ngram(text, n):
            if ng in dic:
                dic[ng] += 1
            else:
                dic[ng] = 1
    return dic


class NgramLanguageModel:
    def __init__(self, ngram_counts, alpha=0.1):
        total = sum(ngram_counts.values())
        vocab_size = len(ngram_counts)
        denom = total + alpha * vocab_size
        self.unknown_log_prob = math.log(alpha / denom)
        self.log_probs = {
            ng: math.log((count + alpha) / denom)
            for ng, count in ngram_counts.items()
        }


    def log_probability(self, ngrams):
        get = self.log_probs.get
        unk = self.unknown_log_prob
        logp = 0.0
        total = sum(ngrams.values()) # added later
        for ng, count in ngrams.items():
            logp += count * get(ng, unk)
        return logp / total # originally return logp / len(ngrams)


def train_language_models(ngram_data, alpha=0.1):
    models = {}
    for lang, counts in ngram_data.items():
        models[lang] = NgramLanguageModel(counts, alpha=alpha)
    return models


def predict_language(models:dict, ngrams:dict, output: str):
    """
    Returns the langauge of a given text.

    Parameters
    ----------
    models : dict
        The models are generated using the train_language_models() function.
    text : str
        The string to be analysed.
    output : str
        'all' for all languages ranked
        
    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    scores = {
        lang: model.log_probability(ngrams) for lang, model in models.items()
    }
    if output == 'max':
        return max(scores, key=scores.get)
    elif output == 'scores':
        return scores
    elif output == 'scores-and-max':
        return (scores, max(scores, key=scores.get))
    return scores

def build_indexer_matrix(models, langs):
    lang_idx = {lang: i for i, lang in enumerate(langs)}
    ngram_idx = {}
    rows, cols, data = [], [], []

    for lang, model in models.items():
        li = lang_idx[lang]
        unk = model.unknown_log_prob
        for ng, val in model.log_probs.items():
            ni = ngram_idx.setdefault(ng, len(ngram_idx))
            rows.append(ni)
            cols.append(li)
            data.append(val - unk)

    M = csr_matrix((data, (rows, cols)), shape=(len(ngram_idx), len(langs)))
    return M, ngram_idx, lang_idx


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