import numpy as np
import math
from collections import Counter

# consider preprocessing empty space \n space tab etc.
#Arabic "٪"

def preprocess(string):
    punctuations = """ \n\t!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，“”:《》（）٪0123456789"""
    for i in range(len(string)):
        if string[i] in punctuations:
            string = string.replace(string[i], '_')
    #string = string.replace('_', '')
    return string

def string2ngram(string, n):
    ngram_array = []
    for i in range(len(string)):
        if i <  len(string) - (n-1):
            current_ngram = []
            for j in range(n):
                current_ngram.append(string[i+j])
            ngram_array.append(current_ngram)
    return ngram_array

def ngram2dic(array:list):
    dic = {}
    for item in array:
        key = ''.join(item)
        if key in dic:
            dic[key] += 1
        else:
            dic[key] = 1
    # sorting dic
    sorted_dic = dict(sorted(dic.items(), key=lambda x:x[1], reverse=True))
    return sorted_dic

# NEED TO EDIT THIS
def trim_ngram_dic(ngram, num):
    """return the n-most common n-grams
    trim_ngram(ngram, 5) returns the 5 most frequent n-grams
    ngram must be a dictionary of dictionaries
    """
    counter = num
    current_lang_dic = {}
    for key, value in ngram.items():
        if(counter>0):
            current_lang_dic[key] = value
            counter -= 1
    return current_lang_dic

def trim_ngram_array(ngram, num):
    """returns the most common ngrams without their frequency"""
    counter = num
    ngram_list = []
    for key, value in ngram.items():
        if(counter>0):
            ngram_list.append(key)
            counter -= 1
    return np.array(ngram_list)

"""
with open('2-gram.pkl', 'rb') as f:NgramLanguageModel
    bigram = pickle.load(f)
"""

class NgramLanguageModel:
    def __init__(self, ngram_counts, alpha=0.1):  
        self.alpha = alpha
        self.ngram_counts = ngram_counts
        self.total = sum(ngram_counts.values())
        self.vocab_size = len(ngram_counts)

    def log_probability(self, ngrams):
        """
        ngrams: iterable of ngram strings
        """
        logp = 0.0
        denom = self.total + self.alpha * self.vocab_size
        for n in ngrams:
            count = self.ngram_counts.get(n, 0)
            prob = (count + self.alpha) / denom
            logp += math.log(prob)
        return logp


def train_language_models(ngram_data, alpha=0.1):
    models = {}
    for lang, counts in ngram_data.items():
        models[lang] = NgramLanguageModel(counts, alpha=alpha)
    return models


def predict_language(models:dict, text:str, output: str):
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
    
    #trigrams = extract_trigrams(normalize(text))
    ngrams = text
    scores = {
        lang: model.log_probability(ngrams)
        for lang, model in models.items()
    }
    return max(scores, key=scores.get), scores if output=='all' else max(scores, key=scores.get) # this may be buggy
    # .Can also return all scores to compare what languages are similar

if __name__ == "__main__":
    models = train_language_models(ngram_data, alpha=0.1)

