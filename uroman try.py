import numpy as np
import pandas as pd
import math
from collections import Counter
import pickle
#from Ngrams import preprocess, string2ngram, ngram2dic
from uroman import Uroman
from Ngrams import preprocess, string2ngram, ngram2dic

# def load_data(filename):
#     """function for quickly loading the data"""
#     data_list = []
#     with open(filename, encoding='utf-8') as fhand:
#         for line in fhand:
#             data_list.append(line.strip())
#     return data_list

# X_train = load_data('./WiLI-2018_dataset/x_train.txt')
# y_train = load_data('./WiLI-2018_dataset/y_train.txt')
# X_test = load_data('./WiLI-2018_dataset/x_test.txt')
# y_test = load_data('./WiLI-2018_dataset/y_test.txt')


# # SHOULD PREPROCESS THE DATA HERE

# # TURN TRAINING DATA TO PANDAS DATAFRAME
# df_train = pd.DataFrame()
# df_train.insert(0, 'text', X_train)
# df_train.insert(1, 'target_language', y_train)

romanizer = Uroman()
df_train.insert(2, 'romanized', df_train.text.apply(lambda text: romanizer.romanize_string(text)))


# save to csv file
df_train.to_csv('data_and_romanized.csv')



# TESTING
df_test = pd.DataFrame()
df_test.insert(0, 'text', X_test)
df_test.insert(1, 'target_language', y_test)

romanizer = Uroman()
df_test.insert(2, 'romanized', df_test.text.apply(lambda text: romanizer.romanize_string(text)))


# save to csv file
df_test.to_csv('testing_romanized.csv')



# how many unique characters are used with this conversion?


def count_chars(string):
    all_chars = {}
    for char in string:
        if char in all_chars:
            all_chars[char] += 1
        else:
            all_chars[char] = 1
    return Counter(all_chars)

count_chars(df_train.romanized.iloc[0,])

dics_combined = Counter()
for i in df_train.romanized:
    dics_combined += count_chars(i)
dics_combined_df = pd.DataFrame(dics_combined.items(),  columns=['character', 'frequency'])
dics_combined_df.to_csv('dics_combined_df.csv')



###############################################################################
# LOAD DATA FROM .CSV
df_imported = pd.read_csv('data_and_romanized.csv')
df_imported.insert(4, 'ngrams', df_imported.romanized.apply(lambda text: 
                                     ngram2dic(string2ngram(preprocess(text),4))))

# COPIED FROM OTHER PYTHON FILES. SHOULD CONSIDER PACKAGING THESE
def combine_dic_pickle(lang):
    """combines dictionaries and dumps them as pkl"""
    lang_df = df_imported[df_imported.target_language==lang]
    total = Counter(dict())
    for i in range(len(lang_df)):
        total += Counter(lang_df.ngrams.values[i])
    with open(f'./quadgrams/{lang}.pkl', 'wb') as f:
        pickle.dump(total, f)

# =============================================================================
# -----     PICKLING THE DATA (ONLY DO ON FIRST TRY)     -----
# =============================================================================
for lang in df_imported.target_language.unique():
    combine_dic_pickle(lang)
print("-----FINISHED PICKLING-----")

# =============================================================================

ngram_data = {}
from os import listdir
all_files = listdir('quadgrams')
for file in all_files:
    language = file[:-4]
    with open(f'./quadgrams/{file}', 'rb') as f:
        ngram_data[language] = pickle.load(f)
print("-----FINISHED LOADING THE PICKLED DATA-----")

class TrigramLanguageModel:
    def __init__(self, trigram_counts, alpha=0.1):  
        self.alpha = alpha
        self.trigram_counts = trigram_counts
        self.total = sum(trigram_counts.values())
        self.vocab_size = len(trigram_counts)

    def log_probability(self, trigrams):
        """
        trigrams: iterable of trigram strings
        """
        logp = 0.0
        denom = self.total + self.alpha * self.vocab_size
        for tri in trigrams:
            count = self.trigram_counts.get(tri, 0)
            prob = (count + self.alpha) / denom
            logp += math.log(prob)
        return logp


def train_language_models(trigram_data, alpha=0.1):
    models = {}
    for lang, counts in trigram_data.items():
        models[lang] = TrigramLanguageModel(counts, alpha=alpha)
    return models

models = train_language_models(ngram_data, alpha=0.1)


# -----     PREDICTION      -----

# TURNING TESTING DATA TO PANDAS DATAFRAME
df_test = pd.DataFrame()
df_test.insert(0, 'text', X_test)
df_test.insert(1, 'target_language', y_test)
df_test.insert(2, 'ngrams', df_test.text.apply(lambda text: 
                                     ngram2dic(string2ngram(preprocess(text),4))))

def predict_language(models, text):
    #trigrams = extract_trigrams(normalize(text))
    trigrams = text
    scores = {
        lang: model.log_probability(trigrams)
        for lang, model in models.items()
    }
    return max(scores, key=scores.get), scores
    # .Can also return all scores to compare what languages are similar

#     -----     EXPERIMENT     -----
# THIS IS VERY SLOW
results = []
#for i in range(len(df_train)):
for i in range(1000):
    text_trigrams = df_test.iloc[i].ngrams
    prediction = predict_language(models, text_trigrams)
    label = df_test.iloc[i].target_language
    results.append([prediction, label])


# FASTER WAY?
#predictions = df_test.ngrams.apply(lambda ngrams: predict_language(models, ngrams))

predictions = [result[0][0] for result in results]
labels = [result[1] for result in results]
results_df = pd.DataFrame()
results_df.insert(0, 'prediction', predictions)
results_df.insert(1, 'y', labels)
results_df.to_csv('results_quadgram.csv', index=False)

from sklearn.metrics import accuracy_score
accuracy_score(labels, predictions, sample_weight=None)


from sklearn.metrics import recall_score
recall_score(labels, predictions, labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn')


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(labels, predictions, labels=np.unique(np.array(y_test)))
matrix.diagonal()/matrix.sum(axis=1)








