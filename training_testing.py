# TRAINING AND TESTING SEPARATELY
import pandas as pd
from text_utils import combine_ngrams
from ngrams import ngram2dic, string2ngram, preprocess, train_language_models, predict_language

# LOAD PICKLED DATA
df_train = pd.read_pickle("df_train.pkl")
df_test = pd.read_pickle("df_test.pkl")

# COMBINING NGRAMS FROM TRAINING DATA FOR TRAINING
lang_text_all_ngrams = {}
for lang in df_train.target_language.unique():
    lang_text_all_ngrams[lang] = combine_ngrams(df=df_train, lang=lang, col='text_ngrams')
lang_romanized_all_ngrams = {}
for lang in df_train.target_language.unique():
    lang_romanized_all_ngrams[lang] = combine_ngrams(df=df_train, lang=lang, col='romanized_ngrams')


# TRAIN MODELS
models_text = train_language_models(lang_text_all_ngrams, alpha=0.1)
models_romanized = train_language_models(lang_romanized_all_ngrams, alpha=0.1)


# TEST 1: Control Group: text_ngrams
M = len(df_test) # Mth first observation in the testing data
from time import perf_counter
start_t = perf_counter()
results_list = []
for i in range(M):
    text_ngrams = df_test.iloc[i].text_ngrams
    prediction = predict_language(models_text, text_ngrams, output='scores')
    label = df_test.iloc[i].target_language
    results_list.append(prediction)
    #results.append([prediction, label])
results_df = pd.DataFrame(results_list)
results_max = results_df.idxmax(axis=1)
results_df['MAX'] = results_max
results_df['LABEL'] = df_test.target_language.iloc[:M]
end_t = perf_counter()
print(end_t - start_t)
results_df.to_csv('text_results_df.csv') # use this dataframe for analysis

# TEST 2: Test Group: romanized_ngrams
M = len(df_test) # Mth first observation in the testing data
from time import perf_counter
start_t = perf_counter()
results_list = []
for i in range(M):
    text_ngrams = df_test.iloc[i].romanized_ngrams
    prediction = predict_language(models_text, text_ngrams, output='scores')
    label = df_test.iloc[i].target_language
    results_list.append(prediction)
    #results.append([prediction, label])
results_df = pd.DataFrame(results_list)
results_max = results_df.idxmax(axis=1)
results_df['MAX'] = results_max
results_df['LABEL'] = df_test.target_language.iloc[:M]
end_t = perf_counter()
print(end_t - start_t)
results_df.to_csv('romanized_results_df.csv') # use this dataframe for analysis

predictions = results_df.MAX
labels = results_df.LABEL
from sklearn.metrics import accuracy_score
accuracy_score(labels, predictions, sample_weight=None) #0.8796

