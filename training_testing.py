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


# TEST
# THIS IS VERY SLOW
from time import perf_counter
start_t = perf_counter()
results = []
#for i in range(len(df_train)):
for i in range(1000):
    text_ngrams = df_test.iloc[i].ngrams
    prediction = predict_language(models_text, text_ngrams, output='')
    label = df_test.iloc[i].target_language
    results.append([prediction, label])
end_t = perf_counter()
print(end_t - start_t)

start_t = perf_counter()
pred2 = df_test.iloc[:1000].ngrams.apply(lambda dic: predict_language(models_text, dic, output=''))
end_t = perf_counter()
print(end_t - start_t)

predictions = [result[0][0] for result in results]
labels = [result[1] for result in results]
from sklearn.metrics import accuracy_score
accuracy_score(labels, predictions, sample_weight=None) #0.8796