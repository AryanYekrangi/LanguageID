import pandas as pd
from text_utils import load_data
from ngrams import ngram2dic, string2ngram, preprocess
from uroman import Uroman
from collections import Counter
from time import perf_counter
import pickle
# VARIABLES
N = 4

# LOAD DATA FROM TXT FILES
X_train = load_data('./WiLI-2018_dataset/x_train.txt')
y_train = load_data('./WiLI-2018_dataset/y_train.txt')
X_test = load_data('./WiLI-2018_dataset/x_test.txt')
y_test = load_data('./WiLI-2018_dataset/y_test.txt')


romanizer = Uroman()
# better way of creating a dataframe
df_train = pd.DataFrame({
    "text": X_train,
    "target_language": y_train
})

start_t = perf_counter()
df_train.insert(2, "romanized", df_train.text.apply(lambda text: romanizer.romanize_string(text)))
end_t = perf_counter()
print(end_t - start_t) # 2116.4 s = 35 m.

df_train.insert(3, "text_ngrams", df_train.text.apply(lambda text: ngram2dic(string2ngram(preprocess(text), N))))
df_train.insert(4, "romanized_ngrams", df_train.text.apply(lambda text: ngram2dic(string2ngram(preprocess(text), N))))


"""
# READING FILE (900MB)
import ast
import time
start_t = time.perf_counter()

new_df = pd.read_csv('df_rm.csv')
new_df_copy = new_df.copy()
str_dic_text = new_df_copy.text_ngrams.apply(lambda string: ast.literal_eval(string))
str_dic_romanized = new_df.romanized_ngrams.apply(lambda string: ast.literal_eval(string))

end_t = time.perf_counter()
print(end_t - start_t) # 382 s

new_df_copy = new_df.copy()
new_df_copy.text_ngrams = str_dic_text
new_df_copy.romanized_ngrams = str_dic_romanized
"""
# TODO: remake combine dictionary function and import into text_utils.py
def combine(df, lang, col):
    lang_df = df[df.target_language==lang]
    total = Counter()
    for i in range(len(lang_df)):
        total += Counter(lang_df[col].values[i])
    return total

start_t = perf_counter()
lang_text_all_ngrams = {}
for lang in df_train.target_language.unique():
    lang_text_all_ngrams[lang] = combine(df=df_train, lang=lang, col='text_ngrams')

lang_romanized_all_ngrams = {}
for lang in df_train.target_language.unique():
    lang_romanized_all_ngrams[lang] = combine(df=df_train, lang=lang, col='romanized_ngrams')
end_t = perf_counter()
print(end_t - start_t) # 321 s


# === TRAIN MODEL ===
from ngrams import NgramLanguageModel, train_language_models
models_text = train_language_models(lang_text_all_ngrams, alpha=0.1)
models_romanized = train_language_models(lang_romanized_all_ngrams, alpha=0.1)


# creating testing data
df_test = pd.DataFrame()
df_test.insert(0, 'text', X_test)
df_test.insert(1, 'target_language', y_test)
# current quick fix
df_test.iloc[41408] = [df_test.iloc[41408].text.replace('十分之', '分之'), df_test.iloc[41408].target_language]
df_test.insert(2, 'ngrams', df_test.text.apply(lambda text: 
                                     ngram2dic(string2ngram(text, N))))
df_test.insert(3, "romanized", df_test.text.apply(lambda text: romanizer.romanize_string(text)))


df_test.to_pickle("df_test.pkl")
df_new_test = pd.read_pickle("df_test.pkl")


def predict_language(models, text):
    #trigrams = extract_trigrams(normalize(text))
    ngrams = text
    scores = {
        lang: model.log_probability(ngrams)
        for lang, model in models.items()
    }
    return max(scores, key=scores.get)#, scores
    # .Can also return all scores to compare what languages are similar


predict_language(models=models_text, text=df_test.iloc[1,].ngrams)

#     -----     EXPERIMENT     -----
# THIS IS VERY SLOW
results = []
#for i in range(len(df_train)):
for i in range(10000):
    text_trigrams = df_test.iloc[i].ngrams
    prediction = predict_language(models_text, text_trigrams)
    label = df_test.iloc[i].target_language
    results.append([prediction, label])

predictions = [result[0] for result in results]
labels = [result[1] for result in results]
from sklearn.metrics import accuracy_score
accuracy_score(labels, predictions, sample_weight=None) #0.8796

start_t = perf_counter()
df_train.to_pickle("df_train.pkl")
end_t = perf_counter()
print(end_t - start_t) # 42 s.

start_t = perf_counter()
df_new = pd.read_pickle("df_train.pkl")
end_t = perf_counter()
print(end_t - start_t) # 26 s.



# AttributeError: 'Edge' object has no attribute 'value'
bad_rows = []
for i, text in enumerate(df_test["text"]):
    try:
        romanizer.romanize_string(text)
    except Exception as e:
        bad_rows.append((i, text, str(e)))
        continue

print("Total bad rows:", len(bad_rows))

