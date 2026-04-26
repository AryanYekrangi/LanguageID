import pandas as pd
from text_utils import load_data
from ngrams import ngram2dic, string2ngram, preprocess
from uroman import Uroman
from collections import Counter

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


df_rm = pd.read_csv('data_and_romanized.csv', encoding='utf-8')
df_rm = df_rm.drop(df_rm.columns[0], axis=1)

df_rm.insert(3,
             'text_ngrams',
             df_rm.text.apply(lambda text: ngram2dic(string2ngram(preprocess(text), 4))))

df_rm.insert(4,
             'romanized_ngrams',
             df_rm.romanized.apply(lambda text: ngram2dic(string2ngram(preprocess(text), 4))))


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

# TODO: remake combine dictionary function and import into text_utils.py
# TODO: 1. define df, define lang, define column (romanized vs text ngrams)
def combine(df, lang, col):
    lang_df = df[df.target_language==lang]
    total = Counter()
    for i in range(len(lang_df)):
        total += Counter(lang_df[col].values[i])
    return total

start_t = time.perf_counter()
lang_text_all_ngrams = {}
for lang in new_df_copy.target_language.unique():
    lang_text_all_ngrams[lang] = combine(df=new_df_copy, lang=lang, col='text_ngrams')

lang_romanized_all_ngrams = {}
for lang in new_df_copy.target_language.unique():
    lang_romanized_all_ngrams[lang] = combine(df=new_df_copy, lang=lang, col='romanized_ngrams')
end_t = time.perf_counter()
print(end_t - start_t)



# === TRAIN MODEL ===
from ngrams import NgramLanguageModel, train_language_models
models_text = train_language_models(lang_text_all_ngrams, alpha=0.1)
models_romanized = train_language_models(lang_romanized_all_ngrams, alpha=0.1)

# FIX FOR NOW REMOVE LATER
import math
for k, v in models_text.items():
    if isinstance(v, float) and math.isnan(v):
        models_text[k] = 'nan'

# creating testing data
df_test = pd.DataFrame()
df_test.insert(0, 'text', X_test)
df_test.insert(1, 'target_language', y_test)
df_test.insert(2, 'ngrams', df_test.text.apply(lambda text: 
                                     ngram2dic(string2ngram(text,4))))

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
for i in range(100):
    text_trigrams = df_test.iloc[i].ngrams
    prediction = predict_language(models, text_trigrams)
    label = df_test.iloc[i].target_language
    results.append([prediction, label])


#TODO: 'nan' is not transferred into dataframes NaN
