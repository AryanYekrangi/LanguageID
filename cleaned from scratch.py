import pandas as pd
from text_utils import load_data, combine_ngrams
from ngrams import ngram2dic, string2ngram, preprocess, train_language_models, predict_language
from uroman import Uroman
from time import perf_counter

# VARIABLES
N = 4

# LOAD DATA FROM TXT FILES
X_train = load_data('./WiLI-2018_dataset/x_train.txt')
y_train = load_data('./WiLI-2018_dataset/y_train.txt')
X_test = load_data('./WiLI-2018_dataset/x_test.txt')
y_test = load_data('./WiLI-2018_dataset/y_test.txt')

romanizer = Uroman()

# CREATNG TRAINING DATAFRAME
df_train = pd.DataFrame({
    "text": X_train,
    "target_language": y_train})
df_train.insert(2, "romanized", df_train.text.apply(lambda text: romanizer.romanize_string(text)))
df_train.insert(3, "text_ngrams", df_train.text.apply(lambda text: ngram2dic(string2ngram(preprocess(text), N))))
df_train.insert(4, "romanized_ngrams", df_train.romanized.apply(lambda text: ngram2dic(string2ngram(preprocess(text), N))))

# CREATING TESTING DATA
df_test = pd.DataFrame()
df_test.insert(0, 'text', X_test)
df_test.insert(1, 'target_language', y_test)
# current quick fix
df_test.iloc[41408] = [df_test.iloc[41408].text.replace('十分之', '分之'), df_test.iloc[41408].target_language]
df_test.insert(2, 'ngrams', df_test.text.apply(lambda text: 
                                     ngram2dic(string2ngram(text, N))))
df_test.insert(3, "romanized", df_test.text.apply(lambda text: romanizer.romanize_string(text)))
df_test.insert(4, "romanized_ngrams", df_test.romanized.apply(lambda text:
                                                         ngram2dic(string2ngram(text, N))))

# EXPORTING TRAINING AND TESTING DATAFRAME
df_train.to_pickle("df_train.pkl")
df_test.to_pickle("df_test.pkl")
df_new_train = pd.read_pickle("df_train.pkl") # this is to see whether the data is saved and loaded correctly
df_new_test = pd.read_pickle("df_test.pkl") # this is to see whether the data is saved and loaded correctly
    
# COMBINING NGRAMS FROM TRAINING DATA FOR TRAINING
lang_text_all_ngrams = {}
for lang in df_train.target_language.unique():
    lang_text_all_ngrams[lang] = combine_ngrams(df=df_train, lang=lang, col='text_ngrams')
lang_romanized_all_ngrams = {}
for lang in df_train.target_language.unique():
    lang_romanized_all_ngrams[lang] = combine_ngrams(df=df_train, lang=lang, col='romanized_ngrams')

# TRAINING MODEL
models_text = train_language_models(lang_text_all_ngrams, alpha=0.1)
models_romanized = train_language_models(lang_romanized_all_ngrams, alpha=0.1)


df_test = df_new_test.copy() # can design to remove this later
#     -----     EXPERIMENT     -----
# THIS IS VERY SLOW
results = []
#for i in range(len(df_train)):
for i in range(10000):
    text_ngrams = df_test.iloc[i].romanized_ngrams
    prediction = predict_language(models_text, text_ngrams, output='')
    label = df_test.iloc[i].target_language
    results.append([prediction, label])

predictions = [result[0][0] for result in results]
labels = [result[1] for result in results]
from sklearn.metrics import accuracy_score
accuracy_score(labels, predictions, sample_weight=None) #0.8796


# LOAD PICKLED DATA
df_train_pkl = pd.read_pickle("df_train.pkl")
df_test_pkl = pd.read_pickle("df_test.pkl")

# 1000 data, accuracy
# texst_ngrams = 0.878
# romanized_ngrams = 0.554, 10000=0.55
