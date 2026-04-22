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
def combine_dic_pickle(df, lang):
    """combines dictionaries and dumps them as pkl"""
    lang_df = df[df.target_language==lang]
    total = Counter(dict())
    for i in range(len(lang_df)):
        total += Counter(lang_df.ngrams.values[i])
    with open(f'./ngrams_for_each_language/unigram_nopreprocessing/{lang}.pkl', 'wb') as f:
        pickle.dump(total, f)