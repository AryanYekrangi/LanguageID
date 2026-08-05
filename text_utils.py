from collections import Counter
from time import perf_counter
from contextlib import contextmanager

def load_data(filename:str, strip=True) -> list[str]:
    """
    Load data from text file and convert it into a list of strings.

    Parameters
    ----------
    filename : str
        Path to a UTF-8 encoded text file. Each line is treated as a separate entry.

    Returns
    -------
    data_list : list of str
        List where each element is a line from the file with leading/trailing whitespace removed.
    
    Examples
    -------
    X_train = load_data('x_train.txt')
    y_train = load_data('y_train.txt')
    """

    with open(filename, encoding='utf-8') as fhand:
        if strip:
            return [line.strip() for line in fhand]
        else:
            return [line.rstrip('\n') for line in fhand]

def combine_ngrams(df, ngrams_col):
    """Sum ngram Counters within each language into one Counter per language."""
    combined = {}
    for lang, group in df.groupby("target_language_TR"):
        total = Counter()
        for ngram_counts in group[ngrams_col]:
            total.update(ngram_counts)
        combined[lang] = total
    return combined

class Timer():
    """
    Callable context manager for timing blocks of code.
    timer = Timer()
    with timer('step 1'):
        # block of code
        """
    def __init__(self):
        self.records = []
    @contextmanager
    def __call__(self, name):
        start = perf_counter()
        yield
        end = perf_counter()
        elapsed = round(end - start, 2)
        print(f"{name}: {elapsed:.4f} s")
        self.records.append({"name": name, "time": elapsed})

timer = Timer()
