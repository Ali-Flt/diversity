from typing import List
import nltk
import numpy as np


def self_repetition(
    data: List[str],
    n: int = 4,
) -> float:
    all_ngrams = [list(nltk.ngrams(d.split(' '), n)) for d in data]
    self_reps = []
    for i, _ in enumerate(data):
        N_is = []
        for d1_ngram in all_ngrams[i]:
            N_i = 0
            for j, _ in enumerate(data):
                if i == j:
                    continue
                if d1_ngram in all_ngrams[j]:
                    N_i += 1
            N_is.append(N_i)
        self_reps.append(np.log(sum(N_is) + 1))
    return sum(self_reps) / len(data)
