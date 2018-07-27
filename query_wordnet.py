import sys
import os

from nltk.corpus import wordnet as wn


def get_ic(ic_str):
    if os.path.exists(ic_str):
        raise NotImplementedError("Parse into something with a words() function e.g. wn.ic(genesis, False, 0.0)")
    raise NotImplementedError("Parse into something with a words() function e.g. wn.ic(genesis, False, 0.0)"


def syn_matrix(word1, word2, ic=None):
    def ic_none(word1, word2):
        for s1 in wn.synsets(word1):
            for s2 in wn.synsets(word2):
                if s1.pos() == s2.pos():
                    v = s1.lch_similarity(s2)
                    if v is not None:
                        yield v

    def ic_some(word1, word2, crit):
        for s1 in wn.synsets(word1):
            for s2 in wn.synsets(word2):
                if s1.pos() == s2.pos():
                    v = s1.res_similarity(s2, crit)
                    if v is not None:
                        yield v

    if ic is None:
        return ic_none(word1, word2)
    elif ic is str:
        return ic_some(word1, word2, get_ic(ic))
    elif ic is dict:
        return ic_some(word1, word2, ic)
    else:
        raise TypeError("Weird type of IC in syn_matrix: " + str(type(ic)))


def max_similarity(word1, word2):
    try:
        return max(syn_matrix(word1, word2))
    except ValueError:
        return 0  # syn_matrix couldn't be generated.

if __name__ == "__main__":
    print(locals()[sys.argv[1]](*sys.argv[2:]))