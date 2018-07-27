from functools import lru_cache

from nltk.corpus import wordnet as wn


def get_ic(ic):
    raise NotImplementedError


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


@lru_cache()
def avg_similarity(word1, word2):
    l = list(syn_matrix(word1, word2))
    if len(l) != 0:
        return sum(l) / len(l)
    return 0


@lru_cache() # this doesn't work in practice.
def match_similarity(word1, word2, ic=None, symmetric=True):
    s1s = wn.synsets(word1)
    s2s = wn.synsets(word2)
    try:
        if ic is None:
            maxes = [max(s1.lch_similarity(s2)
                         if s1.pos() == s2.pos()
                            and s1.lch_similarity(s2) is not None
                         else 0
                         for s2 in s2s)
                     for s1 in s1s]
        elif ic is dict:
            maxes = [max(s1.res_similarity(s2, ic)
                         if s1.pos() == s2.pos()
                            and s1.res_similarity(s2, ic) is not None
                         else 0
                         for s2 in s2s)
                     for s1 in s1s]
        elif ic is str:
            maxes = [max(s1.res_similarity(s2, get_ic(ic))
                         if s1.pos() == s2.pos()
                            and s1.res_similarity(s2, get_ic(ic)) is not None
                         else 0
                         for s2 in s2s)
                     for s1 in s1s]
        if symmetric:
            return sum((sum(maxes) / len(maxes), match_similarity(word2, word1, ic, symmetric=False))) / 2
        else:
            return sum(maxes) / len(maxes)
    except ValueError or ZeroDivisionError:
        return 0


@lru_cache()
def max_similarity(word1, word2):
    try:
        return max(syn_matrix(word1, word2))
    except ValueError:
        return 0  # syn_matrix couldn't be generated.
