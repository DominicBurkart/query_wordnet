import os
import sys
from functools import lru_cache

from nltk.corpus import wordnet as wn


def get_ic(ic_str):
    @lru_cache(maxsize=1)
    def into_corpus(dirstr):
        import pandas as pd
        from nltk.tokenize import word_tokenize
        from nltk.corpus.reader.api import concat

        def files(s):
            for dir, dirs, files in os.walk(s):
                for f in files:
                    yield os.path.join(dir, f)

        def chunks(fp):
            if fp.endswith(".txt"):
                print("reading in text file as document: " + fp)
                with open(fp) as f:
                    return [f.read()]
            elif fp.endswith(".csv.gz"):
                print("reading in file as tweet documents: " + fp)
                return pd.read_csv(fp, compression="gzip").message.values

        return concat(word_tokenize(c) for f in files(dirstr) for c in chunks(f))

    if os.path.exists(ic_str):
        if os.path.isdir(ic_str):
            print("Assuming path leads to EITHER txt or twitter csv.gz files")
            return into_corpus(ic_str)
        elif ic_str == "brown":
            from nltk.corpus import brown
            return brown
        elif ic_str == "web":
            from nltk.corpus import webtext
            return webtext
        elif ic_str.endswith(".dat"):  # assume this is a wordnet corpus.
            return wn.WordNetICCorpusReader(ic_str)
        else:
            raise NotImplementedError


def syn_matrix(word1, word2, ic=None):
    def ic_none(word1, word2):
        for s1 in wn.synsets(word1):
            for s2 in wn.synsets(word2):
                if s1.pos() == s2.pos():
                    v = s1.path_similarity(s2)
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


def max_similarity(word1, word2, ic=None):
    try:
        return max(syn_matrix(word1, word2, ic))
    except ValueError:
        return 0  # syn_matrix couldn't be generated.


if __name__ == "__main__":
    print(locals()[sys.argv[1]](*sys.argv[2:]))
