#!/usr/bin/env python

"""Function for removing stop words from text using a combination of NLTK and custom lists"""

import os

from nltk.corpus import stopwords
from pos_ngrams.preprocessing.tokenizer import tokenizer_word

__author__ = "Peter J Usherwood"
__python_version__ = "3.6"


def stopword_removal(text_string=None,
                     tokens=None,
                     pos_tuples=False,
                     language='english',
                     adhoc_list=[],
                     ignore_nltk=False):
    """
    Main function you should import for stopword removal

    :param text_string: String you wish to remove stopwords from, this should be pre cleaned to lower and remove
    punctuation first, please see cleaning.py
    :param tokens: Python list of strings already tokenized to have stopwords removed
    :param pos_tuples: Bool, if tokens are a list of pos_tuples set this to true
    :param language: String of the language name you wish to remove basic stop words for, by default
    the program will look in NLTK for the language list, and if it cannot find it, it will look in
    utils_data>stopwords>language_basics.
    :param adhoc_list: List of strings of specific adhoc words you would like removed
    :param ignore_nltk: Boolean to ignore NLTK presets for basic language and look first in:
    utils_data>stopwords>language_basics, not recommended for common languages

    :return: Returns the string you entered minus the stopwords in the superset of the above lists
    """

    stopwords_set = create_stopwords_set(basic_language=language,
                                         adhoc_list=adhoc_list,
                                         ignore_nltk=ignore_nltk)

    if tokens is None:
        tokens = tokenizer_word(text_string)
        tokens = [token for token in tokens if token not in stopwords_set]
        stopped = " ".join(tokens)
    elif pos_tuples:
        stopped = [(token, tag) for token, tag in tokens if token not in stopwords_set]
    else:
        stopped = [token for token in tokens if token not in stopwords_set]

    return stopped


def create_stopwords_set(basic_language, adhoc_list=[], ignore_nltk=False):
    """
    Function used to create a superset of stopwords from multiple lists

    :param basic_language: String of the language name you wish to remove basic stop words for, by default
    the program will look in NLTK for the language list, and if it cannot find it, it will look in
    utils_data>stopwords>language_basics.
    :param adhoc_list: List of strings of specific adhoc words you would like removed
    :param ignore_nltk: Boolean to ignor NLTK presets for basic language and look first in:
    utils_data>stopwords>language_basics, not recommended for common languages
    :return: Set of stopwords
    """

    stopwords_set = set([])

    # Append basic language sets using NLTK and/or language_basic files
    stopwords_set = stopwords_set.union(stopwords.words(basic_language))

    # Append adhoc words to set
    adhoc_set = set(adhoc_list)
    stopwords_set = stopwords_set.union(adhoc_set)

    return stopwords_set
