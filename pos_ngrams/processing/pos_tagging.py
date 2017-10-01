#!/usr/bin/env python

"""Functions for Part Of Speech (POS) tagging"""

import os
import pickle
import re

from pos_ngrams.preprocessing.tokenizer import tokenizer_sentence

__author__ = "Peter J Usherwood"
__python_version__ = "3.6"


def tag_snippet(snippet, tagger_name):
    """
    Tag Snippets using a pos tagger

    :param snippet: Text snippet
    :param tagger_name: Name of pos tagger as it appears in utils_data/models/pos_taggers/

    :return: List of tuples for the tagged snippet
    """

    file = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../models/' +
                        tagger_name + '.pkl')

    input = open(file, 'rb')
    tagger = pickle.load(input)
    input.close()

    sent_tagged = []
    for sent in tokenizer_sentence(snippet):
        tokens = re.findall(r"[\w']+|[.,!?;]", sent)
        sent_tagged += tagger.tag(tokens)
    return sent_tagged

