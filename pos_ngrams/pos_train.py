#!/usr/bin/env python

"""Functions for Part Of Speech (POS) tagging"""

import nltk
import os
import numpy as np
from itertools import compress
from nltk.corpus import brown
import string
import pickle


__author__ = "Peter J Usherwood"
__python_version__ = "3.6"


def parse_browns_corpus_to_simplified(browns_tagged_sents):
    """
    Parse the tagged Browns corpus (in an array of sentences) into simplified tags

    :param browns_tagged_sents: Array of tagged Browns sentences

    :return: Array of sentences with simplified tags
    """

    sents_simplified = []
    for sent in browns_tagged_sents:
        sents_simplified.append([(tuples[0], simplify_brown_tags(tuples[1])) for tuples in sent])

    return sents_simplified


def parse_corpus_to_simplified_es(tagged_sents):
    """
    Parse the tagged Spanish corpus (in an array of sentences) into simplified tags

    :param tagged_sents: Array of tagged Browns sentences

    :return: Array of sentences with simplified tags
    """

    sents_simplified = []
    for sent in tagged_sents:
        sents_simplified.append([(tuples[0], simplify_tags_es(tuples[1])) for tuples in sent])

    return sents_simplified


def train_pos_tagger(name='simplified_en',
                     corpus=brown.tagged_sents(),
                     tagset='brown',
                     simplified=True,
                     regex=True,
                     regex_language='en',
                     train_test_split=.8
                     ):
    """
    Train the tag pos tagger and persist to disk

    :param name: The name of the file to persist to
    :param corpus: The tagged corpus to train and test on, it should be a list of sentences, each sentence should
    be a list of tuples with the word first and the pos tag second.
    :param tagset: String, the type of tags to be used, options:
                    - 'brown' (en)
                    - 'parole' (es)
    :param simplified: Bool, True to parse the tags to a simplified subset (good for classification)
    :param regex: Bool, True to use regex to infer the tags that cant
    :param regex_language: String, langauge of the training corpus, used for regex tags.
    :param train_test_split: Decimal between 0 and 1, the ration of the train to test split
    """

    default_tag = None
    patterns = None

    if not corpus:
        print('Error no corpus supplied')
        return False

    if tagset not in ['brown', 'parole']:
        raise Exception('Please choose a valid tagset from:', str(['brown', 'parole']))

    if simplified:

        default_tag = 'NN'

        if tagset == 'brown':
            corpus = parse_browns_corpus_to_simplified(corpus)
        elif tagset == 'parole':
            corpus = parse_corpus_to_simplified_es(corpus)

        if regex_language == 'en':
            patterns = [(r'.*ing$', 'VB'),               # gerunds
                        (r'.*ed$', 'VB'),                # simple past
                        (r'.*es$', 'VB'),                # 3rd singular present
                        (r'^-?[0-9]+(.[0-9]+)?$', 'NU'),  # cardinal numbers
                        (r'.*', 'NN')]                    # nouns (default)
    else:

        if tagset == 'brown':
            default_tag = 'NN'
        elif tagset == 'parole':
            default_tag = 'NCS'

        if regex_language == 'en':
            patterns = [(r'.*ing$', 'VBG'),  # gerunds
                        (r'.*ed$', 'VBD'),  # simple past
                        (r'.*es$', 'VBZ'),  # 3rd singular present
                        (r'.*ould$', 'MD'),  # modals
                        (r'.*\'s$', 'NN$'),  # possessive nouns
                        (r'.*s$', 'NNS'),  # plural nouns
                        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
                        (r'.*', 'NN')]  # nouns (default)

    msk = np.random.rand(len(corpus)) < train_test_split
    train = list(compress(corpus, msk))
    test = list(compress(corpus, [not i for i in msk]))

    if regex:
        t0 = nltk.RegexpTagger(patterns)
    else:
        t0 = nltk.DefaultTagger(default_tag)

    t1 = nltk.UnigramTagger(train, backoff=t0)
    t2 = nltk.BigramTagger(train, backoff=t1)

    print('Accuracy ', str(t2.evaluate(test)))
    print('Saving to models/' + name + '.pkl')

    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'
                        + name + '.pkl')
    save = open(file, 'wb')
    pickle.dump(t2, save, -1)
    save.close()

    return True


def simplify_brown_tags(tag):
    """
    Created simplified tags from the Browns corpus

    :param tag: Str, the current tag to be transformed

    :return: transformed tag
    """

    NP = ['NP']  # proper noun
    NN = ['NR', 'NN']  # noun
    VB = ['VB', 'DO', 'EX', 'HV', 'BE', 'DI', 'HV']  # verb
    NU = ['CD', 'OD']  # numbers
    AD = ['JJ']  # adjective
    QL = ['QL', 'DT', 'WQ']  # qualifier
    AV = ['RB', 'RN', 'RP']  # adverb
    NG = ['*:']  # negator
    PN = ['PN', 'PP', 'PR', 'WP'] # pronoun

    tag = tag[:2]

    if tag in NP:
        tag = 'NP'
    elif tag in list(string.punctuation):
        tag = tag
    elif tag in NN:
        tag = 'NN'
    elif tag in VB:
        tag = 'VB'
    elif tag in NU:
        tag = 'NU'
    elif tag in AD:
        tag = 'AD'
    elif tag in QL:
        tag = 'QL'
    elif tag in AV:
        tag = 'AV'
    elif tag in NG:
        tag = '*:'
    elif tag in PN:
        tag = 'PN'
    else:
        tag = 'OT'

    return tag


def simplify_parole_tags(tag):
    """
    Created simplified tag from a parole tagged corpus

    :param tag: Str, the current tag to be transformed

    :return: transformed tag
    """

    NP = ['NP']  # proper noun
    NN = ['NC']  # noun
    VB = ['VS', 'VM', 'VA']  # verb
    NU = ['Z', 'Zm', 'Zp']  # numbers
    AD = ['AO', 'AQ']  # adjective
    QL = []  # qualifier
    AV = ['RG', 'RN']  # adverb
    NG = []  # negator

    tag = tag[:2]

    if tag in NP:
        tag = 'NP'
    elif tag in list(string.punctuation):
        tag = tag
    elif tag in NN:
        tag = 'NN'
    elif tag in VB:
        tag = 'VB'
    elif tag in NU:
        tag = 'NU'
    elif tag in AD:
        tag = 'AD'
    elif tag in QL:
        tag = 'QL'
    elif tag in AV:
        tag = 'AV'
    elif tag in NG:
        tag = '*:'
    else:
        tag = 'OT'

    return tag


def simplify_tags_es(tag):
    """
    Creates simplified tag for the spanish tagger

    :param tag: Str, the current tag to be transformed

    :return: transformed tag
    """

    NP = ['NP']  # proper noun
    NN = ['NC', 'W', 'NP']  # noun
    VB = ['VA', 'VM', 'VS']  # verb
    NU = ['Z', 'Zd', 'Zm', 'Zp']  # numbers
    AD = ['JJ']  # adjective
    QL = ['DA', 'DD', 'DI', 'DT', 'PD', 'PI']  # qualifier
    AV = ['RG', 'RN']  # adverb
    PN = ['DP', 'PX', 'P0', 'PP', 'PR', 'PT']  # pronoun
    PU = ['Fa', 'Fc', 'Fd', 'Fe', 'Fg', 'Fh', 'Fi', 'Fp', 'Fr', 'Fr', 'Fp', 'Fr', 'Fs', 'Fx', 'Fz']
    # NG = ['*:']  # negator

    tag = tag[:2]

    if tag in NP:
        tag = 'NP'
    elif tag in PU:
        tag = '.'
    elif tag in NN:
        tag = 'NN'
    elif tag in VB:
        tag = 'VB'
    elif tag in NU:
        tag = 'NU'
    elif tag in AD:
        tag = 'AD'
    elif tag in QL:
        tag = 'QL'
    elif tag in AV:
        tag = 'AV'
    # elif tag in NG: # negator
    #     tag = '*:'
    elif tag in PN:
        tag = 'PN'
    else:
        tag = 'OT'

    return tag
