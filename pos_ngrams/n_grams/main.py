#!/usr/bin/env python

"""Main class for performing ngrams analysis on a pandas_df containing a series of text mentions"""

import pandas as pd
from pos_ngrams.preprocessing.preprocess import preprocess_df
from pos_ngrams.n_grams import processes

__author__ = "Peter J Usherwood"
__python_version__ = "3.5"


class NGrams():
    """
    The parent class for managing n-gram analysis
    """

    def __init__(self, data, text_field_key='Snippet'):
        """

        :param data: Pandas dataframe containing a text Snippet field and other metadata
        :param text_field_key: The name of the text field (by default Snippet)
        """

        self.text_field_key = text_field_key
        self.data = data.copy()
        self.cv = None
        self.ngrams_df = pd.DataFrame(['blank'], columns=['Index'])
        self.filtered_ngrams_df = pd.DataFrame(['blank'], columns=['Index'])
        self.ngram_word = None
        self.word_frequency_matrix = pd.DataFrame(['blank'], columns=['Index'])
        self.ids_enriched = False

    def ngram_pipeline(self, min_gram=2, max_gram=4, preprocess_data=False,
                       language='english', adhoc_stopwords=[], max_features=1000,
                       tfidf=True, pos_tuples=False):
        """
        The primary function that creates the ngrams dataframe which contains: NGram name, frequency, and index (until
        fortified with additional data).

        :param data: Pandas dataframe containing a text Snippet field and other metadata
        :param min_gram: Int, The minimum n
        :param max_gram: Int, The maximim n
        :param preprocess_data: Whether the ngrams class should handle the preprocessing of the text data prior to
        ngram analysis. NB It is strongly recommended that the data is cleaned by some process
        :param language: If preprocessing this imput dictates the language choice of stemming and basic stopwords
        :param additional_list: If preprocessing this is the second input into stopwords (see stopwords)
        :param adhoc_stopwords: If preprocessing this is the third input into stopwords (see stopwords)
        :param max_features: Int the maximum number of features to generate
        :param tfidf: Bool, whether to use the rate countvectorizer instead of the deafult counts one
        :param pos_tuples: Bool, if tokens are a list of pos_tuples set this to true
        """

        if preprocess_data:
            self.data = preprocess_df(self.data,
                                      self.text_field_key,
                                      language=language,
                                      adhoc_stopwords=adhoc_stopwords,
                                      pos_tuples=pos_tuples)
            self.text_field_key = 'Preprocessed'

        ngrams, word_frequency_matrix, cv = processes.generate_ngrams(self.data,
                                                                      min_gram,
                                                                      max_gram,
                                                                      self.text_field_key,
                                                                      max_features=max_features,
                                                                      tfidf=tfidf,
                                                                      pos_tuples=pos_tuples)
        self.ngrams_df = ngrams
        self.word_frequency_matrix = word_frequency_matrix
        self.cv = cv

        self.ids_enriched = False

        return True

    def search_on_word(self, ngram_word, stemmed_ngrams=True):
        """
        Populates the filtered_ngrams_df which is a subset of the main ngrams_df but for ngrams containing the key
        search word ngram_word

        :param ngram_word: String, the word to return ngrams containing
        :param stemmed_ngrams: Boolean, if the data has been stemmed set this as true and the ngram_word will be
        stemmed as well, otherwise it wont match.
        """

        self.ngram_word = ngram_word
        if stemmed_ngrams:
            self.ngram_word = processes.stem_text(self.ngram_word)

        self.filtered_ngrams_df = self.ngrams_df[self.ngrams_df['Ngram'].str.contains(self.ngram_word)]

        return True

    def fortify_with_id(self, filtered_df=False, take_top_x=300):

        if filtered_df:
            ngrams = processes.fortify_ngrams_with_ids(self.filtered_ngrams_df, self.word_frequency_matrix, take_top_x=take_top_x)
            self.filtered_ngrams_df = ngrams
            self.ngrams_df['Original Data Keys'] = ngrams['Original Data Keys']
        else:
            ngrams = processes.fortify_ngrams_with_ids(self.ngrams_df, self.word_frequency_matrix, take_top_x=take_top_x)
            self.ngrams_df = ngrams

        self.ids_enriched = True

        return True

    def aggregate_other_data_column(self, column_key_to_agg='Sentiment', new_column_key='Agg', filtered_df=False):

        if self.ids_enriched:
            for ids in self.ngrams_df['Original Data Keys']:
                mean = self.data.ix[ids, column_key_to_agg].mean()

        else:
            print('Enrich with ids first')
            return False
