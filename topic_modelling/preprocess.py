"""
topic_modelling.preprocess
~~~~~~~~~~~~~~~~~~~~~~~~~~
This module contains functions for the preprocessing of data before
ingestion into a machine learning model.
"""

import re
from typing import List, Pattern

import spacy
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models.phrases import Phraser
from gensim.parsing.preprocessing import remove_stopwords

from topic_modelling.types import StringList, PreprocessedData

PTN_NUMBER: Pattern[str] = re.compile(r'\d+')
PTN_SYMBOL: Pattern[str] = re.compile(
    r'[!"£$€%^&*()_=+\\|,<.>/?;:\'@#~\[\]{}”’–]')
PTN_SPACE: Pattern[str] = re.compile(r'\s+')
PTN_DASH: Pattern[str] = re.compile(r'-')

NLP: spacy.lang.en.English = spacy.load(
    'en_core_web_sm', disable=['parser', 'ner'])
LEXICAL_CATEGORIES: StringList = ['NOUN', 'ADJ', 'VERB', 'ADV']

MIN_WORD_LENGTH: int = 2
EXCLUDED_TERMS: StringList = [
    'say', 'year', 'people', 'new', 'good',
    'time', 'come', 'take', 'want', 'use',
    'day', 'week', 'month', 'get', 'set'
]


def preprocess_data(data: StringList) -> PreprocessedData:
    """Performs preprocessing on the given data, removing numbers,
    symbols, stopwords and more.

    Parameters
    ----------
    data : list of str
        A list of string data values.

    Returns
    -------
    data : list of list of str
        A list of preprocessed data values for use by a topic model.
    id2word : :class:`~gensim.corpora.dictionary.Dictionary`
        A mapping of integer IDs to words.
    corpus : list of list of (int, int)
        The preprocessed data converted to bag of words format.
    """
    processed_data: List[StringList] = []

    # Perform basic level of preprocessing
    for value in data:
        value = value.lower()  # Set to lowercase
        value = PTN_NUMBER.sub('', value)  # Remove numbers
        value = PTN_SYMBOL.sub('', value)  # Remove symbols
        value = PTN_SPACE.sub(' ', value)  # Replace whitespace
        value = PTN_DASH.sub(' ', value)  # Replace dashes
        value = value.strip()  # Remove trailing whitespace

        # Split sentence into list of words (tokenise)
        processed_data.append(value.split())

    # Generate bigram model
    bigram_model = Phraser(Phrases(processed_data, min_count=5, threshold=100))

    # Remove stopwords
    processed_data = [
        [remove_stopwords(word) for word in words]
        for words in processed_data
    ]

    # Make and insert bigrams
    processed_data = [bigram_model[words] for words in processed_data]

    # Perform lemmatisation
    processed_data = [
        [word.lemma_ for word in NLP(' '.join(words))
         if word.pos_ in LEXICAL_CATEGORIES]
        for words in processed_data
    ]

    # Remove excluded and short words
    processed_data = [
        [word for word in words
         if word not in EXCLUDED_TERMS and len(word) > MIN_WORD_LENGTH]
        for words in processed_data
    ]

    # Map words to integer IDs
    id2word = Dictionary(processed_data)

    # Convert to "bag of words" format
    corpus = [id2word.doc2bow(words) for words in processed_data]

    return processed_data, id2word, corpus
