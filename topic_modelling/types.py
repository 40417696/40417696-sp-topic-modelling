"""
topic_modelling.types
~~~~~~~~~~~~~~~~~~~~~
This module contains type variables used to abbreviate longer type annotations.
"""

from typing import List, Tuple

from gensim.corpora import Dictionary

StringList = List[str]
BagOfWords = List[List[Tuple[int, int]]]
PreprocessedData = Tuple[List[StringList], Dictionary, BagOfWords]
WeightedWords = List[Tuple[str, float]]
