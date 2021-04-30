"""
Topic Modelling Library
~~~~~~~~~~~~~~~~~~~~~~~
This is a basic topic modelling library, carrying a basic preprocessing
implementation as well as some evaluation functionality.
"""

from .core import calculate_coherence, form_wordcloud, graph_results
from .preprocess import preprocess_data
from .types import StringList

DEFAULT_LABELS: StringList = [
    'business', 'entertainment', 'politics',
    'sport', 'technology'
]

__all__ = [
    'DEFAULT_LABELS',
    'calculate_coherence', 'form_wordcloud',
    'graph_results', 'preprocess_data'
]
