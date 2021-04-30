"""
topic_modelling.core
~~~~~~~~~~~~~~~~~~~~
This module contains functions used for result gathering for topic models.
"""

from typing import List

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models import basemodel, CoherenceModel

from topic_modelling.types import StringList, WeightedWords


def calculate_coherence(model: basemodel.BaseTopicModel,
                        data: List[StringList], id2word: Dictionary,
                        measure: str = 'c_v') -> float:
    """Calculates the coherence score of the given topic model.

    Parameters
    ----------
    model : :class:`~gensim.models.basemodel.BaseTopicModel`
        The topic model we are calculating coherence for.
    data : list of list of str
        A list of preprocessed data values used with the topic model.
    id2word : :class:`~gensim.corpora.dictionary.Dictionary`
        A mapping of integer IDs to words.
    measure : str, optional
        The specific coherence measurement to use.
        Defaults to 'c_v'.

    Returns
    -------
    coherence_score : float
        The coherence score of this topic model.
    """
    coherence_model = CoherenceModel(
        model=model, texts=data, dictionary=id2word, coherence=measure)
    return coherence_model.get_coherence()


def form_wordcloud(title: str, topic: WeightedWords) -> None:
    """Forms a cloud of words for a specific topic.

    Parameters
    ----------
    title : str
        The title of the wordcloud.
    topic : list of (str, float)
        A list of words their and weightings.
    """
    plt.title(title)
    plt.imshow(WordCloud().fit_words(dict(topic)))
    plt.axis('off')
    plt.show()


def graph_results(title: str, num_topics: List[int],
                  coherence_scores: List[float]) -> None:
    """Graphs coherence scores by topic numbers.

    Parameters
    ----------
    title : str
        The title of the graph.
    num_topics : list of int
        A list containing every topic number.
    coherence_scores : list of int
        A list containing every coherence score.
    """
    plt.plot(num_topics, coherence_scores)
    plt.title(title)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.show()
