
# Hints:
# • Use regular expressions (re.search) for camel-case splitting
# • Refer to the Python examples in THEO-10-gensim.pdf and to the Python scripts mentioned in the slides and available on iCorsi
# • Sort the documents in the corpus by similarity to get the top-5 entities most similar to the query for FREQ, TF-IDF, LSI
# • Use function most_similar with topn=5 to get the top-5 entities most similar to the query for Doc2Vec


def print_top_5_entities():
    """
    Given a query string, for each embedding print the top-5 most similar entities
    (entity name, file name, line of code), based on cosine similarity.
    """

def represent_entities():
    """"
    Represent entities using the following vector embeddings:
    - FREQ: frequency vectors
    - TF-IDF: TF-IDF vectors
    - LSI: LSI vectors with k = 300
    - Doc2Vec: doc2vec vectors with k = 300
    """

def create_corpus():
    """
    Create a corpus from the code entity names and comments:
    1. split entity names by camel-case and underscore (e.g., go_to_myHome -> [go, to, my, home])
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    """


def main():
    create_corpus()
    represent_entities()
    query = "Optimizer that implements the Adadelta algorithm"
    print_top_5_entities()


if __name__ == "__main__":
    main()
