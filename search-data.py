import csv
# Hints:
# • Refer to the Python examples in THEO-10-gensim.pdf and to the Python scripts mentioned in the slides and available on iCorsi
# • Sort the documents in the corpus by similarity to get the top-5 entities most similar to the query for FREQ, TF-IDF, LSI
# • Use function most_similar with topn=5 to get the top-5 entities most similar to the query for Doc2Vec
import re


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

def underscore_split(text):
    """
    Split a text by underscore (e.g., go_to_myHome -> [go, to, my, home]).
    Reference: stackoverflow.com/questions/29916065

    :param text: text to be splitted
    """
    matches = text.split("_")
    return matches

def camel_case_split(text):
    """
    Split a text by CamelCase (works with: ABCdef, AbcDef, abcDef, abcDEF)
    Reference: stackoverflow.com/questions/29916065

    :param text: text to be splitted
    """
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", text)
    return [m.group(0) for m in matches]

def data_standardization(data):
    """
    Standardize a text:
    1. split entity names (by CamelCase and underscore)
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    :param data: text to be standardized
    """
    data_standardized = []
    stopwords = ["test", "tests", "main"]

    for element in data:
        words_no_underscore = underscore_split(element)
        for word in words_no_underscore:
            words_standardized = [x.lower() for x in camel_case_split(word)]
            words_filtered = [i for i in words_standardized if i not in stopwords]
            data_standardized += words_filtered

    return data_standardized

def create_corpus():
    """
    Create a corpus from the code entity names and comments.
    """
    data_raw = []
    with open("data.csv") as csv_file:
        extracted_data = csv.DictReader(csv_file, delimiter=",")
        for row in extracted_data:
            data_raw.extend((row["name"], row["comment"]))

    corpus = data_standardization(data_raw)
    print(corpus)

def main():
    create_corpus()
    represent_entities()
    query = "Optimizer that implements the Adadelta algorithm"
    print_top_5_entities()


if __name__ == "__main__":
    main()
