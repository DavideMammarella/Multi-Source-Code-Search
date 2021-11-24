import re
import csv
# Hints:
# • Use regular expressions (re.search) for camel-case splitting
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
    matches = text.split("_")
    return matches

def camel_case_split(text):
    """
    CamelCase split of a text (works with: ABCdef, AbcDef, abcDef, abcDEF)
    Reference: stackoverflow.com/questions/29916065

    :param text: text to be splitted
    """
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", text)
    return [m.group(0) for m in matches]

def data_standardization(data):
    """
    Standardize a text:
    1. split entity names by camel-case and underscore (e.g., go_to_myHome -> [go, to, my, home])
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    :param data: text to be standardized
    """
    data_standardized = []
    stopwords = ["test", "tests", "main"]

    for element in data:
        list_underscore = underscore_split(element)
        for element_underscore in list_underscore:
            list_elements = [x.lower() for x in camel_case_split(element_underscore)]
        data_temp = [i for i in list_elements if i not in stopwords]
        data_standardized += data_temp

    return data_standardized

def create_corpus():
    """
    Create a corpus from the code entity names and comments
    """
    with open("data.csv") as csv_file:
        extracted_data = csv.DictReader(csv_file, delimiter=",")
        names_and_comments_raw = [{key: row[key] for key in ("name", "comment")} for row in extracted_data]
    names_raw = [d["name"] for d in names_and_comments_raw if "name" in d]
    comments_raw = [d["comment"] for d in names_and_comments_raw if "comment" in d]
    data_raw = names_raw + comments_raw

    corpus = data_standardization(data_raw)
    print(corpus)

def main():
    create_corpus()
    represent_entities()
    query = "Optimizer that implements the Adadelta algorithm"
    print_top_5_entities()


if __name__ == "__main__":
    main()
