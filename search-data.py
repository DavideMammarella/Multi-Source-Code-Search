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

def snake_case_to_camel_case(text):
    """
    CamelCase conversion of a snake_case text.
    Reference: stackoverflow.com/questions/19053707

    :param text: text to be converted (snake_case format)
    """
    components = text.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def camel_case_split(text):
    """
    CamelCase split of a text (works with: ABCdef, AbcDef, abcDef, abcDEF)
    Reference: stackoverflow.com/questions/29916065

    :param text: text to be splitted
    """
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", text)
    return [m.group(0) for m in matches]

def create_corpus():
    """
    Create a corpus from the code entity names and comments:
    1. split entity names by camel-case and underscore (e.g., go_to_myHome -> [go, to, my, home])
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    """
    with open("data.csv") as csv_file:
        extracted_data = csv.DictReader(csv_file, delimiter=",")
        names_and_comments_raw = [{key: row[key] for key in ("name", "comment")} for row in extracted_data]

    names_raw = [d["name"] for d in names_and_comments_raw if "name" in d]
    comments_raw = [d["comment"] for d in names_and_comments_raw if "comment" in d]
    stopwords = ["test", "tests", "main"]

    data_standardized = []
    for name in names_raw:
        name = snake_case_to_camel_case(name) #everything in camelcase
        names_temp = [x.lower() for x in camel_case_split(name)] #camelCase split + lowercase
        names_temp = [i for i in names_temp if i not in stopwords] #exclude stopwords
        data_standardized += names_temp

    print(data_standardized)

def main():
    create_corpus()
    represent_entities()
    query = "Optimizer that implements the Adadelta algorithm"
    print_top_5_entities()


if __name__ == "__main__":
    main()
