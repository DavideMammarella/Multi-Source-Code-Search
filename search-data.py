# Hints:
# • Refer to the Python examples in THEO-10-gensim.pdf and to the Python scripts mentioned in the slides and available on iCorsi
# • Sort the documents in the corpus by similarity to get the top-5 entities most similar to the query for FREQ, TF-IDF, LSI
# • Use function most_similar with topn=5 to get the top-5 entities most similar to the query for Doc2Vec
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import TfidfModel, LsiModel, LdaModel
from gensim.corpora import Dictionary
from collections import defaultdict
import gensim.corpora as cp
import gensim
import requests
import pprint
import csv
import re


def print_top_5_entities():
    """
    Given a query string, for each embedding print the top-5 most similar entities
    (entity name, file name, line of code), based on cosine similarity.
    """


def doc2vec_train(corpus):
    """
    Represent entities using the doc2vec vectors with k = 300.
    :param corpus:
    """
    # processed_corpus = need corpus processing
    doc2vec = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
    print("Doc2Vec loaded!")

def lsi_train(corpus):
    """
    Represent entities using the LSI vectors with k = 300.
    :param corpus:
    """
    processed_corpus = process_corpus(corpus)
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]

    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]

    print("LSI trained!")

    query_document = "Optimizer that implements the Adadelta algorithm".lower().split()
    query_bow = dictionary.doc2bow(query_document)
    vec_lsi = lsi[tfidf[query_bow]]
    index = MatrixSimilarity(corpus_lsi)

    similarity = index[vec_lsi]
    rank = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        rank.append([score, corpus[idx]])
    print(rank[:5])

def tf_idf_train(corpus):
    """
    Represent entities using the TF-IDF vectors.
    :param corpus:
    """
    processed_corpus = process_corpus(corpus)
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = TfidfModel(corpus_bow)
    tf_idf_index = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))
    print("Tf_idf trained!")
    query_document = "Optimizer that implements the Adadelta algorithm".lower().split()
    query_bow = dictionary.doc2bow(query_document)
    similarity = tf_idf_index[query_bow]
    rank = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        rank.append([score, corpus[idx]])
    print(rank[:5])

def frequency_train(corpus):
    """
    Represent entities using the FREQ (frequency) vectors.
    :param corpus: processed corpus
    """
    processed_corpus = process_corpus(corpus)
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    frequency_index = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))
    print("Frequency trained!")
    query_document = "Optimizer that implements the Adadelta algorithm".lower().split()
    query_bow = dictionary.doc2bow(query_document)
    similarity = frequency_index[query_bow]
    rank = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        rank.append([score, corpus[idx]])
    print(rank[:5])


def process_corpus(corpus):
    frequency = defaultdict(int)
    for text in corpus:
        for token in text:
            frequency[token] += 1
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in corpus]
    return processed_corpus


def remove_stopwords(text):
    all_stopwords_gensim = STOPWORDS.union(set(["test", "tests", "main"]))
    all_stopwords_gensim = all_stopwords_gensim.difference({"get", "set"})
    tokenized_text = text.split()
    words_filtered = [word for word in tokenized_text if word not in all_stopwords_gensim]
    word = " ".join(words_filtered)
    return word


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
    :param data: ["name", "comment"]
    """
    data_standardized = []
    words_standardized = []
    words_standardized2 = []

    for name, comment in data:
        # il seguente si può ottimizzare splittando nella lista e ritornando la lista.
        words_no_underscore = underscore_split(name) + underscore_split(comment)
        # il seguente si può ottimizzare splittando nella lista e ritornando la lista.
        for word in words_no_underscore:
            words_no_camel_case = camel_case_split(word)
            words_standardized.extend(words_no_camel_case)
        # il seguente si può ottimizzare splittando nella lista e ritornando la lista.
        for word in words_standardized:
            words_no_stopwords = remove_stopwords(word)
            words_standardized2.append(words_no_stopwords)
        data_standardized += [words_standardized2]
        words_standardized = []
        words_standardized2 = []

    # print(data_standardized)
    return data_standardized


def create_corpus():
    """
    Create a corpus from the code entity names and comments.
    """
    data_raw = []
    with open("data.csv") as csv_file:
        extracted_data = csv.DictReader(csv_file, delimiter=",")
        for row in extracted_data:
            if row["comment"] != "":
                data_raw.append([row["name"], row["comment"]])

    # print(data_raw)
    return data_standardization(data_raw)


def main():
    corpus = create_corpus()
    frequency_train(corpus)
    tf_idf_train(corpus)
    lsi_train(corpus)
    doc2vec_train(corpus)
    print_top_5_entities()


if __name__ == "__main__":
    main()
