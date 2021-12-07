# Hints:
# • Sort the documents in the corpus by similarity to get the top-5 entities most similar to the query for FREQ, TF-IDF, LSI
# • Use function most_similar with topn=5 to get the top-5 entities most similar to the query for Doc2Vec
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from collections import defaultdict
import gensim
import csv
import re
import json


def print_top_5_entities():
    """
    Given a query string, for each embedding print the top-5 most similar entities
    (entity name, file name, line of code), based on cosine similarity.
    """


def doc2vec_train(corpus, query):
    """
    Represent entities using the doc2vec vectors with k = 300.
    :param corpus:
    """
    # processed_corpus = need corpus processing
    # use most_similar with topn=5 per le top 5 similar queries
    doc2vec = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
    print("Doc2Vec loaded!")


def lsi_train(corpus, query):
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

    query_bow = dictionary.doc2bow(query.lower().split())
    vec_lsi = lsi[tfidf[query_bow]]
    index = MatrixSimilarity(corpus_lsi)

    similarity = index[vec_lsi]
    rank = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        rank.append([score, corpus[idx]])
    # print(rank[:5])


def tf_idf_train(corpus, query):
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
    query_bow = dictionary.doc2bow(query.lower().split())
    similarity = tf_idf_index[query_bow]
    rank = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        rank.append([score, corpus[idx]])
    # print(rank[:5])


def frequency_train(corpus, query):
    """
    Represent entities using the FREQ (frequency) vectors.
    :param corpus: processed corpus
    """
    processed_corpus = process_corpus(corpus)
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    frequency_index = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))
    print("Frequency trained!")

    query_bow = dictionary.doc2bow(query.lower().split())
    similarity = frequency_index[query_bow]

    rank = []
    # return score and corpus
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        rank.append([score, corpus[idx]])
    return rank[:5]


def process_corpus(corpus):
    frequency = defaultdict(int)
    for text in corpus:
        for token in text:
            frequency[token] += 1
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in corpus]
    return processed_corpus


def remove_stopwords(text):
    """
    STOPWORDS are from Gensim.
    """
    edited_stopwords = STOPWORDS.union(set(["test", "tests", "main"]))
    edited_stopwords = edited_stopwords.difference({"False", "None", "True", "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "BlockingIOError", "BrokenPipeError", "BufferError", "BytesWarning", "ChildProcessError", "ConnectionAbortedError", "ConnectionError", "ConnectionRefusedError", "ConnectionResetError", "DeprecationWarning", "EOFError", "Ellipsis", "EnvironmentError", "Exception", "False", "FileExistsError", "FileNotFoundError", "FloatingPointError", "FutureWarning", "GeneratorExit", "IOError", "ImportError", "ImportWarning", "IndentationError", "IndexError", "InterruptedError", "IsADirectoryError", "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError", "NameError", "None", "NotADirectoryError", "NotImplemented", "NotImplementedError", "OSError", "OverflowError", "PendingDeprecationWarning", "PermissionError", "ProcessLookupError", "RecursionError", "ReferenceError", "ResourceWarning", "RuntimeError", "RuntimeWarning", "StopAsyncIteration", "StopIteration", "SyntaxError", "SyntaxWarning", "SystemError", "SystemExit", "TabError", "TimeoutError", "True", "TypeError", "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError", "UnicodeWarning", "UserWarning", "ValueError", "Warning", "ZeroDivisionError", "_", "build_class", "debug", "doc", "import", "loader", "name", "package", "spec", "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod", "compile", "complex", "copyright", "credits", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec", "exit", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len", "license", "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print", "property", "quit", "range", "repr", "reversed", "round", "get", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip"})
    tokenized_text = text.split()
    words_filtered = [word for word in tokenized_text if word not in edited_stopwords]
    word = " ".join(words_filtered)
    return word


def comment_standardization(data):
    """
    Standardize a comment:
    1. split entity names (by CamelCase and underscore)
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    4. add additional filters for comments
    :param data: ["name", "comment"]
    """
    words = data.replace("_", " ")  # split by underscore
    words = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", words)  # split by CamelCase
    words = remove_stopwords(words)  # remove stopwords
    words = words.lower()  # convert to lowercase
    words = re.sub(r"\((.*?)\)", r"", words)  # remove text in brackets (delete examples)
    words = re.sub(r"(->)+", r" ", words)  # remove ->
    words = re.sub(r"(\s+)", r" ", words)  # replace multiple whitespaces with a single one
    words = re.sub(r"( \. )+", r" ", words)  # replace dot with  double spaces
    words = words.split(". ")

    return words


def method_name_standardization(data):
    """
    Standardize a method name:
    1. split entity names (by CamelCase and underscore)
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    :param data: ["name", "comment"]
    """
    words = data.replace("_", " ")  # split by underscore
    words = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", words)  # split by CamelCase
    words = remove_stopwords(words)  # remove stopwords
    words = words.lower()  # convert to lowercase
    words = words.split()  # create a list of separated words

    return words


def create_corpus(data):
    """
    Create a corpus from the code entity names and comments.
    """
    data_name_comment_standardized = []
    for index, row in enumerate(data, start=2):
        data_name_comment_standardized.append({
            "csv_line": str(index),
            "name": method_name_standardization(row["name"]),
            "comment": comment_standardization(row["comment"])
        })
    # print(json.dumps(data_name_comment_standardized[58], indent=3, default=str))
    # print(json.dumps(data_name_comment_standardized[40], indent=3, default=str))
    # print(json.dumps(data_name_comment_standardized[63], indent=3, default=str))
    # fino a qui ho gli index

    # crea corpus, estrai [[nome1, commento1],[nome2, commento2]]

    return []


def extract_data():
    """
    CSV to DICT with indexing
    """
    data_raw = []
    with open("data.csv") as csv_file:
        extracted_data = csv.DictReader(csv_file, delimiter=",")  # ordered (Py>3.6)
        for index, row in enumerate(extracted_data, start=2):
            data_raw.append({
                "csv_line": str(index),
                "name": row["name"],
                "file": row["file"],
                "line": row["line"],
                "type": row["type"],
                "comment": row["comment"]
            })
    return data_raw


def main():
    data = extract_data()
    # print(json.dumps(data[-1:], indent=3, default=str))
    corpus = create_corpus(data)
    # print(corpus[-1:])
    # query = "Optimizer that implements the Adadelta algorithm"
    # frequency_train(corpus, query)
    # tf_idf_train(corpus, query)
    # lsi_train(corpus, query)
    # doc2vec_train(corpus, query)
    # print_top_5_entities()


if __name__ == "__main__":
    main()
