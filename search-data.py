from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, MmCorpus
from collections import defaultdict
from pathlib import Path
import gensim
import csv
import re
import os


def print_top_5_entities(data, top_5_index, search_engine):
    """
    Given a query string, for each embedding print the top-5 most similar entities
    (entity name, file name, line of code), based on cosine similarity.
    """
    print("\n============================================================\n",
          search_engine, "     ||||   TOP-5 MOST SIMILAR ENTITIES (ASC)\n"
                         "============================================================")
    for elem in top_5_index:
        print("Python class:", data[elem]["name"],
              "\nFile:", data[elem]["file"],
              "\nLine:", data[elem]["line"],
              "\n------------------------------------------------------------")


def read_corpus(corpus):
    for i, line in enumerate(corpus):
        yield gensim.models.doc2vec.TaggedDocument(line, [i])


def doc2vec_query(query):
    model = Doc2Vec.load("utils/doc2vec/model")
    vector = model.infer_vector(query.lower().split())
    sims = model.dv.most_similar([vector], topn=5)

    list_top_5_index = []
    for label, index in [("FIRST", 0), ("SECOND", 1), ("THIRD", 2), ("FOURTH", 3), ("FIFTH", 4)]:
        list_top_5_index.append(sims[index][0])
    return list_top_5_index


def doc2vec_train(corpus):
    Path("utils/doc2vec").mkdir(parents=True, exist_ok=True)
    train_corpus = list(read_corpus(corpus))

    if os.path.exists("utils/doc2vec/model"):
        model = Doc2Vec.load("utils/doc2vec/model")
    else:
        model = Doc2Vec(vector_size=300, min_count=2, epochs=40)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("utils/doc2vec/model")

    return model


def lsi_query(query):
    tfidf = TfidfModel.load("utils/tf_idf/model")
    lsi = LsiModel.load("utils/lsi/model")
    corpus_lsi = MmCorpus("utils/corpus_lsi")

    query_bow = process_query(query)

    vec_lsi = lsi[tfidf[query_bow]]
    index = MatrixSimilarity(corpus_lsi)

    similarity = index[vec_lsi]

    top_5_index = get_top_5_index(similarity)
    return top_5_index


def lsi_train(corpus):
    corpus_bow = MmCorpus("utils/corpus")
    Path("utils/tf_idf").mkdir(parents=True, exist_ok=True)
    Path("utils/lsi").mkdir(parents=True, exist_ok=True)
    dictionary = Dictionary.load("utils/dictionary")

    if os.path.exists("utils/tf_idf/model"):
        tfidf = TfidfModel.load("utils/tf_idf/model")
    else:
        tfidf = TfidfModel(corpus_bow)
        tfidf.save("utils/tf_idf/model")
    corpus_tf_idf = tfidf[corpus_bow]

    if os.path.exists("utils/lsi/model"):
        lsi = LsiModel.load("utils/lsi/model")
    else:
        lsi = LsiModel(corpus_tf_idf, id2word=dictionary, num_topics=300)
        lsi.save("utils/lsi/model")
    corpus_lsi = lsi[corpus_tf_idf]

    MmCorpus.serialize("utils/corpus_lsi", corpus_lsi)  # save corpus


def tf_idf_query(query):
    query_bow = process_query(query)

    tf_idf_index = SparseMatrixSimilarity.load("utils/tf_idf/index")
    sims = tf_idf_index[query_bow]

    top_5_index = get_top_5_index(sims)
    return top_5_index


def tf_idf_train(corpus):
    corpus_bow = MmCorpus("utils/corpus")
    dictionary = Dictionary.load("utils/dictionary")
    Path("utils/tf_idf").mkdir(parents=True, exist_ok=True)

    if os.path.exists("utils/tf_idf/model"):
        tfidf = TfidfModel.load("utils/tf_idf/model")
    else:
        tfidf = TfidfModel(corpus_bow)
        tfidf.save("utils/tf_idf/model")

    tf_idf_index = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))
    tf_idf_index.save("utils/tf_idf/index")

    return tfidf


def frequency_query(query):
    corpus_bow = MmCorpus("utils/corpus")
    dictionary = Dictionary.load("utils/dictionary")

    frequency_index = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))
    query_bow = process_query(query)

    similarity = frequency_index[query_bow]

    top_5_index = get_top_5_index(similarity)
    return top_5_index


def get_top_5_index(similarity):
    list_top_5_index = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        list_top_5_index.append(idx)
    return list_top_5_index[:5]


def process_query(query):
    dictionary = Dictionary.load("utils/dictionary")
    query_bow = dictionary.doc2bow(query.lower().split())
    return query_bow


def process_corpus(corpus):
    frequency = defaultdict(int)
    for text in corpus:
        for token in text:
            frequency[token] += 1
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in corpus]
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]

    Path("utils").mkdir(parents=True, exist_ok=True)
    dictionary.save("utils/dictionary")
    MmCorpus.serialize("utils/corpus", corpus_bow)

    return corpus


def remove_stopwords(text):
    """
    STOPWORDS are from Gensim.
    """
    edited_stopwords = STOPWORDS.union(set(["test", "tests", "main"]))
    edited_stopwords = edited_stopwords.difference(
        {"False", "None", "True", "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
         "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
         "pass", "raise", "return", "try", "while", "with", "yield", "ArithmeticError", "AssertionError",
         "AttributeError", "BaseException", "BlockingIOError", "BrokenPipeError", "BufferError", "BytesWarning",
         "ChildProcessError", "ConnectionAbortedError", "ConnectionError", "ConnectionRefusedError",
         "ConnectionResetError", "DeprecationWarning", "EOFError", "Ellipsis", "EnvironmentError", "Exception", "False",
         "FileExistsError", "FileNotFoundError", "FloatingPointError", "FutureWarning", "GeneratorExit", "IOError",
         "ImportError", "ImportWarning", "IndentationError", "IndexError", "InterruptedError", "IsADirectoryError",
         "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError", "NameError", "None", "NotADirectoryError",
         "NotImplemented", "NotImplementedError", "OSError", "OverflowError", "PendingDeprecationWarning",
         "PermissionError", "ProcessLookupError", "RecursionError", "ReferenceError", "ResourceWarning", "RuntimeError",
         "RuntimeWarning", "StopAsyncIteration", "StopIteration", "SyntaxError", "SyntaxWarning", "SystemError",
         "SystemExit", "TabError", "TimeoutError", "True", "TypeError", "UnboundLocalError", "UnicodeDecodeError",
         "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError", "UnicodeWarning", "UserWarning", "ValueError",
         "Warning", "ZeroDivisionError", "_", "build_class", "debug", "doc", "import", "loader", "name", "package",
         "spec", "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod",
         "compile", "complex", "copyright", "credits", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
         "exit", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex", "id",
         "input", "int", "isinstance", "issubclass", "iter", "len", "license", "list", "locals", "map", "max",
         "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print", "property", "quit", "range",
         "repr", "reversed", "round", "get", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
         "tuple", "type", "vars", "zip"})
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
    for index, row in enumerate(data):
        data_name_comment_standardized.append({
            "csv_line": str(index),
            "name": method_name_standardization(row["name"]),
            "comment": comment_standardization(row["comment"])
        })

    corpus = []
    for dict in data_name_comment_standardized:
        name_and_comment = dict["name"] + dict["comment"]
        corpus.append(name_and_comment)

    return corpus


def extract_data():
    """
    CSV to DICT with indexing
    """
    data_raw = []
    with open("data.csv") as csv_file:
        extracted_data = csv.DictReader(csv_file, delimiter=",")  # ordered (Py>3.6)
        for index, row in enumerate(extracted_data):
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
    corpus = create_corpus(data)
    process_corpus(corpus)
    query = "Optimizer that implements the Adadelta algorithm"

    tf_idf_train(corpus)
    lsi_train(corpus)
    doc2vec_train(corpus)

    freq_top_5 = frequency_query(query)
    tf_idf_top_5 = tf_idf_query(query)
    lsi_top_5 = lsi_query(query)
    doc2vec_top_5 = doc2vec_query(query)

    print_top_5_entities(data, freq_top_5, "FREQ")
    print_top_5_entities(data, tf_idf_top_5, "TF IDF")
    print_top_5_entities(data, lsi_top_5, "LSI")
    print_top_5_entities(data, doc2vec_top_5, "DOC2VEC")


if __name__ == "__main__":
    main()
