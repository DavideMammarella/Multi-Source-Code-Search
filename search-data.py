from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, MmCorpus
from collections import defaultdict
from pathlib import Path
import shutil, gensim, csv, re, os, errno


def print_top_5_entities(data, top_5_index, search_engine):
    """
    Given a query string, for each embedding print the top-5 most similar entities
    (entity name, file name, line of code), based on cosine similarity.
    """
    print("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n",
          search_engine, " top-5 most similar entities \n"
                         "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    for index, elem in enumerate(top_5_index, start=1):
        print(index, "\tPython class:\t", data[elem]["name"],
              "\n\tFile:\t", data[elem]["file"],
              "\n\tLine:\t", data[elem]["line"],
              "\n---------------------------------------------------------------------------------")


def doc2vec_query(query):
    model = Doc2Vec.load("utils/doc2vec/model")

    vector = model.infer_vector(query.lower().split())
    sims = model.dv.most_similar([vector], topn=5)

    list_top_5_index = []
    for label, index in [("FIRST", 0), ("SECOND", 1), ("THIRD", 2), ("FOURTH", 3), ("FIFTH", 4)]:
        list_top_5_index.append(sims[index][0])

    return list_top_5_index


def read_corpus(corpus, tokens_only=False):
    for i, line in enumerate(corpus):
        line = " ".join(line)
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def doc2vec_train(corpus):
    Path("utils/doc2vec").mkdir(parents=True, exist_ok=True)

    train_corpus = list(read_corpus(corpus))

    if os.path.exists("utils/doc2vec/model"):
        pass
    else:
        model = Doc2Vec(vector_size=300, min_count=2, epochs=40)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("utils/doc2vec/model")


def get_top_5_index(similarity):
    list_top_5_index = []
    for idx, score in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True):
        list_top_5_index.append(idx)
    return list_top_5_index[:5]


def lsi_query(query):
    tfidf = TfidfModel.load("utils/tf_idf/model")
    lsi = LsiModel.load("utils/lsi/model")
    corpus_lsi = MmCorpus("utils/lsi/corpus_lsi.mm")

    query_bow = process_query(query)
    vec_lsi = lsi[tfidf[query_bow]]

    lsi_index = MatrixSimilarity(corpus_lsi)
    lsi_index.save("utils/lsi/lsi.index")
    similarity = lsi_index[vec_lsi]

    top_5_index = get_top_5_index(similarity)
    return top_5_index


def lsi_train():
    corpus_bow = MmCorpus("utils/corpus.mm")
    dictionary = Dictionary.load("utils/dictionary.dict")
    Path("utils/tf_idf").mkdir(parents=True, exist_ok=True)
    Path("utils/lsi").mkdir(parents=True, exist_ok=True)

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

    MmCorpus.serialize("utils/lsi/corpus_lsi.mm", corpus_lsi)


def tf_idf_query(query):
    query_bow = process_query(query)

    tf_idf_index = SparseMatrixSimilarity.load("utils/tf_idf/tf_idf.index")
    sims = tf_idf_index[query_bow]

    top_5_index = get_top_5_index(sims)
    return top_5_index


def tf_idf_train():
    corpus_bow = MmCorpus("utils/corpus.mm")
    dictionary = Dictionary.load("utils/dictionary.dict")
    Path("utils/tf_idf").mkdir(parents=True, exist_ok=True)

    if os.path.exists("utils/tf_idf/model"):
        tfidf = TfidfModel.load("utils/tf_idf/model")
    else:
        tfidf = TfidfModel(corpus_bow)
        tfidf.save("utils/tf_idf/model")

    tf_idf_index = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))
    tf_idf_index.save("utils/tf_idf/tf_idf.index")


def freq_query(query):
    corpus_bow = MmCorpus("utils/corpus.mm")
    dictionary = Dictionary.load("utils/dictionary.dict")

    frequency_index = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))

    query_bow = process_query(query)
    similarity = frequency_index[query_bow]

    top_5_index = get_top_5_index(similarity)

    return top_5_index


def process_query(query):
    dictionary = Dictionary.load("utils/dictionary.dict")
    query_bow = dictionary.doc2bow(query.lower().split())
    return query_bow


def process_corpus(corpus):
    """
    Remove words that appear only once.
    """
    frequency = defaultdict(int)
    for text in corpus:
        for token in text:
            frequency[token] += 1
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in corpus]

    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]

    Path("utils").mkdir(parents=True, exist_ok=True)
    dictionary.save("utils/dictionary.dict")
    MmCorpus.serialize("utils/corpus.mm", corpus_bow)


def remove_method_stopwords(text):
    stopwords = {"test", "tests", "main"}
    tokenized_text = text.split()
    words_filtered = [word for word in tokenized_text if word not in stopwords]

    word = " ".join(words_filtered)
    return word


def name_standardization(data, type):
    """
    Standardize a method/comment name:
    1. split entity names (by CamelCase and underscore)
    2. filter stopwords = {test, tests, main}
    3. convert all words to lowercase
    :param data: ["name", "comment"]
    """
    words = data.replace("_", " ")  # split by underscore
    words = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", words)  # split by CamelCase
    if type == "method":
        words = remove_method_stopwords(words)
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
            "csv_line": row["csv_line"],
            "name": name_standardization(row["name"], "name"),
            "comment": name_standardization(row["comment"], "comment")
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
                "csv_line": index,
                "name": row["name"],
                "file": row["file"],
                "line": row["line"],
                "type": row["type"],
                "comment": row["comment"]
            })
    return data_raw


def silentremove(filename):
    dirpath = Path(filename)
    try:
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def main():
    data = extract_data()
    corpus = create_corpus(data)
    process_corpus(corpus)
    query = "Optimizer that implements the Adadelta algorithm"

    tf_idf_train()
    lsi_train()
    doc2vec_train(corpus)

    freq_top_5 = freq_query(query)
    tf_idf_top_5 = tf_idf_query(query)
    lsi_top_5 = lsi_query(query)
    doc2vec_top_5 = doc2vec_query(query)

    print_top_5_entities(data, freq_top_5, "FREQ")
    print_top_5_entities(data, tf_idf_top_5, "TF IDF")
    print_top_5_entities(data, lsi_top_5, "LSI")
    print_top_5_entities(data, doc2vec_top_5, "DOC2VEC")


if __name__ == "__main__":
    silentremove("utils")
    main()
