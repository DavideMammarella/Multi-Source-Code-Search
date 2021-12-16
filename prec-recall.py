from gensim import similarities
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity, Similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, MmCorpus
import importlib
import seaborn as sns
import pandas as pd
import json

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

search_data = importlib.import_module("search-data")


# ----------------------------------------------------------------------------------------------------------------------
# EMBEDDINGs AND PLOT
# ----------------------------------------------------------------------------------------------------------------------

def plot_tsne(embedding_vectors, query_data, plotname):
    tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=3000)
    tsne_results = tsne.fit_transform(embedding_vectors)

    hues = []
    queries = [d["ground_truth_file_name"] for d in query_data]
    for query in queries:
        hues.extend([query] * 6)

    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=hues,
        palette=sns.color_palette("husl", n_colors=10),
        data=df,
        legend="full",
        alpha=1.0
    )

    plt.savefig(plotname + ".png")


def get_embedding_vectors_doc2vec(query_data):
    queries = [d["ground_truth_query"] for d in query_data]

    data = search_data.extract_data()
    corpus = search_data.create_corpus(data)
    model = Doc2Vec.load("utils/doc2vec/model")

    embeddings_doc2vec = []

    for query in queries:
        vector = model.infer_vector(query.lower().split())
        sims = model.dv.most_similar([vector], topn=5)
        sims_sorted = [doc for doc in sims]
        embeddings = [model.infer_vector(query.lower().split())]

        for index, value in sims_sorted:
            doc_corpus = corpus[index]
            embeddings.append(model.infer_vector(doc_corpus))

        for vector in embeddings:
            embeddings_doc2vec.append(vector)

    return embeddings_doc2vec


def get_embedding_vectors_lsi(query_data):
    queries = [d["ground_truth_query"] for d in query_data]

    lsi = LsiModel.load("utils/lsi/model")
    corpus_lsi = MmCorpus("utils/lsi/corpus_lsi")

    embeddings_lsi = []

    for query in queries:
        query_bow = search_data.process_query(query)
        top_5, lsi_index = search_data.lsi_query(query)
        vector = lsi[query_bow]
        sims = abs(lsi_index[vector])
        embeddings = [lsi[query_bow]]

        sims_sorted = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

        for index, value in sims_sorted[:5]:
            embeddings.append(corpus_lsi[index])

        for vector in embeddings:
            temp = []
            for _, value in vector:
                temp.append(value)
            embeddings_lsi.append(temp)

    return embeddings_lsi


# ----------------------------------------------------------------------------------------------------------------------
# PRECISION - RECALL
# ----------------------------------------------------------------------------------------------------------------------

def calculate_avg_precision_recall(search_engine_data):
    total_precision = 0
    total_recall = 0
    for prec, correct in search_engine_data:
        total_precision = total_precision + int(prec)
        total_recall = total_recall + correct
    try:
        avg_precision = total_precision / len(search_engine_data)
        avg_recall = total_recall / len(search_engine_data)
    except:
        avg_precision = 0
        avg_recall = 0

    return avg_precision, avg_recall


def extract_search_engines_data(query_data, search_engines):
    search_engines_data = []

    for search_engine in search_engines:
        engine_data = [[d["top_5_" + search_engine + "_prec"], d["top_5_" + search_engine + "_correct"]] for d in
                       query_data]
        avg_precision, recall = calculate_avg_precision_recall(engine_data)
        search_engines_data.append({
            "search_engine": search_engine,
            "avg_precision": avg_precision,
            "recall": recall
        })

    return search_engines_data


def get_correct_answers(top_5_POS):
    count = 0

    for answer in top_5_POS:
        if answer != 0:
            count = count + 1

    return count

def get_precision(top_5_POS):
    return 1/sum(top_5_POS) if sum(top_5_POS) else 0

def get_POS_list(expected_line, top_5_index):
    expected_line = int(expected_line)
    pos_list = []
    for pos, top_5_line in enumerate(top_5_index, start=1):
        if top_5_line == expected_line:
            pos_list.append(pos)
        else:
            pos_list.append(0)
    return pos_list

def get_position_from_data(expected_name, expected_file):
    """
    Return csv line from data.csv file given python class/method name and file.
    """
    data = search_data.extract_data()
    for d in data:
        values = list(d.values())
        if all(x in values for x in [expected_name, expected_file]):
            return int(d["csv_line"])

def measure_precision_and_recall(query_data, search_engines):
    for d in query_data:  # for all dictionary
        expected_name = d["ground_truth_file_name"]
        expected_file = d["ground_truth_file"]
        expected_line = get_position_from_data(expected_name, expected_file)
        print(expected_line, expected_name, expected_file)
        # expected match with the CSV file line

        for search_engine in search_engines:
            print(d["top_5_" + search_engine])
            top_5_POS = get_POS_list(expected_line, d["top_5_" + search_engine])
            print(top_5_POS)
            top_5_prec = get_precision(top_5_POS)
            top_5_correct = get_correct_answers(top_5_POS)
            d.update({"top_5_" + search_engine + "_POS": top_5_POS})
            d.update({"top_5_" + search_engine + "_prec": top_5_prec})
            d.update({"top_5_" + search_engine + "_correct": top_5_correct})

    # print(json.dumps(query_data, indent=1))
    search_engines_data = extract_search_engines_data(query_data, search_engines)
    print(json.dumps(search_engines_data, indent=1))


# ----------------------------------------------------------------------------------------------------------------------
# QUERY SEARCH ENGINE
# ----------------------------------------------------------------------------------------------------------------------

def query_search_engine(ground_truth, search_engines):
    """
    Querying search engines given a dictionary of queries.
    Results are in a dictionary (every dictionary involve one single query with every result)
    """
    queries_list = [[d["query"], d["function/class name"], d["file"]] for d in ground_truth]
    top_5 = []
    for query, name, file in queries_list:
        temp = {"ground_truth_query": query
            , "ground_truth_file_name": name
            , "ground_truth_file": file}
        for search_engine in search_engines:
            search_engine_query = getattr(search_data, search_engine.lower() + "_query")(query)
            temp.update({"top_5_" + search_engine + "": search_engine_query})
        top_5.append(temp)
    # print(json.dumps(top_5, indent=1))
    return top_5


def ground_truth_txt_to_dict():
    """
    Parse a "ground truth" txt file in a list of dictionaries.
    """
    ground_truth = []  # Blank list
    with open("ground-truth-unique.txt", "r") as file:
        sections = file.read().split("\n\n")  # Split it by double linebreaks
        for section in sections:  # Iterate through sections
            lines = section.split("\n")  # Split sections by linebreaks
            if len(lines) < 3:  # Make sure that there is the correct amount of lines
                return "ERROR!"
            ground_truth.append({
                "query": lines[0].lower(),
                "function/class name": lines[1],
                "file": lines[2].replace("../", "", 1)
            })
    return ground_truth


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------

def main():
    # search_engines = ["FREQ", "TF_IDF", "LSI", "DOC2VEC"]
    search_engines = ["FREQ"]

    ground_truth_dict_list = ground_truth_txt_to_dict()
    query_data = query_search_engine(ground_truth_dict_list, search_engines)
    measure_precision_and_recall(query_data, search_engines)
    # lsi_embeddings = get_embedding_vectors_lsi(query_data)
    # doc2vec_embeddings = get_embedding_vectors_doc2vec(query_data)
    # plot_tsne(doc2vec_embeddings, query_data, "doc2vec")
    # plot_tsne(lsi_embeddings, query_data, "lsi")


if __name__ == "__main__":
    main()
