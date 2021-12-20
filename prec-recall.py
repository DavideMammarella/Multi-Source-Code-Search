from gensim.similarities import MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec
from gensim.corpora import MmCorpus
from gensim.models import LsiModel
import seaborn as sns
import pandas as pd
import importlib
import json
import os

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
    lsi_index = MatrixSimilarity.load("utils/lsi/lsi.index")

    embeddings_lsi = []

    for query in queries:
        query_bow = search_data.process_query(query)
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

# checked
def get_avg_precision_recall(query_data, search_engines):
    search_engines_data = []

    for search_engine in search_engines:
        engine_precision = [d["top_5_" + search_engine + "_prec"] for d in query_data]
        engine_recall = [d["top_5_" + search_engine + "_correct"] for d in query_data]

        avg_precision = sum(engine_precision) / len(engine_precision)
        avg_recall = sum(engine_recall) / len(engine_recall)

        search_engines_data.append({
            "search_engine": search_engine,
            "avg_precision": avg_precision,
            "recall": avg_recall
        })

    return search_engines_data


# checked
def get_POS_list(expected_line, top_5, data):
    expected_line = int(expected_line)

    top_5_index = []
    for elem in top_5:
        top_5_index.append(data[elem]["csv_line"])

    # PRINT THIS TO CHECK LINES AND ENTITIES
    # print("Expected line:\t", expected_line)
    # print("Top 5 index\t:", top_5_index)
    # search_data.print_top_5_entities(data, top_5, "FREQ")

    pos_list = []
    for index, top_5_line in enumerate(top_5_index, start=1):
        if top_5_line == expected_line:
            pos_list.append(index)
        else:
            pos_list.append(0)
    return pos_list


# checked
def get_index_from_data_csv(expected_name, expected_file, data):
    for d in data:
        if d["name"] == expected_name and d["file"] == expected_file:
            return d["csv_line"]


#checked
def measure_precision_and_recall(query_data, search_engines):
    data = search_data.extract_data()

    for d in query_data:
        expected_name = d["ground_truth_file_name"]
        expected_file = d["ground_truth_file"]
        expected_line = get_index_from_data_csv(expected_name, expected_file, data)
        d.update({"ground_truth_line": expected_line})

        for search_engine in search_engines:
            POS = get_POS_list(expected_line, d["top_5_" + search_engine],
                               data)  # replace correct answer with it's position
            precision = 1 / sum(POS) if sum(POS) else 0  # 1/POS if POS not 0
            correct_answers = 5 - (POS.count(0))  # max num of elements - occurrences of 0
            d.update({"top_5_" + search_engine + "_POS": POS})
            d.update({"top_5_" + search_engine + "_prec": precision})
            d.update({"top_5_" + search_engine + "_correct": correct_answers})

    # PRINT THIS FOR TOP_5 POS/PRECISION/CORRECT ANSWERS
    # print(json.dumps(query_data, indent=1))

    search_engines_data = get_avg_precision_recall(query_data, search_engines)
    # PRINT THIS FOR AVERAGE PRECISION/RECALL ABOVE ALL QUERIES
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

            src_path = lines[2].replace("../", "", 1)
            abs_path = os.path.join(os.path.abspath(os.curdir), src_path)
            ground_truth.append({
                "query": lines[0].lower(),
                "function/class name": lines[1],
                "file": abs_path
            })
    return ground_truth


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------

def main():
    #search_engines = ["FREQ", "TF_IDF", "LSI", "DOC2VEC"]
    search_engines = ["FREQ"]

    ground_truth_dict_list = ground_truth_txt_to_dict()
    query_data = query_search_engine(ground_truth_dict_list, search_engines)

    measure_precision_and_recall(query_data, search_engines)

    #lsi_embeddings = get_embedding_vectors_lsi(query_data)
    #doc2vec_embeddings = get_embedding_vectors_doc2vec(query_data)
    #plot_tsne(doc2vec_embeddings, query_data, "doc2vec")
    #plot_tsne(lsi_embeddings, query_data, "lsi")


if __name__ == "__main__":
    main()
