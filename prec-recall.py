from gensim.similarities import MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec
from gensim.corpora import MmCorpus
from gensim.models import LsiModel
import seaborn as sns
import pandas as pd
import importlib, json, os

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

search_data = importlib.import_module("search-data")
create_csv = importlib.import_module("extract-data")


# ----------------------------------------------------------------------------------------------------------------------
# EMBEDDINGs AND PLOT
# ----------------------------------------------------------------------------------------------------------------------

def plot_tsne(embedding_vectors, query_data, plotname):
    tsne = TSNE(n_components=2, verbose=0, perplexity=2, n_iter=3000)
    tsne_results = tsne.fit_transform(embedding_vectors)

    hues = []
    queries = [d["ground_truth_file_name"] for d in query_data]
    for query in queries:
        hues.extend([query] * 6)

    df = pd.DataFrame()
    df["x"] = tsne_results[:, 0]
    df["y"] = tsne_results[:, 1]
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        x="x", y="y",
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
    corpus_lsi = MmCorpus("utils/lsi/corpus_lsi.mm")
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


def get_avg_precision_recall(query_data, search_engines):
    search_engines_data = []

    for search_engine in search_engines:
        engine_precision = [d["top_5_" + search_engine + "_prec"] for d in query_data]
        engine_recall = [d["top_5_" + search_engine + "_correct"] for d in query_data]

        avg_precision = round(sum(engine_precision) / len(engine_precision), 2)
        avg_recall = round(sum(engine_recall) / len(engine_recall), 2)

        search_engines_data.append({
            "search_engine": search_engine,
            "avg_precision": avg_precision,
            "recall": avg_recall
        })

    return search_engines_data


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


def get_index_from_data_csv(expected_name, expected_file, data):
    for d in data:
        if d["name"] == expected_name and d["file"] == expected_file:
            return d["csv_line"]


def output_engines_data(search_engines_data):
    for d in search_engines_data:
        search_engine = d["search_engine"]
        avg_precision = d["avg_precision"]
        recall = d["recall"]
        print("\n>>", search_engine,
              "\n>> Average Precision: ", avg_precision,
              "\n>> Recall: ", recall)


def measure_precision_and_recall(query_data, search_engines):
    data = search_data.extract_data()

    for d in query_data:
        expected_name = d["ground_truth_file_name"]
        expected_file = d["ground_truth_file"]
        expected_line = get_index_from_data_csv(expected_name, expected_file, data)
        if expected_line is None:
            print("Ground truth [", expected_name, expected_file, "] doesn't exists in data.csv...")
            expected_line = 0
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
    output_engines_data(search_engines_data)


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
                "file": lines[2]
            })
    return ground_truth


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------

def gather_train_info():
    answer = input("Has the training already been done? (yes or no): ")

    if answer == "no":
        return True
    else:
        return False


def train_search_engines():
    print("---------------------------------------------------------------------------------\n"
          "Creating data.csv...\n")
    create_csv.silentremove("data.csv")
    create_csv.get_and_visit_files("tensorflow", "*.py")
    create_csv.write_csv()
    search_data.silentremove("utils")
    data = search_data.extract_data()
    corpus = search_data.create_corpus(data)
    print("---------------------------------------------------------------------------------\n"
          "Training...\n")
    search_data.process_corpus(corpus)
    print(">> FREQ trained!")
    search_data.tf_idf_train()
    print(">> TF IDF trained!")
    search_data.lsi_train()
    print(">> LSI trained!")
    search_data.doc2vec_train(corpus)
    print(">> DOC2VEC trained!")


def analyze_search_engines(search_engines):
    print("---------------------------------------------------------------------------------\n"
          "Computing precision and recall...")
    ground_truth_dict_list = ground_truth_txt_to_dict()
    query_data = query_search_engine(ground_truth_dict_list, search_engines)

    measure_precision_and_recall(query_data, search_engines)

    print("---------------------------------------------------------------------------------\n"
          "Plotting t_SNE...")
    lsi_embeddings = get_embedding_vectors_lsi(query_data)
    plot_tsne(lsi_embeddings, query_data, "lsi")
    doc2vec_embeddings = get_embedding_vectors_doc2vec(query_data)
    plot_tsne(doc2vec_embeddings, query_data, "doc2vec")
    print("Done! Plots can be found in Multi-Source-Code-Search folder!\n"
          "---------------------------------------------------------------------------------")


def main():
    while True:
        print("---------------------------------------------------------------------------------\n"
              "Available search engines: FREQ, TF IDF, LSI, DOC2VEC...")
        search_engines = ["FREQ", "TF_IDF", "LSI", "DOC2VEC"]
        train = gather_train_info()

        try:
            if train:
                train_search_engines()
            analyze_search_engines(search_engines)
        except Exception:
            print("Got exception!")
            train_search_engines()
            analyze_search_engines(search_engines)

        break


if __name__ == "__main__":
    main()
