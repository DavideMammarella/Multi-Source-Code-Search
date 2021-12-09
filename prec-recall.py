import json
import importlib

search_data = importlib.import_module("search-data")

def query_search_engine(ground_truth, data):
    """
    Querying search engines given a dictionary of queries.
    Results are in a dictionary (every dictionary involve one single query with every result)
    """
    queries_list = [[d["query"], d["function/class name"], d["file"]] for d in ground_truth]
    corpus = search_data.create_corpus(data)
    top_5 = []
    for query, name, file in queries_list:
        top_5.append({
            "ground_truth_query": query,
            "ground_truth_file_name": name,
            "ground_truth_file": file,
            "top_5_FREQ": search_data.frequency_train(corpus, query)
            # ,"top_5_TF_IDF": search_data.tf_idf_train(corpus, query)
            # ,"top_5_LSI": search_data.lsi_train(corpus, query)
            #,"top_5_DOC2VEC": search_data.doc2vec_train(corpus, query)
        })
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


def get_position_from_data(expected_name, expected_file, data):
    """
    Return csv line from data.csv file given python class/method name and file.
    """
    for d in data:
        values = list(d.values())
        if all(x in values for x in [expected_name, expected_file]):
            return d["csv_line"]



def measure_precision_and_recall(ground_truth, data, query_data):
    #print(json.dumps(query_data, indent=1))
    for d in query_data: # prendi un dizionario
        expected_name = d["ground_truth_file_name"]
        expected_file = d["ground_truth_file"]
        expected_line = get_position_from_data(expected_name, expected_file, data)
        print(expected_line, expected_name, expected_file)
    # analizzi ogni dizionario separatamente e lo updati
    # top_5_FREQ_POS (se la risposta é corretta (stesso nome e file name) allora mettici la posizione che ha in lista, oppure 0
    # top_5_FREQ_prec calcoli la precisione per ogni query 1/sum(POS)
    # ribalti i dizionari, devi creare una lista per ogni search engine list_FREQ, list_TFID in cui all'interno ho dizionari
    # con query

def main():
    ground_truth_dict_list = ground_truth_txt_to_dict()
    data = search_data.extract_data()
    query_data = query_search_engine(ground_truth_dict_list, data)
    measure_precision_and_recall(ground_truth_dict_list, data, query_data)
    # ho una lista di dizionari con tutti i dati di tensorflow = data
    # ho una lista di dizionari con tutte le top 5 per ogni query (con index)
    # effettua misure analizzando questi dizionari

if __name__ == "__main__":
    main()