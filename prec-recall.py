import json
import importlib

search_data = importlib.import_module("search-data")


def query_search_engine(ground_truth):
    """
    Querying search engines given a dictionary of queries.
    Results are in a dictionary (every dictionary involve one single query with every result)
    """
    queries_list = [[d["query"], d["function/class name"], d["file"]] for d in ground_truth]
    top_5 = []
    for query, name, file in queries_list:
        top_5.append({
            "ground_truth_query": query,
            "ground_truth_file_name": name,
            "ground_truth_file": file,
            "top_5_FREQ": search_data.frequency_query(query)
            , "top_5_TF_IDF": search_data.tf_idf_query(query)
            , "top_5_LSI": search_data.lsi_query(query)
            , "top_5_DOC2VEC": search_data.doc2vec_query(query)
        })
    #print(json.dumps(top_5, indent=1))
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
            return int(d["csv_line"])


def get_POS_list(expected_line, top_5_index):
    expected_line = int(expected_line)
    top_5_index = list(map(int, top_5_index))
    pos_list = []
    for index, item in enumerate(top_5_index, start=1):
        if item == expected_line:
            pos_list.append(index)
        else:
            pos_list.append(0)
    return pos_list


def get_precision(top_5_POS):
    top_5_POS = list(map(int, top_5_POS))

    try:
        result = 1 / sum(top_5_POS)
    except:
        result = 0

    return result


def get_correct_answer(top_5_POS):
    top_5_POS = list(map(int, top_5_POS))
    count = 0

    for answer in top_5_POS:
        if answer != 0:
            count = count + 1

    return count


def update_query_data(query_data, data):
    # populate dictionary with precision and recall
    for d in query_data:  # for all dictionary
        expected_name = d["ground_truth_file_name"]
        expected_file = d["ground_truth_file"]
        expected_line = get_position_from_data(expected_name, expected_file, data)
        # print(expected_line, expected_name, expected_file)

        top_5_FREQ_POS = get_POS_list(expected_line, d["top_5_FREQ"])
        top_5_FREQ_prec = get_precision(top_5_FREQ_POS)
        top_5_FREQ_correct = get_correct_answer(top_5_FREQ_POS)
        d.update({"top_5_FREQ": top_5_FREQ_POS})
        d.update({"top_5_FREQ_prec": top_5_FREQ_prec})
        d.update({"top_5_FREQ_correct": top_5_FREQ_correct})

        top_5_TF_IDF_POS = get_POS_list(expected_line, d["top_5_TF_IDF"])
        top_5_TF_IDF_prec = get_precision(top_5_TF_IDF_POS)
        top_5_TF_IDF_correct = get_correct_answer(top_5_TF_IDF_POS)
        d.update({"top_5_TF_IDF": top_5_TF_IDF_POS})
        d.update({"top_5_TF_IDF_prec": top_5_TF_IDF_prec})
        d.update({"top_5_TF_IDF_correct": top_5_TF_IDF_correct})

        top_5_LSI_POS = get_POS_list(expected_line, d["top_5_LSI"])
        top_5_LSI_prec = get_precision(top_5_LSI_POS)
        top_5_LSI_correct = get_correct_answer(top_5_LSI_POS)
        d.update({"top_5_LSI": top_5_LSI_POS})
        d.update({"top_5_LSI_prec": top_5_LSI_prec})
        d.update({"top_5_LSI_correct": top_5_LSI_correct})

        top_5_DOC2VEC_POS = get_POS_list(expected_line, d["top_5_DOC2VEC"])
        top_5_DOC2VEC_prec = get_precision(top_5_DOC2VEC_POS)
        top_5_DOC2VEC_correct = get_correct_answer(top_5_DOC2VEC_POS)
        d.update({"top_5_DOC2VEC": top_5_DOC2VEC_POS})
        d.update({"top_5_DOC2VEC_prec": top_5_DOC2VEC_prec})
        d.update({"top_5_DOC2VEC_correct": top_5_DOC2VEC_correct})

    #print(json.dumps(query_data, indent=1))
    return query_data


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


def extract_search_engines_data(query_data):
    search_engines_data = []

    freq_data = [[d["top_5_FREQ_prec"], d["top_5_FREQ_correct"]] for d in query_data]
    avg_precision, recall = calculate_avg_precision_recall(freq_data)
    search_engines_data.append({
        "search_engine": "FREQ",
        "avg_precision": avg_precision,
        "recall": recall
    })

    tf_idf_data = [[d["top_5_TF_IDF_prec"], d["top_5_TF_IDF_correct"]] for d in query_data]
    avg_precision, recall = calculate_avg_precision_recall(tf_idf_data)
    search_engines_data.append({
        "search_engine": "TF_IDF",
        "avg_precision": avg_precision,
        "recall": recall
    })

    lsi_data = [[d["top_5_LSI_prec"], d["top_5_LSI_correct"]] for d in query_data]
    avg_precision, recall = calculate_avg_precision_recall(lsi_data)
    search_engines_data.append({
        "search_engine": "LSI",
        "avg_precision": avg_precision,
        "recall": recall
    })

    doc2vec_data = [[d["top_5_DOC2VEC_prec"], d["top_5_DOC2VEC_correct"]] for d in query_data]
    avg_precision, recall = calculate_avg_precision_recall(doc2vec_data)
    search_engines_data.append({
        "search_engine": "DOC2VEC",
        "avg_precision": avg_precision,
        "recall": recall
    })

    return search_engines_data


def measure_precision_and_recall(ground_truth, data, query_data):
    query_data = update_query_data(query_data, data)
    search_engines_data = extract_search_engines_data(query_data)
    print(json.dumps(search_engines_data, indent=1))


def main():
    ground_truth_dict_list = ground_truth_txt_to_dict()
    data = search_data.extract_data()
    query_data = query_search_engine(ground_truth_dict_list)
    measure_precision_and_recall(ground_truth_dict_list, data, query_data)


if __name__ == "__main__":
    main()
