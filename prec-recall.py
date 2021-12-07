import json
import importlib

search_data = importlib.import_module("search-data")

def query_search_engine(ground_truth):
    """
    Querying search engines given a dictionary of queries.
    :return:
    """
    queries_list = [d["query"] for d in ground_truth]
    corpus = search_data.create_corpus()
    top_5 = []
    top_5_freq = []
    for query in queries_list:
        top_5_freq.append({  # Add a dictionary to the data with:
            "search engine": "FREQ",
            "query": query,  # First line: name
            "top 5": search_data.frequency_train(corpus, query)
        })
        # top_5_tf_idf = search_data.tf_idf_train(corpus, query)
        # top_5_lsi = search_data.lsi_train(corpus, query)
        # top_5_doc2vec = search_data.doc2vec_train(corpus, query)
    print(json.dumps(top_5_freq, indent=2, default=str))

def ground_truth_txt_to_dict():
    """
    Parse a "ground truth" txt file in a list of dictionaries.

    File structure (name must be ground-truth-unique.txt):
    SECTION
    LINE BREAK
    SECTION
    LINE BREAK
    ...

    Each section consists of:
    query
    function/class name
    file path

    Reference: stackoverflow.com/questions/63406077
    """
    ground_truth = []  # Blank list
    with open("ground-truth-unique.txt", "r") as file:  # Open file
        sections = file.read().split("\n\n")  # Split it by double linebreaks
        for section in sections:  # Iterate through sections
            lines = section.split("\n")  # Split sections by linebreaks
            if len(lines) < 3:  # Make sure that there is the correct amount of lines
                return "ERROR!"
            ground_truth.append({  # Add a dictionary to the data with:
                "query": lines[0].lower(),  # First line: name
                "function/class name": lines[1],  # Second line: date
                "file": lines[2]  # Third line and onwards: info
            })

    return ground_truth
    #print(json.dumps(ground_truth, indent=3, default=str))
    #print("Number of queries: ", len(ground_truth))

def main():
    ground_truth_dict_list = ground_truth_txt_to_dict()
    query_search_engine(ground_truth_dict_list)

if __name__ == "__main__":
    main()