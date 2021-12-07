import json
import importlib

search_data = importlib.import_module("search-data")

def ground_truth_txt_to_dict():
    """
    Parse a "ground truth" txt file in a dictionary.

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
                "Query": lines[0],  # First line: name
                "Function/Class name": lines[1],  # Second line: date
                "File": lines[2]  # Third line and onwards: info
            })

    print(json.dumps(ground_truth, indent=3, default=str))
    print("Number of queries: ", len(ground_truth))

def main():
    ground_truth_txt_to_dict()

if __name__ == "__main__":
    main()