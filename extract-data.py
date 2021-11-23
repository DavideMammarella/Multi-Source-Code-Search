import os
from fnmatch import fnmatch
import ast
import csv
import re

extracted_data = []
file_path_and_name = None


def write_csv():
    """
    Write all dictionaries in a CSV file.
    """
    global extracted_data
    file_name = "data.csv"
    with open(file_name, "w") as csv_file:
        headers = ["name", "file", "line", "type", "comment"]
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for data in extracted_data:
            writer.writerow(data)
    print("Total number of row:", len(extracted_data))


def comment_standardization(comment):
    """
    Standardize a comment line in order to be processed by a dictionary (remove " and \n).
    :param comment: comment line
    :return: standardized comment line
    """
    if comment is not None:
        comment = re.sub(r'["\n]', " ", comment)
    return comment


def is_not_in_blacklist(node_name):
    """
    Check if a node (class/method/function) name is in the blacklist.
    A name is blacklisted if:
    - is "main"
    - starts with with "_"
    - contains the word "test" (including all case variants, such as "TEST")
    :param node_name: name of the node
    :return: True if the name is not in the black list, False otherwise
    """
    caseless_node_name = node_name.casefold()
    if (node_name != "main") and not (node_name.startswith("_")) and ("test" not in caseless_node_name):
        return True


def add_data(name, file, line, type, comment):
    """
    Add a (Python) dictionary to a list of dictionaries.
    A dictionary can contain a Python class, Python method or Python function.
    :param name: name of the class/method/function entity
    :param file: path from repository root of the class/method/function
    :param line: line of the class/method/function entity declaration
    :param type: class/method/function
    :param comment: comment line of the class/method/function entity
    """
    data = {}
    data["name"] = name
    data["file"] = file
    data["line"] = line
    data["type"] = type
    data["comment"] = comment
    extracted_data.append(data)


class AstVisitor(ast.NodeVisitor):
    """
    Subclass of ast.NodeVisitor, with the purpose of adding visitor methods.
    """

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit a ClassDef node (Top Level Classes).
        :param node: ast node to visit
        """
        if is_not_in_blacklist(node.name):
            global file_path_and_name, extracted_data
            comment = comment_standardization(ast.get_docstring(node))
            add_data(node.name, file_path_and_name, node.lineno, "class", comment)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a FunctionDef node (Functions and Methods).
        :param node: ast node to visit
        """
        if is_not_in_blacklist(node.name):
            global file_path_and_name, extracted_data
            comment = comment_standardization(ast.get_docstring(node))
            add_data(node.name, file_path_and_name, node.lineno, "functions", comment)


def main():
    """
    Extract data:
    - Get all *.py files found under input directory (tensorflow) as input
    - Process every Python file to create a CSV file with names/comments of Python classes, methods, functions
    - Save the extracted info into a CSV file
    """
    root = "tensorflow"
    pattern = "*.py"

    # https://stackoverflow.com/questions/2909975/python-list-directory-subdirectory-and-files (answer 2)
    for path, subdirs, files in os.walk(root):
        for file in files:
            if fnmatch(file, pattern):
                global file_path_and_name
                file_path_and_name = os.path.join(path, file)
                with open(file_path_and_name, 'r') as py_file:
                    ast_of_py_file = ast.parse(py_file.read())
                    AstVisitor().visit(ast_of_py_file)

    write_csv()


if __name__ == "__main__":
    main()
