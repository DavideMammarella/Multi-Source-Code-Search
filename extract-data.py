from fnmatch import fnmatch
import ast
import csv
import os
import re

extracted_data = []
file_path_and_name = None


def write_csv():
    """
    Write all dictionaries (extracted info) in a CSV file.
    """
    global extracted_data
    file_name = "data.csv"

    with open(file_name, "w") as csv_file:
        headers = ["name", "file", "line", "type", "comment"]
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for data in extracted_data:
            writer.writerow(data)

    print("Classes: ", len([d for d in extracted_data if d.get("type") == "class"]),
          "\nFunctions: ", len([d for d in extracted_data if d.get("type") == "function"]),
          "\nMethods: ", len([d for d in extracted_data if d.get("type") == "method"]),
          "\nTotal entities:", len(extracted_data))


def comment_standardization(comment):
    """
    Standardize a comment line in order to be processed by a dictionary (remove " and \n).
    :param comment: comment line
    :return: standardized comment line
    """
    if comment is not None:
        comment = re.sub(r'["{1,}]', "", comment)
        comment = re.sub(r"[\n]", " ", comment)
        comment = re.sub(" {2,}", " ", comment)
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


def add_data(name, file, line, entity_type, comment):
    """
    Add a (Python) dictionary to a list of dictionaries.
    A dictionary can contain a Python class, Python method or Python function.
    :param name: name of the class/method/function entity
    :param file: path from repository root of the class/method/function
    :param line: line of the class/method/function entity declaration
    :param entity_type: class/method/function
    :param comment: comment line of the class/method/function entity
    """
    data = {"name": name, "file": file, "line": line, "type": entity_type, "comment": comment}
    extracted_data.append(data)


class AstVisitor(ast.NodeVisitor):
    """
    Subclass of ast.NodeVisitor, with the purpose of adding visitor methods.
    """

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit a ClassDef node (Top Level Classes and Methods).
        :param node: ast node to visit
        """
        global file_path_and_name, extracted_data
        if is_not_in_blacklist(node.name):
            comment = comment_standardization(ast.get_docstring(node))
            add_data(node.name, file_path_and_name, node.lineno, "class", comment)
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and is_not_in_blacklist(method.name):
                comment = comment_standardization(ast.get_docstring(method))
                add_data(method.name, file_path_and_name, method.lineno, "method", comment)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a FunctionDef node (Functions).
        :param node: ast node to visit
        """
        global file_path_and_name, extracted_data
        if is_not_in_blacklist(node.name):
            comment = comment_standardization(ast.get_docstring(node))
            add_data(node.name, file_path_and_name, node.lineno, "function", comment)


def get_and_visit_files(directory, file_extension):
    """
    Extract all "file_extension" file within a "directory" (and any directory below) and visit them, precisely:
    - Get all *.py files found under input directory (tensorflow) as input
    - Process every Python file to create (Python) dictionaries with names/comments of Python classes/methods/functions
    Reference for directory visiting: stackoverflow.com/questions/2909975

    :param directory: input directory
    :param file_extension: extension of file to extract in "*.extension" format
    """
    global file_path_and_name
    files_count = 0
    for path, _, files in os.walk(directory):
        for file in files:
            if fnmatch(file, file_extension):
                file_path_and_name = os.path.join(os.path.abspath(os.curdir), path, file)
                file_path = os.path.join(path, file)
                with open(file_path, 'r') as py_file:
                    files_count = files_count + 1
                    ast_of_py_file = ast.parse(py_file.read())
                    AstVisitor().visit(ast_of_py_file)
    print("Python files: ", files_count)

def main():
    root_directory = "tensorflow"
    get_and_visit_files(root_directory, "*.py")
    write_csv()


if __name__ == "__main__":
    main()
