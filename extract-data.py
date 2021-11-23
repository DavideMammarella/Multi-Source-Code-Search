import os
from fnmatch import fnmatch
import ast

# Save the extracted info into a CSV file.
# TODO (ASK) inner functions/methods can be skipped by just avoiding to trigger a recursive generic_visit call.

extracted_data = []
file_path_and_name = None

def add_data(name, file, line, type, comment):
    """
    Add a (Python) dictionary to a list of dictionaries.
    A dictionary can contain a Python class, Python method or Python function.
    :param name: name of the class/method/function
    :param file: path from repository root of the class/method/function
    :param line: line of the class/method/function declaration
    :param type: class/method/function
    :param comment: comment line of the class/method/function
    """
    data={}
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
        # node.name = names of the ClassDef entity (top level classes)
        # node.lineno = line number of the ClassDef entity
        # comment = extract the comment associate to the ClassDef entity
        # TODO:
        #  exclude name starts with _,
        #  exclude name is "main",
        #  exclude name contains the word "test" (with all the case variants (TEST, teST, ecc)
        global file_path_and_name, extracted_data
        comment = ast.get_docstring(node)
        add_data(node.name, file_path_and_name, node.lineno, "class", comment)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a FunctionDef node (Functions and Methods).
        :param node: ast node to visit
        """
        # node.name = names of the FunctionDef entity (functions and methods)
        # node.lineno = line number of the FunctionDef entity
        # comment = extract the comment associate to the FunctionDef entity
        # TODO:
        #  exclude name starts with _,
        #  exclude name is "main",
        #  exclude name contains the word "test" (with all the case variants (TEST, teST, ecc)
        global file_path_and_name, extracted_data
        comment = ast.get_docstring(node)
        add_data(node.name, file_path_and_name, node.lineno, "functions", comment)


def main():
    """
    Extract data:
    - Get all *.py files found under input directory (tensorflow) as input
    - Process every Python file to create a CSV file with names/comments of Python classes, methods, functions
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

    # dictionary to csv
    print("Total number of row:", len(extracted_data))

if __name__ == "__main__":
    main()
