import os
from fnmatch import fnmatch
import ast


# [extract data] extract names/comments of Python classes, methods, functions
# Save the extracted info into a CSV file.
# When available, extract also the comment line following the class, function or method declaration.

# TODO (ASK) inner functions/methods can be skipped by just avoiding to trigger a recursive generic_visit call.
# • To extract the comment line, you can:
# • read the source file up to the line number of each entity; then, read the next lines until the declaration ends (i.e., ‘):’ or ‘->’ is found in Python): if the next line starts with ”””, in Python it is a comment line to be extracted
# • use this function call: comment = ast.get_docstring(node)
# • locate the following types of nodes: Str(s=‘comment’) (Python 3.7 and earlier) or
# Constant(value=‘comment’, kind=None) (Python 3.8 and later)

class AstVisitor(ast.NodeVisitor):
    """
    Subclass of ast.NodeVisitor, with the purpose of adding visitor methods.
    """
    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit a ClassDef node (Top Level Classes).
        :param node: ast node to visit
        :return:
        """
        # node.name = names of the ClassDef entity (top level classes)
        # node.lineno = line number of the ClassDef entity
        # TODO:
        #  exclude name starts with _,
        #  exclude name is "main",
        #  exclude name contains the word "test" (with all the case variants (TEST, teST, ecc)
        # "file":None (cos
        print(node.name, node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a FunctionDef node (Functions and Methods).
        :param node: ast node to visit
        :return:
        """
        # node.name = names of the FunctionDef entity (functions and methods)
        # node.lineno = line number of the FunctionDef entity
        # TODO:
        #  exclude name starts with _,
        #  exclude name is "main",
        #  exclude name contains the word "test" (with all the case variants (TEST, teST, ecc)
        print(node.name, node.lineno)


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
                with open(os.path.join(path, file), 'r') as py_file:
                    ast_of_py_file = ast.parse(py_file.read())
                    AstVisitor().visit(ast_of_py_file)


if __name__ == "__main__":
    main()
