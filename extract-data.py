import os
from fnmatch import fnmatch

# [extract data] extract names/comments of Python classes, methods, functions
#     - Get all names of top level classes, functions and class methods in *.py files found under input directory


# Use AST for Python parsing.
# Save the extracted info into a CSV file.
# When available, extract also the comment line following the class, function or method declaration.

# You can use ast.NodeVisitor for Python parsing; attribute lineno gives you the line
# number of each entity; inner functions/methods can be skipped by just avoiding to trigger
# a recursive generic_visit call.
# • To extract the comment line, you can:
# • read the source file up to the line number of each entity; then, read the next lines until the declaration ends (i.e., ‘):’ or ‘->’ is found in Python): if the next line starts with ”””, in Python it is a comment line to be extracted
# • use this function call: comment = ast.get_docstring(node)
# • locate the following types of nodes: Str(s=‘comment’) (Python 3.7 and earlier) or
# Constant(value=‘comment’, kind=None) (Python 3.8 and later)

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
                with open(os.path.join(path, file), 'r'):
                    # name of the file (no path) in: file
                    print(file)


if __name__ == "__main__":
    main()