# Multi-Source Code Search
Second project of [Knowledge Analysis & Management](https://search.usi.ch/en/courses/35263581/knowledge-analysis-management) college course<br>
The choices made during the development of this project are documented in the `report.pdf` file within the repository

## Prerequisites
- `Python 3` must be installed on your machine (version `3.8.12` recommended for full compatibility with `Gensim`)
- The libraries needed for the program to work are: `AST` (included in Python), `Gensim`, `t-SNE`

## Download
You can download the compressed file from this page and extract 
it to a folder of your choice, alternatively you can download it directly 
from a terminal using the following commands:

```
git clone https://github.com/DavideMammarella/Multi-Source-Code-Search.git
```

Access the application folder:
```
cd Multi-Source-Code-Search/
```

## Run the application
The application can be used in two ways:
- **Verbose Mode**: Manually run the commands sequentially (following the order in which they are listed below)
- **Terminal Mode**: Run only `evaluate search engines` command (more in the section)

#### Extract data
Extract all top-level class names, functions and class methods (and comments if available) in *.py files found under the input directory <dir> and any directory under <dir> into a CSV file by running the following command:
```
python3 extract-data.py
```
#### Training of search engines
Train search engines (FREQ, TF-IDF, LSI, Doc2Vec) with the following command:
```
python3 search-data.py
```

#### Evaluate search engines
Evaluate average precision and recall for every search engines (FREQ, TF-IDF, LSI, Doc2Vec) against the `ground-truth-unique.txt` file, with the following command:
```
python3 prec-recall.py
```
This script allows interaction with the terminal, which will ask if the train has already been performed, and depending on the answer you will have two different situations:
- Answer `yes`: it will only compute precision and recall (needs: data.csv, trained models in utils folder)
- Answer `no`: all the scripts listed above will be executed sequentially (will create data.csv, train search engines, compute precision and recall)