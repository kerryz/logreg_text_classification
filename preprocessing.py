# -*- coding: utf-8 -*-
"""
preprocessing.py
----------------
Preprocesses the documents and creates feature files that we later can train our system with.

Each document is represented as a bag-of-word vector,
but since this will generally produce a very sparse vector,
only the words (features) with non-zero values are recorded in the feature files.

Instead of using a simple term frequency approach,
inspiration is drawn from the field of information retrieval and each term's
[tf-idf](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) value is used instead.

To run
------
python preprocessing.py [data_root_dir, [output_dir, [class0, [class1]]]]

Defaults:
    data_root_dir = './given_data/data 1/'
    output_dir = './data/'
    class0 = 'baseball'
    class1 = 'hockey'

Assumes the following directory structure for `data_root_dir`:
    data_root_dir
    |-- s1
        |-- class0
            |-- file101
            |-- file102
            |-- ...
        |-- class1
            |-- file111
            |-- file112
            |-- ...
    |-- s2
        |-- class0
            |-- file201
            |-- file202
            |-- ...
        |-- class1
            |-- file211
            |-- file212
            |-- ...
    |-- ...

Outputs
-------
`output_dir/si_features`
    outputs one feature file per subset.
    examples: `./data/s1_features`, `./data/s2_features`, etc.
    file content format: <label> <feature1>:<value1> <feature2>:<value2>
        where <label> == 0 or 1 since we're using logistic regression
        example: "0 22199:0.0 13448:0.00148313739872 15346:7.97308993901e-05"
`output_dir/word2id.json`
    file content format {'word': word_id}
`output_dir/word2documents.json`
    file content format {'word': nr_of_documents_it_appears_in}
"""

from __future__ import division
import os, sys, errno
import nltk
import json
import math


def_data_root_dir = './given_data/data 1/'
def_output_dir = './data/'
def_class0 = 'baseball'
def_class1 = 'hockey'


def main(
        data_root_dir=def_data_root_dir,
        output_dir=def_output_dir,
        (class0, class1)=(def_class0, def_class1)
        ):
    print "Analyzing documents..."
    # write word2id and word2documents to disk as as json files
    generate_word_files(data_root_dir, output_dir)

    word2id_path = os.path.join(output_dir, 'word2id.json')
    word2documents_path = os.path.join(output_dir, 'word2documents.json')
    # write one feature file per subset to disk
    genereate_feature_files(
        data_root_dir, output_dir, (class0, class1),
        word2id_path, word2documents_path
    )


def generate_word_files(data_root_dir, output_dir):
    """
    Writes word2id and word2documents as json files.

    Parameters
    ----------
    data_root_dir : string
        path to the root directory containing all the training data (documents)
        every file in the directory and its subdirectory will be scanned
    output_dir : string
        path to the directory where `word2id.json` and `word2documents.json` will be stored

    document_count : int
        number of documents (i.e. non-hidden files) in `data_root_dir` and all its subdirectories

    Outputs
    -------
    `output_dir/word2id.json`
        file content format {'word': word_id}
    `output_dir/word2documents.json`
        file content format {'word': nr_of_documents_it_appears_in}
    """
    # get document count and dict {term : nr of documents it appears in}
    document_count, word2documents = get_idf_data(data_root_dir)
    # create a word -> id mapping
    word2id = {}
    for i, word in enumerate(word2documents.keys()):
        word2id[word] = i
    # create `output_dir` if it does not exist
    if not os.path.exists(output_dir):
        mkdir_p(output_dir)
    # write word -> id mapping to file in json format
    with open(os.path.join(output_dir, 'word2id.json'), 'w') as word2id_json_file:
        json.dump(word2id, word2id_json_file)
    # write word -> nr_of_documents_it_appears_in in json format
    with open(os.path.join(output_dir, 'word2documents.json'), 'w') as outfile:
        json.dump(word2documents, outfile)
    return document_count


def genereate_feature_files(
        data_root_dir, output_dir, (class0, class1),
        word2id, word2documents):
    """
    Generates one feature file per subset, which containins the features of each subset.
    These feature files are saved to disk as
    `output_dir/s1_features`, `output_dir/s2_features`, etc.

    Parameters
    ----------
    data_root_dir : string
        path to the root folder containing all the subsets and files
        directory structure:
            data_root_dir
            |-- s1
                |-- class0
                    |-- file101
                    |-- file102
                    |-- ...
                |-- class1
                    |-- file111
                    |-- file112
                    |-- ...
            |-- s2
                |-- class0
                    |-- file201
                    |-- file202
                    |-- ...
                |-- class1
                    |-- file211
                    |-- file212
                    |-- ...
            |-- ...

        output_dir : string
            path to the directory where the feature files will be stored
        (class0, class1) : (string, string)
            names of the directories that contain files for class 0 and class 1 respectively
        word2id: string or dict
            if dict: format {'word': word_id}
            elif string: path to a json file with above format
        word2documents : string or dict
            if dict: format {'word': nr_of_documents_it_appears_in}
            elif string: path to a json file with above format

    Outputs
    -------
    `output_dir/si_features`
        outputs one feature file per subset.
        example: `./data/s1_features`
        format: <label> <feature1>:<value1> <feature2>:<value2>
            where <label> == 0 or 1 since we're using logistic regression
            example: "0 22199:0.0 13448:0.00148313739872 15346:7.97308993901e-05"
    """

    # check if input argument is dict, or path to json file and load it
    if isinstance(word2id, basestring):
        with open(word2id) as json_file:
            word2id = json.load(json_file)
    if isinstance(word2documents, basestring):
        with open(word2documents) as json_file:
            word2documents = json.load(json_file)

    # count number of documents in `data_root_dir`
    document_count = sum([len(
        filter(lambda f: f[0] != '.', files))  # do not count hidden files
        for _, _, files in os.walk(data_root_dir)
    ])

    # create `output_dir` if it does not exist
    if not os.path.exists(output_dir):
        mkdir_p(output_dir)

    # get subset directory names
    _, subsets, _ = next(os.walk(data_root_dir))
    subsets[:] = [s for s in subsets if not s[0] == '.']  # remove hidden subdirectories

    for s_i in sorted(subsets):
        print "Generating feature file for subset", s_i
        subset_features = []
        # get class directeroy names
        _, classes, _ = next(os.walk(os.path.join(data_root_dir, s_i)))
        classes[:] = [c for c in classes if not c[0] == '.']  # remove hidden subdirectories
        # go through each class directory

        for c in sorted(classes):
            if c == class0:
                label = "0"
            elif c == class1:
                label = "1"

            documents_path = os.path.join(data_root_dir, s_i, c)
            doc_root, _, documents = next(os.walk(documents_path))
            documents[:] = [d for d in documents if not d[0] == '.']  # remove hidden files

            # collect features of every document in this class
            for document in sorted(documents):
                document_path = os.path.join(doc_root, document)
                document_features = [label]
                # get data for calculating tf-idf of each word in document
                word_count_in_document_dict = get_word_counts_in_document(document_path)
                total_words_in_document = sum(word_count_in_document_dict.values())

                # calculate tf-idf for each word in document
                for word, count in word_count_in_document_dict.iteritems():
                    tf_idf = (count / total_words_in_document) \
                        * math.log(document_count / (word2documents.get(word, 0) + 1))
                    document_features.append(str(word2id[word]) + ":" + str(tf_idf))
                # after calculating tf-idf for each word in this document
                # save them as one line string
                document_features_str = " ".join(document_features)
                # add all the tf_idf values of every word in the document to subset_features
                subset_features.append(document_features_str)

        # write feature values of this subset to file
        with open(os.path.join(output_dir, s_i + "_features"), 'w') as si_feature_file:
            for document_features in subset_features:
                si_feature_file.write(document_features + "\n")


def get_idf_data(data_root_dir):
    """
    Returns the data required to calculate the idf part in tf-idf.

    Parameters
    ----------
    data_root_dir : string
        path to the root directory containing all the training data (documents)
        every file in the directory and its subdirectory will be scanned

    Returns
    -------
    document_count : int
        number of documents in training data (the numerator in idf calclation)
    term_in_documents_dict : dict
        {term : nr of ducoments the term appears in}

    (
    Calculated from all subsets s1 to s5.
    Might be considered contaminating the test set with information
    from the training set, however I think this is universal information of the English language,
    at least within the domain of sports (specifically hockey and baseball), and this universal
    information might as well have come from statistics collected elsewhere
    on much larger data sets.
    )
    """
    document_count = 0
    word2documents = {}  # {term : nr of documents it appears in}

    for root, dirs, files in os.walk(data_root_dir):
        # remove hidden files and directories
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            document_count += 1
            word_count_in_document = get_word_counts_in_document(os.path.join(root, file))
            for word in word_count_in_document.keys():
                word2documents[word] = word2documents.get(word, 0) + 1

    return document_count, word2documents


def get_word_counts_in_document(document_path):
    """
    Computes the frequency of words occuring in a document.

    Document structure:
    * line 1: From (email address)
    * line 2: Subject
    * rest: Body

    line 1 is not processed because the homework instructions forbid it.

    Parameters
    ----------
    document_path : string
        file path to the document
    word_counts_dict : dict
        {'word' : word_count_in_document}
        this function adds word counts to this dict

    Returns
    -------
    word_counts_dict : dict
        {'word' : word_count_in_document}

    """
    word_counts_dict = {}
    with open(document_path) as infile:
        # Special processing for lines 1 and 2
        infile.readline()  # skip line 1 (From)
        line2 = infile.readline().decode('utf-8', 'replace').lower().split()
        line2 = line2[1:]  # Skip the first word in line 2 ('Subject:')
        for word in line2:
            word_counts_dict[word] = word_counts_dict.get(word, 0) + 1

        # Count words in the rest of the document
        sentences = nltk.sent_tokenize(infile.read().decode('utf-8', 'replace').lower())
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            if not words:
                continue
            for word in words:
                word_counts_dict[word] = word_counts_dict.get(word, 0) + 1
        """
        for line in infile.xreadlines():
            words = line.decode('utf-8', 'replace').lower().split()
            for word in words:
                word_counts_dict[word] = word_counts_dict.get(word, 0) + 1
        """
    return word_counts_dict


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    # set default values
    data_root_dir = def_data_root_dir
    output_dir = def_output_dir
    class0 = def_class0
    class1 = def_class1

    # use command line arguments if provided
    arg_count = len(sys.argv)
    if arg_count > 1:
        data_root_dir = sys.argv[1]
    if arg_count > 2:
        output_dir = sys.argv[2]
    if arg_count > 3:
        class0 = sys.argv[3]
    if arg_count > 4:
        class1 = sys.argv[4]

    main(data_root_dir, output_dir, (class0, class1))
