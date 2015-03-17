from __future__ import division
import os
import nltk
import json
import math


def main():
    #generate_word_files('./given_data/data 1/', './data/', 'baseball', 'hockey')
    genereate_feature_files('./given_data/data 1/', './data/', ('baseball', 'hockey'),
                            1989, './data/word2documents.json', './data/word2id.json')
    """
    document_count, term_in_documents_dict = get_idf_data('./given_data/data 1')
    counter = 0
    sorted_terms = sorted(term_in_documents_dict.items(), key=operator.itemgetter(1))
    for (k, v) in sorted_terms:
        print k, ": ", v
        counter += 1
    print '-' * 20
    print "document_count", document_count
    """


def genereate_feature_files(
        data_root_dir, output_dir, (class0, class1),
        document_count, word2documents, word2id):
    """

    """

    # check if input argument is dict, or path to json file and load it
    if isinstance(word2id, basestring):
        with open(word2id) as json_file:
            word2id = json.load(json_file)
    if isinstance(word2documents, basestring):
        with open(word2documents) as json_file:
            word2documents = json.load(json_file)

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

                #######################################################################
                print "words in document", len(word_count_in_document_dict.keys())
                if document_path == './given_data/data 1/s1/baseball/100521':
                    print "nr of documents:", len(documents)
                    print document_path
                    print len(word_count_in_document_dict.keys())
                    ii = 0
                    for k, v in word_count_in_document_dict.iteritems():
                        print k, v
                        if ii > 20:
                            print "etc etc..."
                            break
                        ii += 1
                #######################################################################

                total_words_in_document = sum(word_count_in_document_dict.values())
                # calculate tf-idf for each word in document
                for word, count in word_count_in_document_dict.iteritems():
                    tf_idf = (count / total_words_in_document) \
                        * math.log(document_count / (word2documents.get(word, 0) + 1))
                    document_features.append(str(word2id[word]) + ":" + str(tf_idf))
                    """
                    ##########################################################################
                    print '-' * 30
                    print "word:", word
                    print "count:                 ", count
                    print "total_words_in_document", total_words_in_document
                    print "document_count:        ", document_count
                    print "word2documents_count:  ", word2documents.get(word, 0)
                    print "document_count / word2doc", document_count / (word2documents.get(word, 0) + 1)
                    print "tf:  ", (count / total_words_in_document)
                    print "idf:", math.log(document_count / (word2documents.get(word, 0) + 1))
                    print "tfidf:", tf_idf
                    ##########################################################################
                    """
                # after calculating tf-idf for each word in this document
                # save them as one line string
                document_features_str = " ".join(document_features)
                # add all the tf_idf values of every word in the document to subset_features
                subset_features.append(document_features_str)
                ########################################################################
                # Test
                #print document_features
                #print len(document_features)
                #return
                ########################################################################
        # write feature values of this subset to file
        with open(os.path.join(output_dir, s_i + "_features"), 'w') as si_feature_file:
            for document_features in subset_features:
                si_feature_file.write(document_features + "\n")
                ########################################################################
                # Test
                #print document_features
                #break
                ########################################################################


def generate_word_files(data_root_dir, output_dir, class0, class1):
    """
    Writes word2id and word2documents as json files.
    """
    # get document count and dict {term : nr of documents it appears in}
    document_count, word2documents = get_idf_data(data_root_dir)
    # create a word -> id mapping
    word2id = {}
    for i, word in enumerate(word2documents.keys()):
        word2id[word] = i
    # write word -> id mapping to file in json format
    with open(os.path.join(output_dir, 'word2id.json'), 'w') as word2id_json_file:
        json.dump(word2id, word2id_json_file)
    with open(os.path.join(output_dir, 'word2documents.json'), 'w') as outfile:
        json.dump(word2documents, outfile)


def get_idf_data(data_root_dir):
    """
    Returns the data required to calculate the idf part in tf-idf.

    Parameters
    ----------
    data_root_dir : string
        path to the root direcory containing all the training data (documents)

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


if __name__ == '__main__':
    main()
