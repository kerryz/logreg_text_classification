import json


def features_per_line(file_path):
    features_per_line_list = []
    with open(file_path) as infile:
        for line in infile:
            features = line.split(" ")
            features_per_line_list.append(len(features))
    for feature_count in features_per_line_list:
        print feature_count


def print_id2word_feature_file(feature_file_path, word2id_file_path):
    with open(word2id_file_path) as word2id_file:
        word2id = json.load(word2id_file)
    id2word = dict((value, key) for key, value in word2id.iteritems())

    with open(feature_file_path) as feature_file:
        features = feature_file.readline().split(" ")[1:]
        for feature in features:
            feature_id, feature_value = feature.split(":")
            print id2word[int(feature_id)], feature_value


#features_per_line('./given_data/feature_example_s1')
print_id2word_feature_file('./data/s1_features', './data/word2id.json')
