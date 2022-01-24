import shutil
from shutil import copy2

import nltk
import string
import os
import numpy as np
from nltk.stem import *
import csv

if __name__ == '__main__':

    cwd = os.getcwd()
    ps = PorterStemmer()
    documentDirectory = '/Users/Griffin/Documents/CSE4771/enron2/'
    hamTestLocation = cwd + '/ham/test'
    spamTestLocation = cwd + '/spam/test'


    class LabeledEmailRow:
        def __init__(self,
                     label,
                     column_values):
            self.label = label
            self.column_values = column_values

    # creates temporary directories for training / testing data, and splits based on passed %
    def split_data(percent_training, data_types):
        for dataType in data_types:

            training_dir_loc = cwd + '/' + dataType + '/training'
            test_dir_loc = cwd + '/' + dataType + '/test'

            # clear directories from previous runs
            if os.path.exists(training_dir_loc):
                shutil.rmtree(training_dir_loc)
            os.makedirs(training_dir_loc)

            if os.path.exists(test_dir_loc):
                shutil.rmtree(test_dir_loc)
            os.makedirs(test_dir_loc)

            # get count of number of documents of this type
            count = 0
            with os.scandir(documentDirectory + dataType) as entries:
                for entry in entries:
                    count += 1

            # create copies of documents in temporary training and testing directories
            with os.scandir(documentDirectory + dataType) as entries:
                curr_index = 0
                for entry in entries:
                    if (curr_index / count) < percent_training:
                        copy2(entry, cwd + '/' + dataType + '/training')
                    else:
                        copy2(entry, cwd + '/' + dataType + '/test')
                    curr_index += 1


    def create_dictionary_of_words(local_path, words_dictionary):
        with os.scandir(local_path) as entries:
            doc_type_count = 0
            for entry in entries:
                doc_type_count += 1
                file_content = open(entry, 'r', encoding='utf8', errors='ignore').read()
                tokens = nltk.word_tokenize(file_content)

                for i in tokens:
                    if i.isdigit() or i in string.punctuation:
                        continue
                    clean_word = ps.stem(i)
                    # add word to dictionary, or update count
                    if clean_word not in words_dictionary:
                        words_dictionary[clean_word] = 0
                    words_dictionary[clean_word] += 1

        return words_dictionary, doc_type_count


    def create_words_array(words_dictionary, minimum_term_frequency):
        words_array = np.empty(0)
        for key in words_dictionary:
            if words_dictionary[key] >= minimum_term_frequency:
                words_array = np.append(words_array, key)
        return words_array


    def create_labeled_email_bags(path, array, label):
        email_bags = []
        with os.scandir(path) as entries:
            for entry in entries:
                local_row = np.zeros(len(array), dtype=int)
                file_content = open(entry, 'r', encoding='utf8', errors='ignore').read()
                tokens = nltk.word_tokenize(file_content)
                for i in tokens:
                    if i.isdigit() or i in string.punctuation:
                        continue
                    clean_word = ps.stem(i)
                    index_of_word = np.where(array == clean_word)
                    local_row[index_of_word] += 1
                labeled_email_row = LabeledEmailRow(label, local_row)
                email_bags.append(labeled_email_row)
        return email_bags


    def label_counts(labeled_email_bags):
        counts = {}
        for local_row in labeled_email_bags:
            label = local_row.label
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts


    def label_prediction(labeled_email_bags):
        ham_count = 0
        spam_count = 0
        for local_row in labeled_email_bags:
            label = local_row.label
            if label == 'ham':
                ham_count += 1
            else:
                spam_count += 1
        # arbitrary tiebreaker chooses spam
        if spam_count >= ham_count:
            return 0
        else:
            return 1


    def calculate_gini_impurity(rows):
        counts = label_counts(rows)
        impurity = 1
        for label in counts:
            prob_of_label = counts[label] / float(len(rows))
            impurity -= pow(prob_of_label, 2)
        return impurity


    def calculate_info_gain(left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * calculate_gini_impurity(left) - (1 - p) * calculate_gini_impurity(right)


    def calculate_best_split(labeled_email_bags):
        best_gain = 0
        best_split_feature_and_freq = None
        current_uncertainty = calculate_gini_impurity(labeled_email_bags)

        for col in range(number_of_features):

            # unique values in the feature/column
            values_of_feature = set([local_row.column_values[col] for local_row in labeled_email_bags])

            for val in values_of_feature:  # for each value

                question = SplitQuestion(col, val)

                true_rows, false_rows = split_on_question(labeled_email_bags, question)

                # Can stop here if there is no gain at all
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = calculate_info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_split_feature_and_freq = gain, question

        return best_gain, best_split_feature_and_freq


    def split_on_question(rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows


    class SplitQuestion:

        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, labeled_email):
            val = labeled_email.column_values[self.column]
            return val >= self.value


    class BaseCase:
        def __init__(self, rows):
            self.predictions = label_counts(rows)
            self.label_prediction = label_prediction(rows)


    class DecisionNode:
        def __init__(self,
                     question,
                     true_branch,
                     false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch


    def build_tree(rows, curr_depth):

        gain, question = calculate_best_split(rows)

        # or max depth reached, or impurity reaches some threshold
        if gain < min_gain or curr_depth == max_tree_depth:
            return BaseCase(rows)

        true_rows, false_rows = split_on_question(rows, question)
        true_branch = build_tree(true_rows, curr_depth + 1)
        false_branch = build_tree(false_rows, curr_depth + 1)

        return DecisionNode(question, true_branch, false_branch)


    def classify(current_row, node):

        if isinstance(node, BaseCase):
            return node.label_prediction

        if node.question.match(current_row):
            return classify(current_row, node.true_branch)
        else:
            return classify(current_row, node.false_branch)


    global all_words_array
    global number_of_features
    global max_tree_depth
    global min_gain

    # here is where I experimeted with different values of paramters. For turnin, I have set them to just the best model
    # training_split_percentage = [.25, .5, .75, .9]
    # min_term_frequency = [5, 25, 45]
    # max_tree_depth_list = [2, 4, 8, 12]
    training_split_percentage = [.75]
    min_term_frequency = [45]
    max_tree_depth_list = [12]
    spam_binary = 0
    ham_binary = 1

    min_gain_list = [float(.01)]

    count = 0

    for curr_training_split_percentage in training_split_percentage:
        for curr_min_term_frequency in min_term_frequency:
            for curr_max_tree_depth in max_tree_depth_list:
                for curr_min_gain in min_gain_list:

                    # update global values
                    max_tree_depth = curr_max_tree_depth
                    min_gain = curr_min_gain

                    # split data
                    split_data(curr_training_split_percentage, ['spam', 'ham'])

                    # create dictionary of words that are in training set and their frequencies
                    all_words_dictionary = dict()
                    all_words_dictionary, hamDocsCount = create_dictionary_of_words(cwd + '/ham/training',
                                                                                    all_words_dictionary)
                    all_words_dictionary, spamDocsCount = create_dictionary_of_words(cwd + '/spam/training',
                                                                                     all_words_dictionary)

                    # create array of features (words) from the dictionary,
                    # maintain only words that are seen with minimum term frequency
                    all_words_array = create_words_array(all_words_dictionary, curr_min_term_frequency)
                    number_of_features = len(all_words_array)

                    # turn each training email into array of word values and a label
                    list_of_labeled_bags_spam = create_labeled_email_bags(cwd + '/spam/training', all_words_array,
                                                                          'spam')
                    list_all_labeled_data_ham = create_labeled_email_bags(cwd + '/ham/training', all_words_array, 'ham')
                    list_all_labeled_data = list_all_labeled_data_ham + list_of_labeled_bags_spam

                    my_tree = build_tree(list_all_labeled_data, 0)

                    list_all_labeled_data_ham_test = create_labeled_email_bags(cwd + '/ham/test', all_words_array,
                                                                               'ham')
                    list_of_labeled_bags_spam_test = create_labeled_email_bags(cwd + '/spam/test', all_words_array,
                                                                               'spam')

                    correct_classification_count_spam = 0
                    incorrect_classification_count_spam = 0
                    correct_classification_count_ham = 0
                    incorrect_classification_count_ham = 0

                    # classify ham test data points
                    for row in list_all_labeled_data_ham_test:
                        if classify(row, my_tree) == 1:
                            correct_classification_count_ham += 1
                        else:
                            incorrect_classification_count_ham += 1

                    # classify spam test data points
                    for row in list_of_labeled_bags_spam_test:
                        if classify(row, my_tree) == 0:
                            correct_classification_count_spam += 1
                        else:
                            incorrect_classification_count_spam += 1

                    total_correct = correct_classification_count_ham + correct_classification_count_spam
                    total_incorrect = incorrect_classification_count_ham + incorrect_classification_count_spam
                    accuracy = total_correct / (total_incorrect + total_correct)
                    count = count + 1

                    print(total_correct)
                    print(total_incorrect)
                    print(accuracy)

                    # Here is where I would write the outputs to a csv, then run analysis in Excel
                    # with open(cwd + '/decision_tree_data.csv', mode='a', newline='') as output_file:
                    #   output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    #    output_writer.writerow([curr_training_split_percentage,
                    #                            curr_min_term_frequency,
                    #                            curr_max_tree_depth,
                    #                            curr_min_gain,
                    #                            correct_classification_count_ham,
                    #                            incorrect_classification_count_ham,
                    #                            correct_classification_count_spam,
                    #                            incorrect_classification_count_spam,
                    #                            total_correct, total_incorrect, accuracy, '', '\n'])
                    #    output_file.close()
                    #    print("!!!!FINISHED RUN # " + str(count))
                    #    print("training perc " + str(curr_training_split_percentage)
                    #          + " min term freq " + str(curr_min_term_frequency)
                    #          + "depth" + str(curr_max_tree_depth)
                    #          + " gain " + str(curr_min_gain)
                    #          + "accuracy" + str(accuracy))
