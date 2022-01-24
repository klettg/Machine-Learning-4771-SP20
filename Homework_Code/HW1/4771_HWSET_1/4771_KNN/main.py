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
    documentDirectory = '/Users/Griffin/Documents/CSE4771/enron1/'
    hamTestLocation = cwd + '/ham/test'
    spamTestLocation = cwd + '/spam/test'

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
            doc_count = 0
            with os.scandir(documentDirectory + dataType) as entries:
                for entry in entries:
                    doc_count += 1

            # create copies of documents in temporary training and testing directories
            with os.scandir(documentDirectory + dataType) as entries:
                curr_index = 0
                for entry in entries:
                    if (curr_index / doc_count) < percent_training:
                        copy2(entry, cwd + '/' + dataType + '/training')
                    else:
                        copy2(entry, cwd + '/' + dataType + '/test')
                    curr_index += 1

    # creates an array of all words we see in this path file location's emails
    def create_array_of_words(local_path, words_array, index_dictionary):
        with os.scandir(local_path) as entries:
            for entry in entries:
                file_content = open(entry, 'r', encoding='utf8', errors='ignore').read()
                tokens = nltk.word_tokenize(file_content)

                for i in tokens:
                    if i.isdigit() or i in string.punctuation:
                        continue
                    clean_word = ps.stem(i)
                    # add word to array
                    if clean_word not in words_array:
                        words_array.append(clean_word)
                        index = len(words_array) - 1
                        index_dictionary[index] = set()
        return words_array, index_dictionary

    def create_labeled_rows_of_word_array(path, array, label, inverted_index):
        global global_identifier_variable
        add_to_index = len(inverted_index) > 0

        # and this
        labeled_rows = None
        with os.scandir(path) as entries:
            for entry in entries:
                local_row = np.zeros(len(array) + 2, dtype=int)
                global_identifier_variable += 1
                local_row[-2] = global_identifier_variable
                local_row[-1] = label
                file_content = open(entry, 'r', encoding='utf8', errors='ignore').read()
                tokens = nltk.word_tokenize(file_content)
                for i in tokens:
                    if i.isdigit() or i in string.punctuation:
                        continue
                    clean_word = ps.stem(i)
                    if clean_word in array:
                        index_of_word = array.index(clean_word)
                        local_row[index_of_word] += 1
                        if add_to_index:
                            inverted_index[index_of_word].add(global_identifier_variable)
                # this is one operation im worried about
                if labeled_rows is None:
                    labeled_rows = np.array(local_row)
                else:
                    labeled_rows = np.vstack([labeled_rows, local_row])
        return labeled_rows

    def classify(current_row, current_list_all_labeled_data_training, inverted_index_local, k_size):
        # do L1 norm on set to get k nearest
        k_nearest_neighbors = linear_search_for_nearest(current_row, current_list_all_labeled_data_training, k_size)

        # get label from neighbors
        label_pred = label_prediction(k_nearest_neighbors)
        return label_pred == current_row[-1]

    def linear_search_for_nearest(current_row, filtered_rows, k_size):
        # distance_identifier_dict = dict()
        current_row_filtered = current_row[:-2]
        euclidean_rows = filtered_rows[:, :-2]

        distances = np.power(euclidean_rows - current_row_filtered, 2).sum(axis=1)
        d_trans = distances.reshape(distances.shape[0], -1)
        filtered_rows = np.hstack((filtered_rows, d_trans))
        sorted_rows = filtered_rows[filtered_rows[:, -1].argsort()]

        neighbors = sorted_rows[:curr_number_of_neighbors, :]
        neighbors = neighbors[:, :-1]
        return neighbors

    def label_prediction(neighbor_rows):
        ham_count = 0
        spam_count = 0
        for local_row in neighbor_rows:
            if local_row[-1] == ham_binary:
                ham_count += 1
            else:
                spam_count += 1
        # arbitrary tiebreaker chooses spam
        if spam_count >= ham_count:
            return spam_binary
        else:
            return ham_binary

    # Here is where I ran different values for experimentation. For turn in, I set it to just run the best model
    # training_split_percentage = [.25, .5, .75, .9]
    # number_of_neighbors = [1, 5, 10, 25]
    training_split_percentage = [.9]
    number_of_neighbors = [1]
    spam_binary = 0
    ham_binary = 1
    global_identifier_variable = 0

    count = 0

    for curr_training_split_percentage in training_split_percentage:
        for curr_number_of_neighbors in number_of_neighbors:

            # update global values

            # split data
            split_data(curr_training_split_percentage, ['spam', 'ham'])

            # create array of words that are in training set
            all_words_array = []
            inverted_index_dictionary = dict()

            all_words_array, inverted_index_dictionary = create_array_of_words(
                cwd + '/ham/training',
                all_words_array,
                inverted_index_dictionary)

            all_words_array, inverted_index_dictionary = create_array_of_words(
                cwd + '/spam/training',
                all_words_array,
                inverted_index_dictionary)

            number_of_features = len(all_words_array)

            # turn each training email into array of word values and a label
            list_of_labeled_rows_spam = create_labeled_rows_of_word_array(
                cwd + '/spam/training',
                all_words_array,
                spam_binary,
                inverted_index_dictionary)

            list_all_labeled_rows_ham = create_labeled_rows_of_word_array(
                cwd + '/ham/training',
                all_words_array,
                ham_binary,
                inverted_index_dictionary)

            list_all_labeled_data_training = np.vstack([list_all_labeled_rows_ham, list_of_labeled_rows_spam])

            print("created word arrays for training data, now creating for test")

            # turn each test point into an array of word values and a label
            list_all_labeled_data_ham_test = create_labeled_rows_of_word_array(
                cwd + '/ham/test',
                all_words_array,
                ham_binary,
                dict())
            list_of_labeled_bags_spam_test = create_labeled_rows_of_word_array(
                cwd + '/spam/test',
                all_words_array,
                spam_binary,
                dict())

            print("Entering classifying mode: ")

            correct_classification_count_spam = 0
            incorrect_classification_count_spam = 0
            correct_classification_count_ham = 0
            incorrect_classification_count_ham = 0

            for row in list_all_labeled_data_ham_test:
                output_true_false = classify(row, list_all_labeled_data_training, inverted_index_dictionary, curr_number_of_neighbors)
                if output_true_false:
                    correct_classification_count_ham += 1
                else:
                    incorrect_classification_count_ham += 1

            for row in list_of_labeled_bags_spam_test:
                if classify(row, list_all_labeled_data_training, inverted_index_dictionary, curr_number_of_neighbors):
                    correct_classification_count_spam += 1
                else:
                    incorrect_classification_count_spam += 1

            total_correct = correct_classification_count_ham + correct_classification_count_spam
            total_incorrect = incorrect_classification_count_ham + incorrect_classification_count_spam
            accuracy = total_correct / (total_incorrect + total_correct)

            print(total_incorrect)
            print(total_correct)
            print(accuracy)

            # Here is where I would record the output of each run to a csv, then did the analysis in excel
            # with open(cwd + '/knn_results.csv', mode='a') as output_file:
            #    output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #    output_writer.writerow([curr_training_split_percentage,
            #                            curr_number_of_neighbors,
            #                            correct_classification_count_ham,
            #                            incorrect_classification_count_ham,
            #                           correct_classification_count_spam,
            #                            incorrect_classification_count_spam,
            #                            total_correct, total_incorrect, accuracy])
            #    output_file.close()
            #    print("!!!!FINISHED RUN # " + str(count))
            #    print(" training perc: " + str(curr_training_split_percentage)
            #          + " number of nearest neighbors: " + str(curr_number_of_neighbors)
            #          + "accuracy" + str(accuracy))
            #count += 1

            # I did not end up using this method
            # def get_search_space_rows(local_row, inverted_index_dictionary_local_copy):
            #    valid_row_identifiers = set()

            # iterate over each column (corresponding to a word) and add rows with also this word to searchable space
            # -2 since last two columns are label & global Indentifier
            #    for i in range(len(local_row) - 2):
            #        if local_row[i] > 0 and not i == 0:
            #            valid_rows = inverted_index_dictionary_local_copy[i]
            #            for rowIdentifier in valid_rows:
            #                valid_row_identifiers.add(rowIdentifier)
            #    return valid_row_identifiers
