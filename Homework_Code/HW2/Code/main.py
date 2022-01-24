import csv
import os

import scipy.io
import pandas as pd


def read_data(file_path):
    mat = scipy.io.loadmat(file_path)
    x_data = mat['X']  # variable in mat file
    x_columns = [f'col_{num}' for num in range(len(x_data[0]))]
    x_rows = [f'index_{num}' for num in range(len(x_data))]
    x_df = pd.DataFrame(x_data, columns=x_columns, index=x_rows)
    y_data = mat['Y']
    y_column = ['label']
    y_index = [f'index_{num}' for num in range(len(y_data))]
    y_df = pd.DataFrame(y_data, columns=y_column, index=y_index)
    total_df = pd.concat([x_df, y_df], axis=1)
    # print(total_df.head())
    return total_df


def split_data(data_df, pct_training):
    train = data_df.sample(frac=pct_training, random_state=150)
    test = data_df.drop(train.index)
    return train, test


def get_label_counts(rows):
    counts = rows['label'].value_counts(sort=False)
    return counts


def calc_gini_impty(rows):
    counts = get_label_counts(rows)
    impurity = 1
    for i in range(10):
        prob_of_label = counts.get(i, 0) / float(len(rows.index))
        impurity -= pow(prob_of_label, 2)
    return impurity


def calculate_info_gain(left_rows, right_rows, current_uncertainty):
    p_left = float(len(left_rows.index)) / (len(left_rows.index) + len(right_rows.index))
    current_uncertainty -= p_left * calc_gini_impty(left_rows)
    current_uncertainty -= (1 - p_left) * calc_gini_impty(right_rows)
    return current_uncertainty


def calculate_best_split(rows):
    best_gain = float(0)
    best_column = None
    best_value = None
    current_uncertainty = calc_gini_impty(rows)

    for col in rows.columns:

        # unique values in the feature/column
        # values_of_feature = rows[col].unique()
        values_of_feature = [40, 80, 120, 160, 200, 230]

        for val in values_of_feature:  # for each value

            true_rows, false_rows = split_on_col_val(rows, col, val)

            # Can stop here if there is no gain at all
            if len(true_rows.index) == 0 or len(false_rows.index) == 0:
                continue

            # Calculate the information gain from this split
            gain = calculate_info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain = gain
                best_column = col
                best_value = val

    return best_gain, best_column, best_value


def split_on_col_val(rows, col, val):
    mask = rows[col] > val
    true_rows = rows[mask]
    false_rows = rows.drop(true_rows.index)
    return true_rows, false_rows


class BaseCase:
    def __init__(self, rows):
        self.predictions = get_label_counts(rows)
        self.label_prediction = label_prediction(rows)


def label_prediction(rows):
    return get_label_counts(rows).idxmax()


class DecisionNode:
    def __init__(self,
                 column,
                 value,
                 true_branch,
                 false_branch):
        self.column = column,
        self.value = value,
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows, remaining_depth):

    gain, col, value = calculate_best_split(rows)

    # or max depth reached, or impurity reaches some threshold
    if gain < .000001 or remaining_depth == 0:
        return BaseCase(rows)

    true_rows, false_rows = split_on_col_val(rows, col, value)
    true_branch = build_tree(true_rows, remaining_depth - 1)
    false_branch = build_tree(false_rows, remaining_depth - 1)

    return DecisionNode(col, value, true_branch, false_branch)


def classify(row, tree_node):
    if isinstance(tree_node, BaseCase):
        row['predicted_label'] = tree_node.label_prediction
        return row
    elif row[tree_node.column[0]][0] > tree_node.value[0]:
        return classify(row, tree_node.true_branch)
    else:
        row_val = row[tree_node.column[0]]
        tuple_var = tree_node.value[0]
        return classify(row, tree_node.false_branch)

def classify_data(data, tree):
    classified_df = pd.DataFrame()
    for row_name in data.index.values:
        classified_df = classified_df.append(classify(data.loc[[row_name]], tree))
    return classified_df

def get_error_and_correct(classified_df):
    mask = classified_df['predicted_label'] == classified_df['label']
    correct_rows = classified_df[mask]
    total_count = len(classified_df.index)
    correct_count =len (correct_rows.index)
    error = 1 - (correct_count / total_count)
    return error, correct_count, total_count


if __name__ == '__main__':
    path = '/Users/Griffin/PycharmProjects/digits.mat'
    cwd = os.getcwd()
    pct_training = [.15, .30, .45, .6, .75, .9]
    max_depth = [3, 6, 9, 15, 25, 40]
    data_df = read_data(path)
    count = 0

    for curr_train_pct in pct_training:
        for curr_max_depth in max_depth:

            train_df, test_df = split_data(data_df, curr_train_pct)
            tree_start_node = build_tree(train_df, curr_max_depth)

            test_classified_df = classify_data(test_df, tree_start_node)
            test_error, test_correct_count, test_total_count = get_error_and_correct(test_classified_df)

            train_classified_df = classify_data(train_df, tree_start_node)
            train_error, train_correct_count, train_total_count = get_error_and_correct(train_classified_df)

            with open(cwd + '/decision_tree_data.csv', mode='a', newline='') as output_file:
                output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow([curr_max_depth,
                                        curr_train_pct,
                                        test_error,
                                        test_correct_count,
                                        test_total_count,
                                        train_error,
                                        train_correct_count,
                                        train_total_count,
                                        '', '\n'])
                output_file.close()
                print("!!!!FINISHED RUN # " + str(count))

            count = count + 1

