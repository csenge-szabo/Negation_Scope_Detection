import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file_path):
    """
    Read gold labels and system predictions from a TSV file, return a lists of dictionaries storing the gold labels, predicted labels and tokens for each sentence.
    Param file_path (str): Path to the TSV file.
    Returns:
        gold_data, system_data, data_dict (list of dicts): Lists of dictionaries with sentence IDs as keys and labels or tokens as values.
    """
    data_dict = {}
    gold_data = []
    system_data = []
    current_gold_scopes = []
    current_system_scopes = []
    current_sentence_id = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            sentence_id, token, gold_label, system_label = parts

            if sentence_id not in data_dict:
                data_dict[sentence_id] = []

            data_dict[sentence_id].append(token)

            if sentence_id != current_sentence_id:
                # Store data for the previous sentence
                if current_gold_scopes and current_system_scopes:
                    gold_data.append({current_sentence_id: current_gold_scopes})
                    system_data.append({current_sentence_id: current_system_scopes})
                # Reset for the new sentence
                current_gold_scopes = []
                current_system_scopes = []
                current_sentence_id = sentence_id

            current_gold_scopes.append(gold_label)
            current_system_scopes.append(system_label)

        # Add the last sentence's data
        if current_gold_scopes and current_system_scopes:
            gold_data.append({current_sentence_id: current_gold_scopes})
            system_data.append({current_sentence_id: current_system_scopes})

    return gold_data, system_data, data_dict

def is_in_scope(label):
    """
    Determines if a token is within the negation scope.
    Parameter label (str): The label to check.
    Return (bool): True if the label indicates the token is in scope, False otherwise.
    """
    return label != 'OS'

def exact_span_based_evaluation(system_data, gold_data, data_dict):
    """
    Evaluates the system's performance on a span level, comparing exact matches of spans between system predictions and gold labels.
    Parameters:
        - system_data (list): The system's predicted labels for each token in each sentence.
        - gold_data (list): The gold labels for each token in each sentence.
        - data_dict (dict): A dictionary mapping sentence IDs to the list of tokens.
    Return:
        - precision (float): The precision of the system's span predictions.
        - recall (float): The recall of the system's span predictions.
        - f1_score (float): The F1 score of the system's span predictions.
        - dictionaries categorising different types of errors (fn1_dict, fn2_dict, fp_dict, fnfp_dict)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    y_true = []  # List to store gold values
    y_pred = []  # List to store predicted values
    fn1_dict = {}
    fn2_dict = {}
    fp_dict = {}
    fnfp_dict = {}

    # Iterate over each sentence in the system predicted data
    for system_pred in system_data:
        for sentence_id, system_labels in system_pred.items():
            # Find the corresponding gold data for this sentence ID
            for gold in gold_data:
                if sentence_id in gold:
                    gold_labels = gold[sentence_id]
                    # print(f'Gold labels: \n{gold_labels}')

                    if all(label == 'OS' for label in system_labels) and all(label == 'OS' for label in gold_labels):
                        true_negatives += 1
                        y_true.append(0)
                        y_pred.append(0)
                        # print("True negative")

                    elif system_labels == gold_labels and any(label != 'OS' for label in system_labels):
                        true_positives += 1
                        y_true.append(1)
                        y_pred.append(1)
                        # print("True positive")

                    elif all(label == 'OS' for label in system_labels) and any(label != 'OS' for label in gold_labels):
                        false_negatives += 1
                        y_true.append(1)
                        y_pred.append(0)
                        # print("False negative Type 1: system found no scope at all")
                        # print(sentence_id)
                        # print(data_dict[sentence_id])
                        # print(gold_labels, '\n', system_labels, '\n')
                        fn1_dict[sentence_id] = [data_dict[sentence_id], gold_labels, system_labels]

                    elif all(label == 'OS' for label in gold_labels) and any(label != 'OS' for label in system_labels):
                        false_positives += 1
                        y_true.append(0)
                        y_pred.append(1)
                        # print("False positive only")
                        # print(sentence_id)
                        # print(data_dict[sentence_id])
                        # print(gold_labels, '\n', system_labels, '\n')
                        fp_dict[sentence_id] = [data_dict[sentence_id], gold_labels, system_labels]

                    elif all(s == g or s == 'OS' for s, g in zip(system_labels, gold_labels)) and system_labels.count(
                            'IS') < gold_labels.count('IS'):
                        false_negatives += 1
                        y_true.append(1)
                        y_pred.append(0)
                        # print("False negative Type 2: partial match")
                        # print(sentence_id)
                        # print(data_dict[sentence_id])
                        # print(gold_labels, '\n', system_labels, '\n')
                        fn2_dict[sentence_id] = [data_dict[sentence_id], gold_labels, system_labels]

                    elif any(s != g and s == 'IS' for s, g in zip(system_labels, gold_labels)) and any(
                            s == g and s == 'IS' for s, g in zip(system_labels, gold_labels)):
                        false_positives += 1
                        false_negatives += 1
                        y_true.append(1)
                        y_pred.append(0)
                        y_true.append(0)
                        y_pred.append(1)
                        # print("Both false positive and false negative: incorrect match with some overlap")
                        # print(sentence_id)
                        # print(data_dict[sentence_id])
                        # print(gold_labels, '\n', system_labels, '\n')
                        fnfp_dict[sentence_id] = [data_dict[sentence_id], gold_labels, system_labels]

                    elif any(s == 'IS' and g == 'OS' for s, g in zip(system_labels, gold_labels)) and any(
                            s == 'OS' and g == 'IS' for s, g in zip(system_labels, gold_labels)):
                        false_positives += 1
                        false_negatives += 1
                        y_true.append(1)
                        y_pred.append(0)
                        y_true.append(0)
                        y_pred.append(1)
                        # print("Both false positive and false negative: incorrect and there is no overlap")
                        # print(sentence_id)
                        # print(data_dict[sentence_id])
                        # print(gold_labels, '\n', system_labels, '\n')
                        fnfp_dict[sentence_id] = [data_dict[sentence_id], gold_labels, system_labels]

                    else:
                        print("Unidentified case")

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, fn1_dict, fn2_dict, fp_dict, fnfp_dict

file_path = 'data/bert_postprocessed_testset.tsv'
error_file = 'data/erroroutput-classified.tsv'

# Read the gold labels and system predictions
gold_data, system_data, data_dictx = read_data(file_path)

# Run span-based evaluation
precision, recall, f1_score, fn1_dict, fn2_dict, fp_dict, fnfp_dict = exact_span_based_evaluation(system_data,
                                                                                                  gold_data, data_dictx)

with open(error_file, 'w') as file:
    file.write('Error statistics:\nFNFP: ' + str(len(fnfp_dict)) + '\nFN1: ' + str(len(fn1_dict)) + '\nFN2: ' + str(
        len(fn2_dict)) + '\nFP: ' + str(len(fp_dict)))
    file.write('\n===========================FNFP Errors:===========================\nAmount:' + ' ' + str(
        len(fnfp_dict)) + '\n')

    for key, value in fnfp_dict.items():
        token = value[0]
        gold_label = value[1]
        pred = value[2]

        file.write('\n')
        for t, g, p in zip(token, gold_label, pred):
            print(t, g, p)

            file.write(f'{key}\t{t}\t{g}\t{p}\n')

    file.write(
        '===========================FN1 Errors:===========================\nAmount:' + ' ' + str(len(fn1_dict)) + '\n')

    for key, value in fn1_dict.items():
        token = value[0]
        gold_label = value[1]
        pred = value[2]

        file.write('\n')
        for t, g, p in zip(token, gold_label, pred):
            print(t, g, p)

            file.write(f'{key}\t{t}\t{g}\t{p}\n')

    file.write(
        '===========================FN2 Errors:===========================\nAmount:' + ' ' + str(len(fn2_dict)) + '\n')

    for key, value in fn2_dict.items():
        token = value[0]
        gold_label = value[1]
        pred = value[2]

        file.write('\n')
        for t, g, p in zip(token, gold_label, pred):
            print(t, g, p)

            file.write(f'{key}\t{t}\t{g}\t{p}\n')

    file.write(
        '===========================FP Errors:===========================\nAmount:' + ' ' + str(len(fp_dict)) + '\n')

    for key, value in fp_dict.items():
        token = value[0]
        gold_label = value[1]
        pred = value[2]

        file.write('\n')
        for t, g, p in zip(token, gold_label, pred):
            print(t, g, p)

            file.write(f'{key}\t{t}\t{g}\t{p}\n')
