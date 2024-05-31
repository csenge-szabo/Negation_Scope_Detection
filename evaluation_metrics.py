import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file_path):
    """
    Read the data from a TSV file and return a list of dictionaries, where dictionary keys are unique identifiers of
    the sentence, and dictionary values are a list of negation scopes.
    Params:
        file_path (str): path to a TSV file containing the data.
    Returns: a list of dictionaries, where each dictionary maps a sentence ID to a list of scopes.
    """
    data = []
    current_sentence_id = ""
    current_labels = []
    sentence_ids=set()

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')

        for row in reader:
            if len(row) > 20:
            # Unpack the row into its components
                chapter, sentence, token_no, word, lemma, pos, constituency, negation_cue, scope, negated_property, syntactic_distance, is_same_clause, is_same_phrase, is_punct, sentence_postion, is_cue, distance_to_cue, dependency_label, dependency_head, distance_to_root, distance_to_cue = row
                label = row[8]

                # Construct a unique sentence ID
                sentence_id = f"{chapter}_{sentence}"
                sentence_ids.add(sentence_id)

                # Check if we've moved on to a new sentence
                if sentence_id != current_sentence_id:
                    # If not the first sentence, save the previous sentence's data
                    if current_labels:
                        data.append({current_sentence_id: current_labels})

                    # Reset for the new sentence
                    current_sentence_id = sentence_id
                    current_labels = []

                # Append the current label to the list for the current sentence
                current_labels.append(label)

        if current_labels:
            data.append({current_sentence_id: current_labels})

    return data

def is_in_scope(label):
    """
    Determines if a token is within the negation scope.
    Parameter label (str): The label to check.
    Return (bool): True if the label indicates the token is in scope, False otherwise.
    """
    return label != 'OS'

def token_based_evaluation(gold_file, system_file):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    y_true = []  # List to store gold values
    y_pred = []  # List to store predicted values

    # Iterate over each sentence in the system predicted data
    for system_pred in system_data:
        for sentence_id, system_labels in system_pred.items():
            # Find the corresponding gold data for this sentence ID
            # print(sentence_id)
            # print(f'Predicted labels: {system_labels}')
            for gold in gold_data:
                if sentence_id in gold:
                    gold_labels = gold[sentence_id]
                    # print(f'Gold labels: {gold_labels}\n')
                  
                    for system_label, gold_label in zip(system_labels, gold_labels):
                        system_in_scope = is_in_scope(system_label)
                        gold_in_scope = is_in_scope(gold_label)
                        
                        if system_in_scope and gold_in_scope:
                            true_positives += 1
                            y_true.append(1)
                            y_pred.append(1)
                            # print("True positives")
                        elif system_in_scope and not gold_in_scope:
                            false_positives += 1
                            y_true.append(0)
                            y_pred.append(1)
                            # print("False positives")
                        elif not system_in_scope and not gold_in_scope:
                            true_negatives += 1
                            y_true.append(0)
                            y_pred.append(0)
                            # print("True negatives")
                        elif not system_in_scope and gold_in_scope:
                            false_negatives += 1
                            y_true.append(1)
                            y_pred.append(0)
                            # print("False negatives")
                  
    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                  
    # print(sum(y_true), sum(y_pred))
                  
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Not in Scope', 'In Scope']

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)

    # Adding the aesthetics
    plt.title('Token Confusion Matrix')
    plt.ylabel('Gold')
    plt.xlabel('Prediction')

    # Show the plot
    plt.show()

    return precision, recall, f1_score

def exact_span_based_evaluation(system_data, gold_data):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    y_true = []  # List to store gold values
    y_pred = []  # List to store predicted values

    # Iterate over each sentence in the system predicted data
    for system_pred in system_data:
        for sentence_id, system_labels in system_pred.items():
            # Find the corresponding gold data for this sentence ID
            # print(sentence_id)
            # print(f'System labels:\n{system_labels}')
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
                        false_negatives +=1
                        y_true.append(1)
                        y_pred.append(0)
                        # print("False negative Type 1: system found no scope at all")
                                        
                    elif all(label == 'OS' for label in gold_labels) and any(label != 'OS' for label in system_labels):
                        false_positives +=1
                        y_true.append(0)
                        y_pred.append(1)
                        # print("False positive only")
                
                    elif all(s == g or s == 'OS' for s, g in zip(system_labels, gold_labels)) and system_labels.count('IS') < gold_labels.count('IS'):
                        false_negatives += 1
                        y_true.append(1)
                        y_pred.append(0)
                        # print("False negative Type 2: partial match")
                        
                    elif any(s != g and s == 'IS' for s, g in zip(system_labels, gold_labels)) and any(s == g and s == 'IS' for s, g in zip(system_labels, gold_labels)):
                        false_positives += 1
                        false_negatives += 1
                        y_true.append(1)  
                        y_pred.append(0)  
                        y_true.append(0) 
                        y_pred.append(1) 
                        # print("Both false positive and false negative: incorrect match with some overlap")

                    elif any(s == 'IS' and g == 'OS' for s, g in zip(system_labels, gold_labels)) and any(s == 'OS' and g == 'IS' for s, g in zip(system_labels, gold_labels)):
                        false_positives += 1
                        false_negatives += 1
                        y_true.append(1)  
                        y_pred.append(0)  
                        y_true.append(0) 
                        y_pred.append(1) 
                        # print("Both false positive and false negative: incorrect and there is no overlap")

                    else:
                        print("Unidentified case")


    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    cm = confusion_matrix(y_true, y_pred)
    classes = ['Not in Scope', 'In Scope']

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)

    # Adding the aesthetics
    plt.title('Span Confusion Matrix')
    plt.ylabel('Gold')
    plt.xlabel('Prediction')

    # Show the plot
    plt.show()

    return precision, recall, f1_score

gold_file = 'data/with_complete_features_test.tsv'
system_file = 'data/crf_test_output_all_features.tsv'

# Uncomment the lines below if you want to evaluate the CRF model only with constituency/dependency features
# system_file = 'data/crf_dev_output_dep_features.tsv' 
# system_file = 'data/crf_dev_output_constituency_features.tsv'

system_data = read_data(system_file)
gold_data = read_data(gold_file)

precision, recall, f1_score = token_based_evaluation(gold_file, system_file)
print(f"Token-based Precision: {round(precision, 3)}")
print(f"Token-based Recall: {round(recall, 3)}")
print(f"Token-based F1-Score: {round(f1_score, 3)} \n")

precision, recall, f1_score = exact_span_based_evaluation(system_data, gold_data)
print(f"Span-based Precision: {round(precision, 3)}")
print(f"Span-based Recall: {round(recall, 3)}")
print(f"Span-based F1-Score: {round(f1_score, 3)}")
