import pandas as pd
import pycrfsuite

def read_and_format_data(file_path):
    """
    Reads data from a TSV file and formats it into a list of sentences.
    Args: file_path (str): Path to the input TSV file. 
    Returns: List: A list of sentences, where each sentence is a list of token information.
    """
    # Define the column names for the TSV file
    columns = ['doc_id', 'sentence_num', 'token_num', 'token', 'lemma', 'pos', 'constituency', 'cue', 'label', 'focus', 'constituency_distance', 'same_clause', 'same_phrase', 'is_punct', 'sentence_position', 'is_negation_cue', 'token_distance', 'dependency_type', 'dependency_head', 'distance_to_root', 'distance_to_cue']
    
    # Read the data from the TSV file into a Pandas DataFrame
    data_df = pd.read_csv(file_path, sep='\t', names=columns)
    
    # Initialize an empty list to store formatted data
    formatted_data = []

    # Group data by document ID and sentence number, creating sentences
    for _, group in data_df.groupby(['doc_id', 'sentence_num']):
        # Extract token information for each row and create a tuple for each token
        sentence = [(row['token'], row['lemma'], row['pos'], row['cue'], row['constituency_distance'], row['same_clause'], row['same_phrase'], row['is_punct'], row['sentence_position'], row['is_negation_cue'], row['token_distance'], row['dependency_type'], row['dependency_head'], row['distance_to_root'], row['distance_to_cue'], row['label']) for index, row in group.iterrows()]
        
        # Append the sentence to the list of formatted data
        formatted_data.append(sentence)

    return formatted_data

def extract_features(sentence, exclude_feature=None):
    """
    Extracts features from a sentence for use in CRF model training and prediction.
    Args:sentence (list): A list of token information for a single sentence.
    Returns:list: A list of feature dictionaries, one for each token in the sentence.
    """
    sentence_features = []

    for i in range(len(sentence)):
        # Current word and its features
        token, lemma, pos, cue, constituency_distance, same_clause, same_phrase, is_punct, sentence_position, is_negation_cue, token_distance, dependency_type, dependency_head, distance_to_root, distance_to_cue, label = sentence[i]

        # Previous and next POS tags
        prev_pos = sentence[i - 1][2] if i > 0 else 'START'
        next_pos = sentence[i + 1][2] if i < len(sentence) - 1 else 'END'

        # Constructing features
        features = {
            'token': token,
            'lemma': lemma,
            'pos': pos,
            'lexicalized_pos': f"{lemma}_{pos}",
            'cue': cue,
            'prev_pos': prev_pos,
            'next_pos': next_pos,
            'constituency_distance': constituency_distance,
            'same_clause': same_clause,
            'same_phrase': same_phrase,
            'is_punct': is_punct,
            'sentence_position': sentence_position,
            'is_negation_cue': is_negation_cue,
            'token_distance': token_distance,
            'dependency_type': dependency_type,
            'dependency_head': dependency_head,
            'distance_to_root': distance_to_root,
            'distance_to_cue': distance_to_cue
        }

        # Exclude the specified feature
        if exclude_feature and exclude_feature in features:
            del features[exclude_feature]

        sentence_features.append(features)

    return sentence_features

def extract_labels(sentence):
    """
    Extracts labels from a sentence for use in CRF model training and evaluation.
    Args:sentence (list): A list of token information for a single sentence.
    Returns:list: A list of labels corresponding to each token in the sentence.
    """
    return [label for token, lemma, pos, cue, constituency_distance, same_clause, same_phrase, is_punct, sentence_position, is_negation_cue, token_distance, dependency_type, dependency_head, distance_to_root, distance_to_cue, label in sentence]

def write_predictions_to_file(original_file_path, sentences_with_predictions, output_file_path):
    """
    Writes predicted labels to an output file based on original data and sentence predictions.
    Args:
        original_file_path (str): Path to the original TSV file.
        sentences_with_predictions (list): List of sentences with predicted labels.
        output_file_path (str): Path to the output file.
    """
    columns = ['doc_id', 'sentence_num', 'token_num', 'token', 'lemma', 'pos', 'constituency', 'cue', 'label', 'focus', 'constituency_distance', 'same_clause', 'same_phrase', 'is_punct', 'sentence_position', 'is_negation_cue', 'token_distance', 'dependency_type', 'dependency_head', 'distance_to_root', 'distance_to_cue']
    original_df = pd.read_csv(original_file_path, sep='\t', names=columns, header=None)

    # Flatten the list of sentences with predictions into a single list of predictions
    predictions_flat = [label for sentence in sentences_with_predictions for label in sentence]

    # Replace the 'label' column with the predictions
    original_df['label'] = predictions_flat
    
    original_df.to_csv(output_file_path, sep='\t', index=False, header=None)

def is_in_scope(label):
    """
    Determines if a label is within the desired scope for evaluation.
    Args: Label (str): A label to be evaluated. 
    Returns:bool: True if the label is within the desired scope, False otherwise.
    """
    return label != 'OS'

def calculate_metrics(y_true, y_pred):
    """
    Calculates precision, recall, and F1-score based on true and predicted labels.
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    Returns: tuple: A tuple containing precision, recall, and F1-score.
    """
    true_positives = sum(1 for yt, yp in zip(y_true, y_pred) if is_in_scope(yt) and is_in_scope(yp))
    false_positives = sum(1 for yt, yp in zip(y_true, y_pred) if not is_in_scope(yt) and is_in_scope(yp))
    false_negatives = sum(1 for yt, yp in zip(y_true, y_pred) if is_in_scope(yt) and not is_in_scope(yp))
    true_negatives = sum(1 for yt, yp in zip(y_true, y_pred) if not is_in_scope(yt) and not is_in_scope(yp))

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def train_and_evaluate(X_train, y_train, X_dev, y_dev, exclude_feature=None):
    """
    Trains a Conditional Random Fields (CRF) model on training data and evaluates it on development data.
    Args:
        X_train (list): A list of feature sequences for training.
        y_train (list): A list of label sequences for training.
        X_dev (list): A list of feature sequences for development.
        y_dev (list): A list of label sequences for development.
        exclude_feature (str, optional): A feature name to exclude from training and evaluation.
    Returns:tuple: A tuple containing the F1-score and predicted labels on the development data.
    """
    # Create a CRF trainer for training the model
    trainer = pycrfsuite.Trainer(verbose=False)
    
    # Append training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    # Set hyperparameters for the CRF model (default values)
    trainer.set_params({
        'c1': 0.1,                 # L1 regularization parameter
        'c2': 0.1,                 # L2 regularization parameter
        'max_iterations': 100,     # Maximum number of iterations
        'feature.possible_transitions': True  # Include possible transitions as features
    })
    
    # Train the CRF model and save it to 'crf.model'
    trainer.train('crf.model')

    # Create a CRF tagger to load the trained model
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')

    # Use the trained CRF model to predict labels for development data
    y_pred = [tagger.tag(xseq) for xseq in X_dev]

    # Flatten the nested lists of true labels and predicted labels
    y_dev_flat = [label for sentence in y_dev for label in sentence]
    y_pred_flat = [label for sentence in y_pred for label in sentence]

    # Calculate precision, recall, and F1-score using the true and predicted labels
    precision, recall, f1_score = calculate_metrics(y_dev_flat, y_pred_flat)
    
    # Return the F1-score and predicted labels
    return f1_score, y_pred

train_file_path = 'data/with_complete_features_training.tsv'  
dev_file_path = 'data/with_complete_features_dev.tsv'  

train_sentences = read_and_format_data(train_file_path)
dev_sentences = read_and_format_data(dev_file_path)

# Applying feature extraction to the training and development data
X_train = [extract_features(sentence) for sentence in train_sentences]
y_train = [extract_labels(sentence) for sentence in train_sentences]

X_dev = [extract_features(sentence) for sentence in dev_sentences]
y_dev = [extract_labels(sentence) for sentence in dev_sentences]

# Baseline model performance
baseline_f1_score, baseline_predictions = train_and_evaluate(X_train, y_train, X_dev, y_dev)
print(f"Baseline F1 Score (All Features): {baseline_f1_score}")

# Feature ablation study by removing 1 feature at a time
features_to_ablate = ['token', 'lemma', 'pos', 'lexicalized_pos', 'prev_pos', 'next_pos', 'cue', 'constituency_distance', 'same_clause', 'same_phrase', 'is_punct', 'sentence_position', 'is_negation_cue', 'token_distance', 'dependency_type', 'dependency_head','distance_to_root', 'distance_to_cue']

for feature in features_to_ablate:
    f1_score, feature_predictions = train_and_evaluate(
        [extract_features(sentence, exclude_feature=feature) for sentence in train_sentences],
        y_train,
        [extract_features(sentence, exclude_feature=feature) for sentence in dev_sentences],
        y_dev
    )
    print(f"F1 Score without '{feature}': {round(f1_score, 3)}, Difference: {round(baseline_f1_score - f1_score, 3)}")
    output_file_path = f'predictions_without_{feature}.tsv'
    write_predictions_to_file(dev_file_path, feature_predictions, output_file_path)