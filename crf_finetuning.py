import pandas as pd
import pycrfsuite

def read_and_format_data(file_path):
    """
    Reads data from a TSV file and formats it into a list of sentences.
    Args: file_path (str): Path to the input TSV file. 
    Returns: List: A list of sentences, where each sentence is a list of token information.
    """
    # Define the column names for the TSV file
    columns = ['doc_id', 'sentence_num', 'token_num', 'token', 'lemma', 'pos', 'syntax_tree', 'cue', 'label', 'focus', 'constituency_distance', 'same_clause', 'same_phrase', 'is_punct', 'sentence_position', 'is_negation_cue', 'token_distance', 'dependency_type', 'dependency_head', 'distance_to_root', 'distance_to_cue']
    
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

def extract_features(sentence):
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

        sentence_features.append(features)

    return sentence_features

def extract_labels(sentence):
    """
    Extracts labels from a sentence for use in CRF model training and evaluation.
    Args:sentence (list): A list of token information for a single sentence.
    Returns:list: A list of labels corresponding to each token in the sentence.
    """
    return [label for token, lemma, pos, cue, constituency_distance, same_clause, same_phrase, is_punct, sentence_position, is_negation_cue, token_distance, dependency_type, dependency_head, distance_to_root, distance_to_cue, label in sentence]

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

def train_crf(X_train, y_train, c1, c2):
    """
    Trains a Conditional Random Fields (CRF) model with specified hyperparameters.
    Args:
        X_train (list): A list of feature sequences for training.
        y_train (list): A list of label sequences for training.
        c1 (float): L1 regularization parameter.
        c2 (float): L2 regularization parameter.
    """
    # Create a CRF trainer with optional verbosity for training progress
    trainer = pycrfsuite.Trainer(verbose=True)
    
    # Iterate over each training example and add it to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    # Set the hyperparameters for the CRF model
    trainer.set_params({
        'c1': c1,  # L1 regularization parameter
        'c2': c2   # L2 regularization parameter
    })
    
    # Train the CRF model and save it to 'crf.model'
    trainer.train('crf.model')

def evaluate_model(X_dev, y_dev):
    """
    Evaluates a trained CRF model on the development data.
    Args:
        X_dev (list): A list of feature sequences for development.
        y_dev (list): A list of label sequences for development.  
    Returns: tuple: A tuple containing precision, recall, and F1-score.
    """
    # Create a CRF tagger to load the trained model
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    
    # Use the trained CRF model to predict labels for development data
    y_pred = [tagger.tag(xseq) for xseq in X_dev]
    
    # Close the tagger after prediction
    tagger.close()

    # Flatten the nested lists of true labels and predicted labels
    y_dev_flat = [label for sentence in y_dev for label in sentence]
    y_pred_flat = [label for sentence in y_pred for label in sentence]
    
    # Calculate precision, recall, and F1-score using the true and predicted labels
    return calculate_metrics(y_dev_flat, y_pred_flat)

# Load training and development data
train_file_path = 'data/with_complete_features_training.tsv'
dev_file_path = 'data/with_complete_features_dev.tsv'

train_sentences = read_and_format_data(train_file_path)
dev_sentences = read_and_format_data(dev_file_path)

# Applying feature extraction to the training and development data
X_train = [extract_features(sentence) for sentence in train_sentences]
y_train = [extract_labels(sentence) for sentence in train_sentences]
X_dev = [extract_features(sentence) for sentence in dev_sentences]
y_dev = [extract_labels(sentence) for sentence in dev_sentences]

# Define the hyperparameter grid
param_grid = {
    'c1': [0.01, 0.1, 1, 10],
    'c2': [0.01, 0.1, 1, 10]
}

# Grid search over the parameter grid
best_score = 0
best_params = None
for c1 in param_grid['c1']:
    for c2 in param_grid['c2']:
        train_crf(X_train, y_train, c1, c2)
        precision, recall, f1_score = evaluate_model(X_dev, y_dev)
        print(f"Training with c1={c1}, c2={c2} | Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")
        if f1_score > best_score:
            best_score = f1_score
            best_params = {'c1': c1, 'c2': c2}

print(f"Best parameters: {best_params}, F1-Score: {best_score}")

