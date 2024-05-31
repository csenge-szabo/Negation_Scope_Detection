import pandas as pd
from nltk import Tree
import string

def is_punctuation(token):
    """
    Check if a token is a punctuation mark.
    """
    other_punct_marks = ['``', "''", '--']
    return 1 if token in string.punctuation or token in other_punct_marks else 0

# We used this resource for creating this function: https://www.nltk.org/book/ch07.html
def extract_constituency_features(tree, negation_index):
    """
    Calculate syntactic constituency distance, clause membership, and phrase membership for each token in a sentence relative to a negation cue.
    :param tree: the parse tree of the sentence (tree).
    :param negation_index: the index of the negation cue in the sentence (int).
    :return three lists containing the constituency distances, clause memberships, and phrase memberships of each token in a sentence (tuple).
    """
    distances, clause_memberships, phrase_memberships = [], [], []

    negation_path = tree.leaf_treeposition(negation_index)            # Find the tree position of the negation cue
    negation_clause = find_lowest_S_ancestor(tree, negation_path)     # Find the lowest 'S' ancestor for the negation cue
    negation_phrase = negation_path[:-1]                              # Find the immediate parent (phrase) of the negation cue

    for i in range(len(tree.leaves())):
        token_path = tree.leaf_treeposition(i)                          # Find the tree position for each token
        token_clause = find_lowest_S_ancestor(tree, token_path)         # Find the lowest 'S' ancestor for the token
        token_phrase = token_path[:-1]                                  # Find the immediate parent (phrase) of the token

        # Calculate constituency distance
        common_length = min(len(negation_path), len(token_path))
        for j in range(common_length):
            if negation_path[j] != token_path[j]:
                break
        distance = (len(negation_path[j:]) - 1) + (len(token_path[j:]) - 1)
        distances.append(distance)

        # Determine if token is in the same clause as negation cue
        clause_memberships.append(1 if negation_clause == token_clause else 0)

        # Determine if token is in the same phrase as negation cue
        phrase_memberships.append(1 if negation_phrase == token_phrase else 0)

    return distances, clause_memberships, phrase_memberships

def find_lowest_S_ancestor(tree, path):
    """
    Find the lowest ancestor in a parse tree that corresponds to a sentence clause ('S').
    param tree: parse tree of the sentence (tree).
    param path: tree path to a specific token (tuple).
    return: tree path to the lowes 'S' ancestor of the specified token.
    """
    # Iterate over the path in reverse to find the nearest 'S' ancestor
    for depth, position in enumerate(reversed(path), 1):
        ancestor_path = path[:-depth]
        if tree[ancestor_path].label() == 'S':
            return ancestor_path
    return None

def process_sentences(sentences):
    """
    Process each sentence and append various linguistic features for each token. These features include constituency distance, clause membership, phrase membership, punctuation, position in sentence, negation cue status, and distance to the closest negation cue. If there is no negation cue in the sentence, the distance value is set to 0 for all tokens.
    :param sentences: a list of sentences, where each sentence is a list of tokens.
    :return a list of sentences with appended linguistic information for each token.
    """
    results = []

    for sentence in sentences:
        # Collect the parse tree components from the sentence
        parse_tree_str = " ".join([token[6] for token in sentence])

        # List to store indices of all negation cues
        negation_cue_indices = []
        for i, token in enumerate(sentence):
            if token[7] != '***' and token[7] != '_':
                negation_cue_indices.append(i)

        try:
            # Create a parse tree from the sentence
            tree = Tree.fromstring(parse_tree_str.strip())

            # Initialize constituency feature lists
            distances, clause_memberships, phrase_memberships = [], [], []

            # Calculate constituency features for the first negation cue, if any
            first_negation_cue_index = negation_cue_indices[0] if negation_cue_indices else None
            if first_negation_cue_index is not None:
                distances, clause_memberships, phrase_memberships = extract_constituency_features(tree, first_negation_cue_index)

            # Process each token in the sentence
            for i, token in enumerate(sentence):
                token_text = token[3]

                # Determine if the token is a punctuation mark
                punctuation = is_punctuation(token_text)

                # Determine the token's position in the sentence (beginning, inside, end)
                position = 'BOS' if i == 0 else 'EOS' if i == len(sentence) - 1 else 'INS'

                # Get constituency features for the token, use default values if no negation cue is present
                distance = distances[i] if distances else 0
                clause_membership = clause_memberships[i] if clause_memberships else 0
                phrase_membership = phrase_memberships[i] if phrase_memberships else 0

                # Check if the token is a negation cue
                is_negation_cue = 1 if i in negation_cue_indices else 0

                # Calculate the distance from the closest negation cue
                # For the next 5 lines  GPT-4 was utilised to find out how to measure distance to the closest negation cue instead of the first negation cue.
                # The input was existing code to the first negation cue within the sentence.
                if negation_cue_indices:
                    distance_to_closest_negation_cue = min(abs(i - cue_index) for cue_index in negation_cue_indices)
                else:
                    distance_to_closest_negation_cue = 0

                # Extend the token information with the calculated features
                token.extend([distance, clause_membership, phrase_membership, punctuation, position, is_negation_cue, distance_to_closest_negation_cue])

            # Append the updated tokens to the result
            results.extend(sentence)

            # Add an empty line to separate sentences in the output
            results.append([])

        except Exception as e:
            # Handle any errors encountered during processing
            # print(f"Failed to process Sentence: {sentence}. Error: {e}")
            pass

    return results


def process_and_write_file(input_file_path, output_file_path):
    """
    Process a given input file containing sentences, extract the features and write the results with added features to an output file.
    :param input_file_path: Path to the input file containing the sentences (str).
    :param output_file_path: Path where the output file with processed sentences will be saved (str).
    """
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    sentences = []
    current_sentence = []
    for line in lines:
        if line.strip() == "":
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(line.strip().split('\t'))

    results = process_sentences(sentences)

    with open(output_file_path, 'w') as output_file:
        for result in results:
            if result:
                output_file.write("\t".join(map(str, result)) + "\n")
            else:
                output_file.write("\n")

dev_file_path = "data/preprocessed_dev.tsv"
output_dev_file_path = "data/with_constituency_features_dev.tsv"

training_file_path = "data/preprocessed_training.tsv"
output_training_file_path = "data/with_constituency_features_training.tsv"

test_file_path = "data/preprocessed_test_combined.tsv"
output_test_file_path = "data/with_constituency_features_test.tsv"

process_and_write_file(dev_file_path, output_dev_file_path)
process_and_write_file(training_file_path, output_training_file_path)
process_and_write_file(test_file_path, output_test_file_path)

