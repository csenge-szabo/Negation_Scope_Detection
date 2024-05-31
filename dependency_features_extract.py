import spacy
import pandas as pd
import networkx as nx
import re
import os
from spacy.language import Language
from spacy.util import filter_spans

# We utilised this blog to be informed about extracting dependency features: https://towardsdatascience.com/how-to-find-shortest-dependency-path-with-spacy-and-stanfordnlp-539d45d28239

# This function was inspired by this blog: https://spacy.io/usage/processing-pipelines. GPT4 was utilised to handcraft this function for our specific needs of handling hyphenated words.
@Language.component("merge_hyphenated_compounds")
def merge_hyphenated_compounds(doc):
    """
    Custom spaCy pipeline component to merge hyphenated compounds into single tokens.
    This function iterates through the tokens of a spaCy document. When it finds a hyphenated compound, it merges the tokens forming the compound into a single token.
    Param doc: A spaCy Doc object representing the processed text.
    Returns: The modified spaCy Doc object with hyphenated compounds merged.
    """
    # Regular expression pattern to find hyphenated compounds
    pattern = re.compile(r'\b\w+-\w+\b')
    spans_to_merge = []

    # Iterate over the tokens in the document
    for i in range(1, len(doc) - 1):
        if doc[i].text == '-':                                # Check if the current token is a hyphen
            compound = f"{doc[i-1].text}-{doc[i+1].text}"     # Form a compound string with the tokens surrounding the hyphen
            if pattern.match(compound):                       # If the compound matches the hyphenated pattern, add it to the list
                spans_to_merge.append(doc[i-1 : i+2])

    filtered_spans = filter_spans(spans_to_merge)             # Use filter_spans to remove any overlapping spans

    with doc.retokenize() as retokenizer:                     # Merge identified spans
        for span in filtered_spans:
            retokenizer.merge(span)
    return doc

def extract_first_negation_cue(sentence_cues):
    """
    Extracts the first negation cue from a list of sentence cues.
    Param sentence_cues (list): A list of negation cues in a sentence.
    Returns (str): The first negation cue if available, otherwise None.
    """
    return sentence_cues[0] if sentence_cues else None

def preprocess_text(sentence):
    """
    Preprocesses a sentence by normalizing quotation marks and handling special tokens.
    Param sentence (str): The sentence to preprocess.
    Returns (str): The preprocessed sentence.
    """
    sentence_tokens = sentence.split()
    processed_tokens = []
    for token in sentence_tokens:
        if token == 'No.':
            token = token.replace('No.', 'Number')
        elif token == '``':
            token = token.replace('``', '"')
        # Replace a number followed by a period with just the number e.g. 1. -> 1
        token = re.sub(r'^(\d+)\.$', r'\1', token)
        # Replace a token that starts with a quotation mark followed by digits e.g. '86 -> 86
        token = re.sub(r"^'(\d+)$", r'\1', token)
        # Replace a token that starts with letters followed by a quotation mark -> Shalley' -> Shalley
        token = re.sub(r"^'([a-zA-Z]+)$", r'\1', token)
        processed_tokens.append(token)

    sentence = ' '.join(processed_tokens)
    return sentence

def map_spacy_tokens_to_df_tokens(spacy_doc, df_tokens):
    """
    Maps spaCy tokens to DataFrame tokens, accounting for potential differences in tokenization.
    Params:
    - spacy_doc (spaCy Doc): A spaCy Doc object containing the processed text.
    - df_tokens (list): A list of tokens from the DataFrame.
    Returns:
    - dict: A dictionary mapping DataFrame tokens to spaCy tokens.
    """
    mapping = {}
    spacy_index = 0

    # Iterate over each token in the dataframe token list
    for df_token in df_tokens:
        # Compare the current SpaCy token with the dataframe token until a matching token is found
        while spacy_index < len(spacy_doc) and spacy_doc[spacy_index].text.lower() != df_token.lower():
            spacy_index += 1
        # If a matching token is found, add it to the mapping dictionary
        if spacy_index < len(spacy_doc):
            mapping[df_token] = spacy_doc[spacy_index]
            spacy_index += 1
        else:
            # if no matching token is found, map the dataframe token to None
            mapping[df_token] = None

    return mapping

def calculate_dependency_features(group, mapping, first_negation_cue, spacy_doc):
    """
    Calculates dependency features for tokens in a DataFrame group based on spaCy analysis.
    Params:
    - group (pandas DataFrame group): A group of rows from the DataFrame representing a sentence.
    - mapping (dict): A dictionary mapping DataFrame tokens to spaCy tokens.
    - first_negation_cue (str): The first negation cue in the sentence.
    - spacy_doc (spaCy Doc): A spaCy Doc object containing the processed text.
    Returns:
    - pandas DataFrame group: The modified DataFrame group with new dependency features added.
    """
    # Create a graph of token dependencies using NetworkX. This graph is used to calculate the shortest path between tokens.
    edges = [(token.i, child.i) for token in spacy_doc for child in token.children]
    graph = nx.Graph(edges)

    # Find the SpaCy token corresponding to the first negation cue in the sentence
    first_negation_cue_token = None
    if first_negation_cue:
        first_negation_cue_token = next((token for token in spacy_doc if token.text == first_negation_cue), None)

    # Iterate over each row in the group representing a token in the sentence
    for index, row in group.iterrows():
        original_token = row['Token']
        spacy_token = mapping.get(original_token)
        
        # If corresponding SpaCy token is found, calculate dependency features 
        if spacy_token:
            # Assign dependency features based on SpaCy's analysis
            group.at[index, 'Dependency Label'] = str(spacy_token.dep_)
            group.at[index, 'Head Token'] = str(spacy_token.head.text)
            # Calculate the path length from the current token to the root of the dependency tree
            path_length_to_root = len(list(spacy_token.ancestors))
            group.at[index, 'Path Length to Root'] = path_length_to_root

            # Calculating path length to fist negation cue if any
            if first_negation_cue_token:
                try:
                    path_length_to_negation = len(nx.shortest_path(graph, source=spacy_token.i, target=first_negation_cue_token.i)) - 1
                    group.at[index, 'Path Length to Negation Cue'] = path_length_to_negation
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    group.at[index, 'Path Length to Negation Cue'] = None
            else:
                group.at[index, 'Path Length to Negation Cue'] = None
        else:
            # Handling tokens not found in SpaCy doc
            group.at[index, 'Dependency Label'] = None
            group.at[index, 'Head Token'] = None
            group.at[index, 'Path Length to Root'] = None
            group.at[index, 'Path Length to Negation Cue'] = None
    return group

def custom_sort_key(sentence_id):
    """
    Sort sentence IDs similar to the input data.
    """
    # Splitting the sentence_id by underscores and converting each part to an integer
    parts = map(int, sentence_id.split('_'))
    return tuple(parts)

def process_file(input_file_path, output_file_path):
    """
    Processes an input file to calculate and add dependency features, then saves to an output file.
    Params:
    - input_file_path (str): Path to the input TSV file.
    - output_file_path (str): Path where the output TSV file will be saved.
    """
    # Read the DataFrame
    df = pd.read_csv(input_file_path, sep="\t", header=None, names=["Chapter", "SentenceID", "TokenID", "Token", "LemmatizedToken", "POS", "constituency", "Cue", "Label", "Focus", "Syntactic Distance", "isSameClause", "isSamePhrase", "isPuct", "SentencePosition", "isNegationCue", "Token Distance"])

    # Initialize new columns in df
    df['Dependency Label'] = pd.NA
    df['Head Token'] = pd.NA
    df['Path Length to Root'] = pd.NA
    df['Path Length to Negation Cue'] = pd.NA

    processed_rows = []

    # Group by Chapter and SentenceID to reconstruct each sentence
    grouped_df = df.groupby(['Chapter', 'SentenceID'])

    # Processing each sentence separately in the dataframe
    for (chapter, sentence_id), group in grouped_df:
        sentence_tokens = [str(token) for token in group['Token'].tolist()]                # Extract tokens for current sentence
        processed_tokens = [preprocess_text(token) for token in sentence_tokens]
        processed_sentence = ' '.join(processed_tokens)

        spacy_doc = nlp(processed_sentence)                                               # Process the sentence with SpaCy to get a Doc object
        token_mapping = map_spacy_tokens_to_df_tokens(spacy_doc, processed_tokens)        # Map the dataframe tokens to SpaCy tokens for alignment

        current_sentence_cues = [row['Cue'] for index, row in group.iterrows() if row['Cue'] != '_' and row['Cue'] != '***'] # Extract negation cues from the current sentence
        first_negation_cue = extract_first_negation_cue(current_sentence_cues)

        processed_group = calculate_dependency_features(group, token_mapping, first_negation_cue, spacy_doc) # Calculate dependency features for the sentence
        processed_rows.extend(processed_group.values)

    # Convert processed rows to DataFrame
    processed_df = pd.DataFrame(processed_rows, columns=df.columns)
    
    # Convert values to integers and fill in missing values by -1
    processed_df['Path Length to Root'] = processed_df['Path Length to Root'].fillna(-1).astype(int)
    processed_df['Path Length to Negation Cue'] = processed_df['Path Length to Negation Cue'].fillna(-1).astype(int)

    # Write the DataFrame to a TSV file
    processed_df.to_csv(output_file_path, sep='\t', index=False, header=None, na_rep='NA')
    print(f"Data saved to: {output_file_path}")
    
def remove_files(file_paths):
    """
    Remove a list of files.
    Param: file_paths (list of str): list of file paths to be removed.
    Returns: None
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File removed: {file_path}")
        else:
            print(f"File {file_path} does not exist.")

# Load spaCy model and add the custom component
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("merge_hyphenated_compounds", after='ner')

file_paths = [
    ("data/with_constituency_features_training.tsv", "data/with_complete_features_training.tsv"),
    ("data/with_constituency_features_dev.tsv", "data/with_complete_features_dev.tsv"),
    ("data/with_constituency_features_test.tsv", "data/with_complete_features_test.tsv")]

for input_path, output_path in file_paths: # Extracting dependency features
    process_file(input_path, output_path)

# Remove the files containing only constituency features
files_to_remove = [
    'data/with_constituency_features_dev.tsv',
    'data/with_constituency_features_training.tsv', 
    'data/with_constituency_features_test.tsv']
remove_files(files_to_remove)

