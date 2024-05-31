import csv
import os

def calculate_corpus_statistics(input_file_path):
    """
    Calculates and prints statistics for the files within the corpus.
    Params:
        input_file_path (str): path to the input TSV file.
    Returns: None
    """
    num_tokens = 0
    num_sentences = 0
    num_negated_sentences = 0
    num_negation_cues = 0
    num_scopes = 0
    unique_cues = set()
    sentences_without_scope = []

    file_name = os.path.basename(input_file_path)
    
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        tsv_reader = csv.reader(infile, delimiter='\t')
        sentence_ids = set()
        current_sentence_id = ''
        sentence_has_scope = False
        sentence_has_negation = False
        negation_cues_in_sentence = []

        for row in tsv_reader:
            if row:
                sentence_id = f"{row[0]}_{row[1]}" # Unique sentence ID by joining chapter name and sentence ID
                num_tokens += 1

                if sentence_id != current_sentence_id:
                    if current_sentence_id:
                        if sentence_has_negation: # Check if sentence is negated
                            num_negated_sentences += 1
                            if not sentence_has_scope: 
                                sentences_without_scope.append(current_sentence_id) # If negated but has no scope, keep track of it

                            # Process negation cues for the complete sentence
                            if len(negation_cues_in_sentence) == 1:  # For single word negation cues, we can just add them to the set
                                unique_cues.add(negation_cues_in_sentence[0].lower())
                            elif len(negation_cues_in_sentence) > 1: # For multi-word negation cues, we join them with an underscore e.g. by_no_means
                                combined_cue = '_'.join(negation_cues_in_sentence)
                                unique_cues.add(combined_cue.lower())
                        if sentence_has_scope:
                            num_scopes += 1
                    
                    # Reset variables for new sentence
                    current_sentence_id = sentence_id
                    sentence_has_negation = False
                    sentence_has_scope = False
                    negation_cues_in_sentence = []
                    sentence_ids.add(sentence_id)

                # Check for negation cues and scopes
                if row[7] != '***' and row[7] != '_':
                    num_negation_cues += 1
                    sentence_has_negation = True
                    negation_cues_in_sentence.append(row[7])
                if row[8] == 'IS':
                    sentence_has_scope = True

        # Update counters with last sentence
        if sentence_has_negation:
            num_negated_sentences += 1
            if not sentence_has_scope:
                sentences_without_scope.append(current_sentence_id)
            if sentence_has_scope:
                num_scopes += 1

            if len(negation_cues_in_sentence) == 1:
                unique_cues.add(negation_cues_in_sentence[0])
            elif len(negation_cues_in_sentence) > 1:
                combined_cue = '_'.join(negation_cues_in_sentence)
                unique_cues.add(combined_cue.lower())

        # Update total number of sentences
        num_sentences = len(sentence_ids)

        # Calculate percentages
        percentage_negated_sentences = (num_negated_sentences / num_sentences) * 100 if num_sentences else 0
        percentage_unique_cues = len(unique_cues)

    # Print statistics
    print(f"Statistics for {file_name}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of negated sentences: {num_negated_sentences}")
    print(f"Percentage of negated sentences: {percentage_negated_sentences:.2f}%")
    print(f"Number of negation cues: {num_negation_cues}")
    print(f"Number of unique cues: {percentage_unique_cues}")
    print(f"Number of scopes: {num_scopes}")
    print(f"Sentences with negation cue but no scope: {sentences_without_scope}\n")

calculate_corpus_statistics('data/preprocessed_training.tsv')
calculate_corpus_statistics('data/preprocessed_dev.tsv')
calculate_corpus_statistics('data/preprocessed_test_combined.tsv')

