import csv
import os
import re

def process_and_save_data(input_file_path, output_file_path):
    """
    Processes the input data, extracts negation information, duplicates sentences with more than one negation,
    and saves the processed data to a TSV file.
    Params:
        input_file_path (str): path to input data.
        output_file_path (str): path to output TSV file where the processed data will be stored.
    Returns: None
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
            open(output_file_path, 'w', encoding='utf-8') as outfile:

        tsv_reader = csv.reader(infile, delimiter='\t')
        tsv_writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')

        sentences = dict()
        sentence_repetitions = dict()  # Track repetitions of sentences with negations


        # Read the file and group rows by sentences
        for row in tsv_reader:
            if len(row) == 0:
                continue
            chapter = row[0]
            sent_num = row[1]
            sentence_key = (chapter, sent_num)

            if sentence_key not in sentences:
                sentences[sentence_key] = []
            sentences[sentence_key].append(row)

        # The block below (36-88) is partially retrieved from GPT-4 and edited for our specific needs. 
        # The input was previously existing code which worked for sentences with max 3 negations.
        # Process each sentence
        for sentence_key, rows in sentences.items():
            # Check if the sentence has no negation cues.
            # This is determined by two conditions:
            # 1. Each row (token) in the sentence must have exactly 8 columns.
            #    This implies it does not have extra columns for additional negation cues.
            # 2. The 8th column (index 7) in each row must be '***'.
            #    The '***' indicates that there is no negation cue for this particular token.
            # The 'all' function ensures that both conditions are true for every row in the sentence.
            if all(len(row) == 8 and row[7] == '***' for row in rows):
                # If the above conditions are met for all rows in the sentence,
                # it means the sentence has no negation cues.
                # Therefore, we write each row of the sentence to the output file as is.
                # It copies the entire sentence to the output file unchanged.
                for row in rows:
                    parse_tree = re.sub(r'\*', f' {row[3]}', row[6])  # Process the parse tree information for NLTK compatibility
                    tsv_writer.writerow(row[:6] + [parse_tree] + [row[7]] + ['OS', '_'])
                # Add an empty line after each sentence
                tsv_writer.writerow([])

            else:
                # This block handles sentences that have negation cues.
                # We iterate over the columns to find and process negation cues dynamically.
                num_columns = len(rows[0])  # Get the number of columns (which is the same for all rows in the sentence)
                cue_indices = [i for i in range(7, num_columns, 3)]  # Calculate the indices of negation cues

                for cue_index in cue_indices:
                    # Check if any row in the sentence has a negation cue at the current index.
                    # Conditions for a negation cue:
                    # 1. The row has enough columns to include the cue index.
                    # 2. The value at the cue index is not '_' (no negation cue).
                    if any(len(row) > cue_index and row[cue_index] != '_' for row in rows):
                        # If the above condition is true, it means there is at least one negation cue
                        # at the current index in the sentence.
                        if sentence_key not in sentence_repetitions:
                            sentence_repetitions[sentence_key] = 1  # Initialize count
                        else:
                            sentence_repetitions[sentence_key] += 1  # Increment count for each repetition

                        repetition_count = sentence_repetitions[sentence_key]

                        # We then iterate over each row in the sentence to process this negation cue.
                        for row in rows:
                            negation_cue = row[cue_index] if row[cue_index] != '_' else '_'
                            scope = 'IS' if row[cue_index + 1] != '_' else 'OS'
                            negated_property = row[cue_index + 2] if row[cue_index + 2] != '_' else '_'
                            parse_tree = re.sub(r'\*', f' {row[3]}', row[6])  # Process the parse tree information for NLTK compatibility
                            modified_sent_num = f'{row[1]}_{repetition_count}'  # Modify sentence ID with repetition count
                            # Write the row to the output file with the extracted negation data.
                            tsv_writer.writerow(row[:1] + [modified_sent_num] + row[2:6] + [parse_tree] + [negation_cue, scope, negated_property])
                        tsv_writer.writerow([])  # Add an empty row after the sentence

def combine_test_files(input_file_paths, output_file_path):
    """
    Combines two preprocessed TSV test files into one, keeping the empty lines between sentences.
    Params:
        input_file_paths (list of str): List of paths to the two input test files.
        output_file_path (str): Path to the output combined TSV file.
    Returns: None
    """
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        tsv_writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')

        for input_file_path in input_file_paths:
            with open(input_file_path, 'r', encoding='utf-8') as infile:
                tsv_reader = csv.reader(infile, delimiter='\t')
                current_sentence = []
                for row in tsv_reader:
                    if row:  # Check if the row is not empty
                        current_sentence.append(row)
                    else:
                        if current_sentence:
                            for sentence_row in current_sentence:
                                tsv_writer.writerow(sentence_row)
                            tsv_writer.writerow([])  # Add an empty line after each sentence
                            current_sentence = []


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

data_annotation_path = 'data/raw_data/dev_to_annotate.tsv'
out_annotation_path = 'data/dev_to_annotate_preprocessed.tsv'

data_training_path = 'data/raw_data/SEM-2012-SharedTask-CD-SCO-training-09032012.txt'
out_training_path = 'data/preprocessed_training.tsv'

data_dev_path = 'data/raw_data/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt'
out_dev_path = 'data/preprocessed_dev.tsv'

data_test_path_1 = 'data/raw_data/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'
out_test_path_1 = 'data/preprocessed_test1_gold.tsv'

data_test_path_2 = 'data/raw_data/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt'
out_test_path_2 = 'data/preprocessed_test2_gold.tsv'

combined_test_path = 'data/preprocessed_test_combined.tsv'

# Process the data and save it to the new file
process_and_save_data(data_annotation_path, out_annotation_path)
process_and_save_data(data_dev_path, out_dev_path)
process_and_save_data(data_training_path, out_training_path)
process_and_save_data(data_test_path_1, out_test_path_1)
process_and_save_data(data_test_path_2, out_test_path_2)

# Combine the two preprocessed test files
combine_test_files([out_test_path_1, out_test_path_2], combined_test_path)

# Remove the 2 separate test files
files_to_remove = [
    'data/preprocessed_test1_gold.tsv',
    'data/preprocessed_test2_gold.tsv']
remove_files(files_to_remove)





