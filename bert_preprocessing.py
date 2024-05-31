import csv

def process_and_save_data(input_file_path, output_file_path):
    """
    Process a TSV file by reading each sentence, apply modifications based on negation cues, and write the processed data to a new TSV file.
    Params:
        - input_file_path (str): Path to the input TSV file.
        - output_file_path (str): Path for the output TSV file.
    Returns: None
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
            open(output_file_path, 'w', encoding='utf-8') as outfile:

        tsv_reader = csv.reader(infile, delimiter='\t')
        tsv_writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')

        current_sentence = []

        for row in tsv_reader:
            if len(row) == 0:
                # Empty line indicates the end of a sentence
                if current_sentence:
                    process_sentence(current_sentence, tsv_writer)
                    current_sentence = []
                    tsv_writer.writerow([])  # Write the empty line to separate sentences
                continue

            current_sentence.append(row)

        # Process the last sentence if the file doesn't end with an empty line
        if current_sentence:
            process_sentence(current_sentence, tsv_writer)


def process_sentence(rows, tsv_writer):
    """
    Process a sentence by modifying its tokens based on the presence and type of negation cues.
    It modifies tokens in a sentence with 'CUE[0]', 'CUE[1]', or 'CUE[2]' prefixes depending on the type and number of negation cues present in the sentence.
    Params:
        - rows (list of lists): The sentence to process, represented as a list of token rows.
        - tsv_writer (csv.writer): The CSV writer object to write the processed sentence.
    Returns: None
    """
    # Count negation cues and store their positions
    negation_cues = [(index, row[7]) for index, row in enumerate(rows) if row[7] != '***' and row[7] != '_']

    if len(negation_cues) == 0:
        # No negation cues, write the sentence as it is
        for row in rows:
            tsv_writer.writerow(row)
    
    elif len(negation_cues) == 1:
        # Single negation cue
        cue_index, cue_token = negation_cues[0]
        for index, row in enumerate(rows):
            if index == cue_index and row[3] == cue_token:
                # Negation cue matches the token e.g. 'not'
                row[3] = 'CUE_0 ' + row[3]
            elif index == cue_index:
                # Negation cue is as an affix e.g. 'im' in 'impossible'
                row[3] = 'CUE_1 ' + row[3]
            tsv_writer.writerow(row)
    
    else:
        # Multiple negation cues e.g. 'by no means', 'neither'...'nor'
        cue_tokens = [cue for _, cue in negation_cues]
        for index, row in enumerate(rows):
            if row[3] in cue_tokens:
                row[3] = 'CUE_2 ' + row[3]

            tsv_writer.writerow(row)

process_and_save_data('data/preprocessed_dev.tsv', 'data/bert_input_dev.tsv')
process_and_save_data('data/preprocessed_training.tsv', 'data/bert_input_training.tsv')
process_and_save_data('data/preprocessed_test_combined.tsv', 'data/bert_input_test.tsv')


