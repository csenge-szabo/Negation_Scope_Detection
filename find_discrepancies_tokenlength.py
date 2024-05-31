def read_file_tokens(file_path, is_input_file=True):
    """
    Read a given file and count the number of tokens per sentence.
    Params:
        - file_path (str): The path to the file to be read.
        - is_input_file (bool): Expression to distinguish between input and output file formats.
    Returns:
        dict: A dictionary mapping sentence IDs to their respective token counts.
    """
    sentence_tokens_count = {}
    current_sentence_id = None
    token_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():  # Check for empty lines indicating new sentences or end of file
                if current_sentence_id:
                     # Save the token count for the current sentence before moving to the next
                    sentence_tokens_count[current_sentence_id] = token_count
                current_sentence_id = None
                token_count = 0
            
            else:
                parts = line.strip().split('\t')
                # Different handling based on file type (input vs output)
                if is_input_file and len(parts) >= 3:  # BERT input file structure
                    chapter, sent_id, *_ = parts
                    sentence_id = f"{chapter}_{sent_id}"
                elif not is_input_file and len(parts) >= 1:  # BERT output file structure
                    sentence_id, *_ = parts
                else:
                    print('Warning: unexpected function input!')
                
                # Check if it is still the same sentence
                if current_sentence_id != sentence_id:
                    if current_sentence_id:
                        # Save the token count for the previous sentence
                        sentence_tokens_count[current_sentence_id] = token_count
                    current_sentence_id = sentence_id
                    token_count = 1  # Reset token count for new sentence
                else:
                    token_count += 1

        # check if the last sentence has been handled
        if current_sentence_id:
            # This handles the case where the file doesn't end with an empty line
            sentence_tokens_count[current_sentence_id] = token_count

    return sentence_tokens_count

def compare_token_counts(input_file, output_file):
    """
    Compare the token counts for each sentence between two files. Print sentences where token counts differ.
    Params:
        - input_file (str): Path to the input file.
        - output_file (str): Path to the output file.
    Returns: None
    """
    input_counts = read_file_tokens(input_file, is_input_file=True)
    output_counts = read_file_tokens(output_file, is_input_file=False)

    # Compare token counts for each sentence ID.
    for sentence_id, input_count in input_counts.items():
        output_count = output_counts.get(sentence_id, 0)
        if input_count != output_count:
            print(f"Sentence ID: {sentence_id} has {input_count} tokens in input file but {output_count} tokens in output file.")

input_file = 'data/bert_input_test.tsv'
output_file = 'data/bert_postprocessed_testset.tsv'

# input_file = 'data/bert_input_dev.tsv'
# output_file = 'data/bert_postprocessed_devset.tsv'

compare_token_counts(input_file, output_file)
