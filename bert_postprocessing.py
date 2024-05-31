def process_and_write_sentences(input_file, output_file):
    """
    Read sentences from an input file, process each sentence to merge tokens and handle special cases, write the processed sentences to an output file.
    Parameters:
        - input_file (str): Path to the file containing unprocessed sentence tokens.
        - output_file (str): Path to the file where processed sentence tokens will be written.
    Return: None
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    current_sentence_id = ""
    current_sentence_tokens = []
    processed_data = []

    for line in lines:
        # Check for empty lines indicating the end of a sentence
        if not line.strip():
            # Process the tokens of the current sentence
            if current_sentence_tokens:
                processed_sentence = process_sentence(current_sentence_tokens, current_sentence_id)
                processed_data.extend(processed_sentence)
                # processed_data.append(("\n"))  # Add an empty row to separate sentences
                current_sentence_tokens = []  # Reset for the next sentence
            continue

        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue

        sent_id, token, gold_label, pred_label = parts

        # Check if the current row belongs to a new sentence
        if sent_id != current_sentence_id:
            if current_sentence_tokens:
                processed_sentence = process_sentence(current_sentence_tokens, current_sentence_id)
                processed_data.extend(processed_sentence)
                
            current_sentence_id = sent_id
            current_sentence_tokens = []

        current_sentence_tokens.append((token, gold_label, pred_label)) # Add the current token to the list of tokens for the current sentence

    if current_sentence_tokens:
        processed_sentence = process_sentence(current_sentence_tokens, current_sentence_id)
        processed_data.extend(processed_sentence)

    with open(output_file, "w") as f:
        for line in processed_data:
            if line:
                f.write('\t'.join(line) + "\n")
            else:
                f.write("\n")

def process_sentence(sentence_tokens, sentence_id):
    """
    Process a list of tokens for a given sentence, handle special cases such as merging tokens, handle abbreviations and correct punctuation.
    Parameters:
        - sentence_tokens (list of tuples): The tokens of the sentence along with their labels.
        - sentence_id (str): The identifier of the sentence being processed.
    Return: list of tuples: The processed tokens of the sentence, with special cases handled.
    """
    processed_tokens = []
    contractions_list = ['ll', 's', 'm', 're', 've', 'well', 'd']
    abbreviations_list = ["mr", "dr", "mrs", "ms", "co", "rev"]
    
    i = 0 # Index to keep track of the current position in the list of tokens
    while i < len(sentence_tokens):
        token, gold_label, pred_label = sentence_tokens[i]

        # Skip "cue_X" patterns
        if token == "cue" and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] == '_':
            i += 3
            continue

        # Attach period to the preceding number e.g. 1.
        elif token == "." and i > 0 and i < len(sentence_tokens) - 1 and processed_tokens[-1][1].isdigit():
            prev_sentence_id, prev_token, prev_gold, prev_pred = processed_tokens.pop()
            processed_tokens.append((prev_sentence_id, prev_token + token, prev_gold, prev_pred))
        
        # Merge single quotation mark followed by a number e.g. '86
        elif token == "'" and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0].isdigit():
            next_token = sentence_tokens[i + 1][0]
            merged_token = token + next_token
            processed_tokens.append((sentence_id, merged_token, gold_label, pred_label))
            i += 1 
        
        # Handling time markers like "p.m." and special cases like "T.A."
        # For the block of code from line 92-109 GPT 4 was utilized, whereby the prompt was the existing script
        # The prompt included the problematic examples in the data which needed addressing in the output
        # The code was then tailored to our specific needs
        elif i + 3 < len(sentence_tokens) and \
           sentence_tokens[i + 1][0] == "." and sentence_tokens[i + 3][0] == ".":
            
            # Process "t.a." as separate tokens, circle01_324_1
            if token.lower() == "t" and sentence_tokens[i + 2][0].lower() == "a":
                processed_tokens.append((sentence_id, token + '.', gold_label, pred_label))
                i += 1  # Skip only the period after "t"
            
            # Check if the next letter is the same or specific cases like "p.m."
            elif len(token) == 1 and token.isalpha() and \
            len(sentence_tokens[i + 2][0]) == 1 and sentence_tokens[i + 2][0].isalpha():
                merged_token = token + '.' + sentence_tokens[i + 2][0] + '.'
                processed_tokens.append((sentence_id, merged_token, gold_label, pred_label))
                i += 3  # Skip the next three tokens (., next letter, .)
            elif token in ["p", "a"] and sentence_tokens[i + 2][0] == "m":
                merged_token = token + 'm.'
                processed_tokens.append((sentence_id, merged_token, gold_label, pred_label))
                i += 3  # Skip the next three tokens for specific cases like "p.m."

        # Handling regular initials like "T.A."
        elif len(sentence_tokens) > 5 and len(token) == 1 and token.isalpha() and \
           i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] == "." and \
           i + 2 < len(sentence_tokens):
            processed_tokens.append((sentence_id, token, gold_label, pred_label))  # Keep the initial as a separate token
            i += 1  
        
        # Handle common abbreviations and titles followed by a period e.g. Mrs.
        elif token in abbreviations_list and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] == ".":
            token += '.'
            i += 1  # skip the period in the next iteration
            processed_tokens.append((sentence_id, token, gold_label, pred_label))  # Append the merged abbreviation to processed tokens

        # Handling contractions e.g. 'll
        elif token == "'" and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] in contractions_list:
            next_token = sentence_tokens[i + 1][0]
            if next_token in contractions_list:
                merged_contraction = token + next_token
                processed_tokens.append((sentence_id, merged_contraction, gold_label, pred_label))
                i += 1  # Skip the next token since it's part of the contraction

        # Handle the "n't" case
        elif token.lower() == "n" and i + 2 < len(sentence_tokens) and \
             sentence_tokens[i + 1][0] == "'" and sentence_tokens[i + 2][0].lower() == "t":
            processed_tokens.append((sentence_id, "n't", gold_label, pred_label))
            i += 2

        # Handle the 'o'clock' case
        elif token == "o" and i + 2 < len(sentence_tokens) and \
             sentence_tokens[i + 1][0] == "'" and sentence_tokens[i + 2][0].lower() == "clock":
            processed_tokens.append((sentence_id, "o'clock", gold_label, pred_label))
            i += 2  # Skip the next two tokens (' and clock)
        
        # Concatenate two consecutive dashes
        elif token == "-" and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] == "-":
            processed_tokens.append((sentence_id, "--", gold_label, pred_label))
            i += 1
        
        # Concatenate subword tokens
        elif token.startswith("##"):
            processed_tokens[-1] = (processed_tokens[-1][0], processed_tokens[-1][1] + token[2:], gold_label, pred_label)
        
        # Replace two consecutive single quotes with one double quote
        elif token == "'" and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] == "'":
            processed_tokens.append((sentence_id, '"', gold_label, pred_label))
            i += 1
        elif token == "`" and i + 1 < len(sentence_tokens) and sentence_tokens[i + 1][0] == "`":
            processed_tokens.append((sentence_id, '``', gold_label, pred_label))
            i += 1
        
        # Handle hyphenation
        elif token == "-":
            if i > 0 and i < len(sentence_tokens) - 1:
                prev_sentence_id, prev_token, prev_gold, prev_pred = processed_tokens.pop()
                next_token, next_gold, next_pred = sentence_tokens[i + 1]

                # Merge the previous token, hyphen, and next token
                hyphenated_token = prev_token + token + next_token
                processed_tokens.append((sentence_id, hyphenated_token, prev_gold, pred_label))
                i += 1
            else:
                processed_tokens.append((sentence_id, token, gold_label, pred_label))
            
        # Special case: cardboard_450 ('it)
        elif token == "'" and i + 2 < len(sentence_tokens) and \
        sentence_tokens[i + 1][0].lower() == "it" and \
        sentence_tokens[i - 1][0] == "`":
            merged_token = token + sentence_tokens[i + 1][0]
            processed_tokens.append((sentence_id, merged_token, gold_label, pred_label))
            i += 1   

        else:
            processed_tokens.append((sentence_id, token, gold_label, pred_label))

        i += 1 # Move to the next token

    return processed_tokens

input_file = 'data/bert_rawoutput_testset.tsv'
output_file = 'data/bert_postprocessed_testset.tsv'

# input_file = 'data/bert_rawoutput_devset.tsv'
# output_file = 'data/bert_postprocessed_devset.tsv'

process_and_write_sentences(input_file, output_file)


