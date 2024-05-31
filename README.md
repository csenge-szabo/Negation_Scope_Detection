# Negation Scope Detection

## Authors: 
Selin Acikel, Murat Ertas, Guo Ningxuan, Csenge Szabo

## Introduction

This repository contains scripts related to negation scope detection and evaluation as described in the *SEM 2012 Shared Task. This study is the final outcome of the project work for the Applied Text Mining Methods Master's course at VU Amsterdam. Our task was to preprocess the STARSEM CD-SCO corpus, extract features from the data, use feature-based and transformer-based classifiers to carry out negation scope detection, and evaluate the models' peformance in a token-based and span-based manner.

For the full project report, read `Negation_Scope_Detection_Report.pdf`.

-------------
## STEP-BY-STEP GUIDES:

**Experiments with CRF models:**
1. Run conll_reformat.py
2. Run constituency_features_extract.py
3. Run dependency_features_extract.py
4. Run crf_model_all_features.py
5. Run evaluation_metrics.py
- Optional steps: (1) For feature ablation run crf_feature_ablation.py, (2) for fine-tuning experiments run crf_finetuning.py.
- Using selected features: Run crf_model_dep_features.py or crf_model_constituency_features.py instead of crf_model_all_features.py.

**Experiments with the fine-tuned DistilBERT model:**
1. Run conll_reformat.py
2. Run bert_preprocessing.py
3. Use bert_fine_tuning.ipynb
4. Run bert_postprocessing.py
5. Run evaluation_metrics_bert.py
- Optional steps: For error analysis run extract_errors.py.

-------------
## DATA
The 'data' folder in this repository contains the following files:

**raw_data subfolder:**
- SEM-2012-SharedTask-CD-SCO-dev-09032012.txt: The raw development data file with gold negation cues, scopes, and focuses.
- SEM-2012-SharedTask-CD-SCO-training-09032012.txt: The raw training data file with gold negation cues, scopes, and focuses.
- SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt: The raw test data file with gold negation cues, scopes, and focuses. (1)
- SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt: The raw test data file with gold negation cues, scopes and focuses. (2)
- dev_to_annotate.tsv: File with a selection of sentences from the development data for annotation purposes.

**Preprocessed files:**
- preprocessed_dev.tsv: Preprocessed development data with gold negation cues, scopes and focuses.
- preprocessed_training.tsv: Preprocessed training data with gold negation cues, scopes and focuses.
- preprocessed_test_combined.tsv: Preprocessed test data with gold negation cues, scopes and focuses. This file combines the two raw test gold files.
- preprocessed_annotation.tsv: Preprocessed file with a selection of sentences from the development data for annotation purposes.

**Processed files for transformer-based classifier input:**
- bert_input_training.tsv: Preprocessed training data with gold negation cues, scopes and focuses. Negation cue tokens are marked with a special CUE_X token.
- bert_input_dev.tsv: Preprocessed development data with gold negation cues, scopes and focuses. Negation cue tokens are marked with a special CUE_X token.
- bert_input_test.tsv: Preprocessed test data with gold negation cues, scopes and focuses. Negation cue tokens are marked with a special CUE_X token.
 
**Output files after feature extraction:**
- with_complete_features_training.tsv: Training data with gold negation cues, scopes and focuses and extracted features (lexical, constituency-based, dependency-based).
- with_complete_features_dev.tsv: Development data with gold negation cues, scopes and focuses and extracted features (lexical, constituency-based, dependency-based).
- with_complete_features_test.tsv: Test data with gold negation cues, scopes and focuses and extracted features (lexical, constituency-based, dependency-based).

**Output of the CRF models containing predicted negation scope labels:**
- crf_dev_output_all_features.tsv: File containing the predictions of the CRF model utilizing all 18 features (lexical, constituency-based, dependency-based) on dev set.
- crf_dev_output_constituency_features.tsv: File containing the predictions of the CRF model utilizing only lexical and constituency-based features on dev set.
- crf_dev_output_dep_features.tsv: File containing the predictions of the CRF model utilizing only lexical and dependency-based features on dev set.
- crf_test_output_all_features.tsv: File containing the predictions of the CRF model utilizing all 18 features (lexical, constituency-based, dependency-based) on test set.

**Output of the BERT-based model containing predicted negation scope labels before postprocessing:**
- bert_rawoutput_devset.tsv: File containing the predictions of the fine-tuned DistilBERT model on the development set.
- bert_rawoutput_testset.tsv: File containing the predictions of the fine-tuned DistilBERT model on the test set.

**Output of the BERT-based model containing predicted negation scope labels after postprocessing:**
- bert_postprocessed_devset.tsv: Postprocessed file containing the predictions of the fine-tuned DistilBERT model on the development set.
- bert_postprocessed_testset.tsv: Postprocessed file containing the predictions of the fine-tuned DistilBERT model on the test set.

**File annotated by group B:**
- dev_annotated.tsv: This file contains 17 annotated sentences -and their copies in case of more than one negation- with negation cues and negation scopes. The annotation was carried out collectively by members of Group B.

**File for error analysis:**
- erroroutput-classified.tsv: File containing the sentences with errors produced by the transformer-based model, utilized for error analysis.

-------------
## SCRIPTS

### 1. Data reformatting and analysis-related scripts:
- conll_reformat.py: 
This script can preprocess the raw TXT data, extract negation information, create copies of sentences with more than one negation, and save the processed data to a new TSV file. The script adjusts each sentence ID to be unique for sentences that contain a negation cue. The operations in the script include reducing the original 16 columns to 10 columns. Syntactical information in column 7 was processed to make it compatible for parsing with NLTK. A function is included to combine the two separate test files into one.

- data_distribution_stats.py
This script calculates and prints out statistical information for the preprocessed corpus. It computes the number of tokens, sentences, negated sentences, negation cues, unique negation cues, scopes, and percentage of negated sentences within the corpus.

### 2. Feature-based model-related scripts (CRF):
- constituency_features_extract.py:
This script extracts lexical and syntactic constituency-related features from the input file and writes a new TSV file, where a new column is added for each additional feature. It processes each sentence by parsing its syntactic structure, calculating syntactic distance, clause membership, phrase membership, punctuation, position in the sentence, negation cue status, and distance to the closest negation cue for each token. The resulting features are extracted for each token in each sentence and the sentences are written out to a new TSV file with the added features.

- dependency_features_extract.py:
This script takes a TSV file with constituency features (output of constituency_features_extract.py), processes the sentences, analyzes the dependency relationships in each sentence using SpaCy, and adds 4 additional columns to each line in the files indicating dependency relation, dependency head, dependency distance to root, dependency distance to first negation cue within the sentence.

- crf_feature_ablation.py:
This script utilizes a CRF model for negation scope detection. It reads, formats, extracts features and labels from the input data, then trains and evaluates the CRF model's performance on a token level for precision, recall, and F1 score. It also conducts a feature ablation study to assess the impact of each feature on the model's performance. This is conducted by systematically removing one feature at a time and indicating its impact on the F1 score.

- crf_finetuning.py:
This script utilizes a CRF model for negation scope detection. It reads, formats, and extracts features and labels from the input data, then trains the model, carries out scope detection on unseen data, and evaluates the CRF model's performance on a token level using precision, recall, and F1 score. The script allows for hyper-parameter tuning by performing a grid search over the 'c1' and 'c2' parameters to optimize the model's performance given the selected set of features. It chooses the best-performing hyperparameter combination based on the F1-score.

- crf_model_all_features.py:
This script utilizes a CRF model for negation scope detection. It reads, formats, and extracts features and labels from the input data, then trains the model, carries out scope detection on unseen data, and evaluates the CRF model's performance on a token level using precision, recall, and F1 score. It utilizes all eighteen extracted features, including lexical, constituency-based and dependency-based features.

- crf_model_dep_features.py:
This script utilizes a CRF model for negation scope detection. It reads, formats, and extracts features and labels from the input data, then trains the model, carries out scope detection on unseen data, and evaluates the CRF model's performance on a token level using precision, recall, and F1 score. It utilizes only a selected set of features: lexical and constituency-based features.

- crf_model_constituency_features.py:
This script utilizes a CRF model for negation scope detection. It reads, formats, and extracts features and labels from the input data, then trains the model, carries out scope detection on unseen data, and evaluates the CRF model's performance on a token level using precision, recall, and F1 score. It utilizes only a selected set of features: lexical and dependency-based features.

### 3. Transformer-based model related scripts:

- bert_fine_tuning.ipynb:
This notebook includes all the steps required for fine-tuning a DistilBERT model for the task of negation scope detection. After training, the model is evaluated and the predictions of the model are written into a TSV file.

- bert_preprocessing.py:
This script transforms a previously preprocessed TSV file (output of conll_reformat.py) and adds a special CUE_X token to mark negation cue tokens. The output is written to a new TSV file, which serves as input for bert_fine_tuning.ipynb.

- bert_postprocessing.py:
This script transforms the direct output of bert_fine_tuning.ipynb by rejoining subtokens and saves the data to a new TSV file.

- find_discrepancies_tokenlength.py:
This script served as aid for adjusting the script bert_postprocessing.py. It analyses the sentences in 2 TSV files and prints out the sentences that do not have an equal number of tokens.

### 4. Evaluation related scripts:

- evaluation_metrics.py: 
This script calculates token-based and span-based precision, recall, and F1 scores based on a pairwise comparison between the model-generated file and the file containing gold labels. It creates and displays confusion matrices for both token-level and span-level comparisons.

- evaluation_metrics_bert.py: 
This script calculates token-based and span-based precision, recall, and F1 scores based on a pairwise comparison between gold labels and predicted labels by a transformer-based model. The script assumes that gold labels and predictions are located in the same file in separate columns. It creates and displays confusion matrices for both token-level and span-level comparisons. This script can also be utilized to assess the annotations by Group B.

- extract_errors.py:
This script extracts the erronous sentences from the output of the fine-tuned DistilBERT model on the test set, writes the erronous sentences into a new TSV file based on error categories, such as FN. The output of this script can be utilized for error analysis.

-------------
## USAGE

Usage of the scripts: python [script_name].py 


