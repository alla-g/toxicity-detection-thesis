# toxicity-detection-thesis
Code and data for my thesis on automatic toxicity detection in social media for russian

Repository structure:
├── hypothesis_testing_data  # data needed to test the hypothesis
│   ├── uncorrected_data_NEW.tsv  # uncorrected test comments
│   ├── corrected_data_NEW.tsv  # test comments with manual correction
|   └── preprocessed_data_NEW.tsv  # test comments preprocessed automatically
│
├── preprocessing_data  # data needed for preprocessing approach
│   ├── bad_wordlist.txt  # list of offensive, obscene and otherwise toxic words
|   └── replacement.json  # rules for replacing cyrillic letters
│
├── toxicity_corpus  # folder for publishing collected distorted toxicity data
│   ├── DATASTATEMENT.md  # data statement fot the corpus
|   └── distorted_toxicity.tsv  # corpus file
│    
├── training_data  # train and val data for training neural networks
│   ├── toxicity_train.tsv
│   ├── toxicity_val.tsv
│   ├── toxicity_data.tsv
│   ├── distortion_train.tsv.
│   ├── distortion_val.tsv
│   ├── distortion_data.tsv
│   ├── combined_train.tsv
|   └── combined_val.tsv
│
├── Testing models  # notebook for first experiment
├── Approach 1 - preprocessing  # notebook for first approach of second experiment
├── Approach 2 - MT BERT  # notebook for first approach of second experiment
├── parsing and preparing data # code for getting and structuring data
├── corpus analysis # code for counting some corpus statistics
└── README.md
