# Automatic Toxic Comment Detection in Social Media for Russian
NRU HSE, Fundamental and computational linguistics, Moscow 2022  
  
All collected and utilized data is provided in the corresponding folders and files (see detailed structure below).  
Code for replicating one of the models is also provided in [this](https://github.com/alla-g/Russan-Hate-speech-Recognition) fork.  

## Links to the trained single- and multitask BERT models:

**vk data, 1 task**  
https://drive.google.com/uc?id=1barEeEUgEUXHHkYN-l-s2i8AZYEyTtYp  
**vk data, 2 tasks**  
https://drive.google.com/uc?id=1--iwGBQHBUXXktC9kqHllnmPzwN9wRYz  
**several source data, 1 task**  
https://drive.google.com/uc?id=1gJ1IPzpaVG81EzyyF7l9m67L_IbH4uZQ  
**several source data, 2 tasks****  
https://drive.google.com/uc?id=1Xu-4-3kYv8HCU2j7zgx84FZm778lzIKk  

In case links become unavailable, feel free to contact me on alla.s.gorbunova@gmail.com

## Repository structure:  
```
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
│   ├── ...     
│  
├── Testing models.ipynb  # notebook for first experiment  
├── Approach 1 - preprocessing.ipynb  # notebook for first approach of second experiment  
├── Approach 2 - MT BERT.ipynb  # notebook for first approach of second experiment  
├── parsing and preparing data.ipynb # code for getting and structuring data  
├── corpus analysis.ipynb # code for counting some corpus statistics  
└── README.md
```

