# Data statement
## A. Header

* Dataset Title: Russian Distorted Toxicity
* Dataset Curator(s): Alla Gorbunova, HRU HSE student
* Dataset Version: 1.0
* Dataset Citation and, if available, DOI: TBA
* Data Statement Author(s): Alla Gorbunova, HRU HSE student
* Data Statement Version: 1.0
* Data Statement Citation: N/A 

## B. Executive Summary

Russian Distorted Toxicity is a dataset of comments from Russian social network VKontakte (vk.com) collected from three public groups. Is was created in order to test how existing toxicity classifiers for Russian language perform on data with distorted words (such as "sh!t" and similar distortions with letters, symbols or digits). Each text is annotated for presence of toxicity and presence of distortion. The dataset size is 3000 samples: 561 toxic comment and 126 distorted comments. The data was collected on 14.02.2022.

## C. Curation Rationale 

The data was collected in order to test the models' ability to process distorted texts, thus the comments needed to be taken from the source rich with both abusive and distorted comments. The empirically identified source selection criteria are listed as follows: the public group within the VKontakte network has to attract viable audiences with opposing views, there have to be many discussions on sensitive topics, it has to have automated moderation enabled, but comments should not be removed manually. All of these are necessary to provoke heated debates and force users to distort abusive words. A sample of 1000 latest comments was collected from each of the manually chosen groups. 

## D. Documentation for Source Datasets

No source datasets were used.
  
## E. Language Varieties

BCP-47 language tag: ru-RU: Russian language as spoken in Russia and Post-Soviet states written in Cyrillic.

## F. Speaker Demographic

Detailed speaker demographics in not available as users are not obliged to provide their true age, gender, race or other characteristics.
We suppose that the majority of the audience is white males, age from 20 to 30, from poor and near-poor to lower and middle economic class. For most of them Russian is the first language, through people who learned Russian as L2 in school might be present. 

* Age: mostly 20-30
* Gender: mostly male
* Race/ethnicity: mostly Russians, also present Ukrainians, Kazakhs and other ethnicities
* Socioeconomic status: from poor and near-poor to lower and middle class
* First language(s): mostly Russian
* Proficiency in the language(s) of the data: mostly native (L1)
* Number of different speakers represented: from 2 to 3 thousands
* Presence of disordered speech: unknown

## G. Annotator Demographic

* Age: 22
* Gender: female
* Race/ethnicity: Russian
* First language: Russian
* Proficiency in the language of the data being annotated: native (L1)
* Number of different annotators represented: 1
* Relevant training: linguistics student

## H. Speech Situation and Text Characteristics

* Time and place of linguistic activity: 
* Date of data collection: 14.02.2022
* Modality: written
* Spontaneous
* Asynchronous interaction (comment section)
* Speakers’ intended audience: other Internet users and commenters
* Genre: social media comment
* Topic: various, mostly politics and sport
* Non-linguistic context: posts with text or images they users comment (not included in the dataset)

## I. Preprocessing and Data Formatting

Newline characters ("\n") were replaced with whitespaces (" ").

## J. Capture Quality

Captured exactly as posted online by the speakers.

## K. Limitations

Criteria for selection sources with high density of toxicity and distortions impose limitations on data diversity: the only groups suitable for the criteria happened to be politically oriented.

## L. Metadata

License: MIT

## M. Disclosures and Ethical Review

N/A

## N. Other

Contains highly abusive language with all sorts of discrimination and unethical insults, reader discretion is advised.

### O. Glossary

N/A

________
Data Statement template taken from https://github.com/TechPolicyLab/Data-Statements/blob/main/README.md
