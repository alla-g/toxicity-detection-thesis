import nltk
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.dataset import read_dataset, balance_dataset
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor
from src.sentiment_analyzer import SentimentAnalyzer

nltk.download('punkt')
nltk.download('stopwords')

if __name__ == "__main__":

    # read the dataset
    df = read_dataset()

    # make the dataset balanced (stratify)
    df = balance_dataset(df)

    # preprocessing stage 1: Regular expressions & Filters
    print("preprocessing stage 1..")
    preprocessor = Preprocessor()
    df['text'] = df['text'].apply(preprocessor.preprocess)

    # extract the important features
    print("extraction of features..")
    feature_extractor = FeatureExtractor()
    df['offensive_words_count'] = df['text'].apply(feature_extractor.find_offensive_words)
    df['caps_words_count'] = df['text'].apply(feature_extractor.find_capsed_words)

    # if there is no sentiment data, label it automatically
    if not df['sentiment'].any():
        print("initiating sentiment analyzer..")
        df = SentimentAnalyzer().sentiment_label_dataframe(df)

    # acquire TD-IDF features
    print("acquring TF-IDF features..")
    tfidf, vocab, idf_dict = preprocessor.get_TFIDF_features(df, filter_stopwords=False)

    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    df['text'] = df['text'].apply(preprocessor.tokenize)
    df['text'] = df['text'].apply(preprocessor.lemmatize)

    # extracting the textual features
    print("extracting features..")
    feats = feature_extractor.get_feature_array(df)
    M = np.concatenate([tfidf, feats], axis=1)

    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    # MODEL #

    X = DataFrame(M)
    y = df['hate_speech'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    print("training model..")
    model = LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge', multi_class='ovr',
                      max_iter=3000, dual=False)
    model.fit(x_train, y_train)

    ### ADD MY TEST CORPORA ###
    uncorrected = pd.read_csv('/content/toxicity-detection-thesis/uncorrected_data.tsv', sep='\t')
    corrected = pd.read_csv('/content/toxicity-detection-thesis/corrected_data.tsv', sep='\t')
    
    uncorrected = pd.read_csv('/content/toxicity-detection-thesis/data/uncorrected_data.tsv', sep='\t')
    corrected = pd.read_csv('/content/toxicity-detection-thesis/data/corrected_data.tsv', sep='\t')

    # extract the important features
    print("extraction of features..")
    feature_extractor = FeatureExtractor()
    uncorrected['offensive_words_count'] = uncorrected['text'].apply(feature_extractor.find_offensive_words)
    uncorrected['caps_words_count'] = uncorrected['text'].apply(feature_extractor.find_capsed_words)

    corrected['offensive_words_count'] = corrected['text'].apply(feature_extractor.find_offensive_words)
    corrected['caps_words_count'] = corrected['text'].apply(feature_extractor.find_capsed_words)

    # if there is no sentiment data, label it automatically
    print("initiating sentiment analyzer..")
    uncorrected = SentimentAnalyzer().sentiment_label_dataframe(uncorrected)
    corrected = SentimentAnalyzer().sentiment_label_dataframe(corrected)

    # acquire TD-IDF features for uncor
    print("acquring TF-IDF features for uncor..")
    tfidf, vocab, idf_dict = preprocessor.get_TFIDF_features(uncorrected, filter_stopwords=False)
    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    uncorrected['text'] = uncorrected['text'].apply(preprocessor.tokenize)
    uncorrected['text'] = uncorrected['text'].apply(preprocessor.lemmatize)
    # extracting the textual features
    print("extracting features..")
    feats = feature_extractor.get_feature_array(uncorrected)
    M = np.concatenate([tfidf, feats], axis=1)
    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    X_uncor = DataFrame(M)

    # acquire TD-IDF features for cor
    print("acquring TF-IDF features for uncor..")
    tfidf, vocab, idf_dict = preprocessor.get_TFIDF_features(corrected, filter_stopwords=False)
    # preprocessing stage 2: Tokenize & Lemmatize
    print("preprocessing stage 2..")
    corrected['text'] = corrected['text'].apply(preprocessor.tokenize)
    corrected['text'] = corrected['text'].apply(preprocessor.lemmatize)
    # extracting the textual features
    print("extracting features..")
    feats = feature_extractor.get_feature_array(corrected)
    M = np.concatenate([tfidf, feats], axis=1)
    variables = [''] * len(vocab)
    for k, v in vocab.items():
        variables[v] = k

    feature_names = variables + feature_extractor.get_feature_names()

    X_cor = DataFrame(M)

    # конец препроца #

    y_uncor = uncorrected['toxicity'].astype(int)
    y_cor = corrected['toxicity'].astype(int)

    ### PREDICT ON MY TEST ###
    print('predicting on uncorrected')
    y_preds = model.predict(X_uncor)
    report = classification_report(y_uncor, y_preds)
    print(report)
    plot_confusion_matrix(model, x_test, y_uncor)

    print('predicting on corrected')
    y_preds = model.predict(X_cor)
    report = classification_report(y_cor, y_preds)
    print(report)
    plot_confusion_matrix(model, x_test, y_cor)
    
    
    ## preprocessing ##

    import re
import string
from typing import List, Any

import nltk
import pymorphy2
from nltk import ngrams
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def preprocess(self, text: str) -> str:
        text_string = text

        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        hashtag_regex = '#[\w\-]+'
        retweet_regex = 'RT'
        e_regex = 'ё'
        e_big_regex = 'Ё'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)  # urls do not give information
        parsed_text = re.sub(mention_regex, '@', parsed_text)
        parsed_text = re.sub(hashtag_regex, '#', parsed_text)
        parsed_text = re.sub(retweet_regex, 'retweethere', parsed_text)
        parsed_text = re.sub(e_regex, 'е', parsed_text)
        parsed_text = re.sub(e_big_regex, 'Е', parsed_text)

        return parsed_text

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def tokenize(self, sentence: str) -> List[str]:
        return self.lemmatize(nltk.word_tokenize(sentence))

    def get_TFIDF_features(self, df: DataFrame, filter_stopwords: bool = False,
                           vectorizer=None) -> Any:
        stopwords = nltk.corpus.stopwords.words("russian")
        
        if vectorizer == None:
            vectorizer = TfidfVectorizer(
                tokenizer=self.tokenize,
                preprocessor=self.preprocess,
                ngram_range=(1, 3),
                stop_words=stopwords if filter_stopwords else None,  # somehow the score is better if we keep the stopwords
                use_idf=True,
                smooth_idf=False,
                norm=None,
                decode_error='replace',
                max_features=10000,
                min_df=5,
                max_df=0.501
            )

        tfidf = vectorizer.fit_transform(df['text']).toarray()
        vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        idf_dict = {i: idf_vals[i] for i in vocab.values()}

        return tfidf, vocab, idf_dict, vectorizer
