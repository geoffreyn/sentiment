#!/usr/bin/env python3

r"""
Sentiment analysis on data in data/

Terminal Arguments:
    regularize (bool): Use sentence-length regularization
"""

import sys
import warnings

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import KFold

nltk.download('punkt')

# TODO investigate case-sensitivity: sparser but capitlization can be indicative of emotional weight
# TODO word stemming and other techniques to reduce dimensionality of input space
# TODO try weighing word sentiment by length (longer words should carry more emotion)
str_splitter = lambda str_: [s.lower() for s in nltk.tokenize.word_tokenize(str_)]
accuracy = lambda df : (np.sum(df['prediction'] == df['sentiment']) /
                        len(df) * 100)

basic_dict = {'positive': +1, 'neutral': 0, 'negative': -1}

n_folds = 5


def word_to_dict_key(wordlist: pd.Series):
    blank_dict = {}
    for row in wordlist:
        for word in str_splitter(row):
            if word not in blank_dict:
                blank_dict[word] = 0

    return blank_dict


class Sentiment(object):

    # Initialize dicts
    def __init__(self,
                 wc_dict = {},
                 sent_dict = {},
                 sent_dict_wcr = {}):
        self.wc_dict = wc_dict # for regularization by word frequency -
                               # requires two runs of fit-like functions
        self.sent_dict = sent_dict
        self.sent_dict_wcr = sent_dict_wcr

    def fit(self, regularize: bool, content: str, sentiment_int: int,
            **kwargs):
        sentence_split = str_splitter(content)

        if regularize:
            sentiment_int /= len(content)
            wc_factor = 1 / len(content)
        else:
            wc_factor = 1

        for word in sentence_split:
            if word in self.sent_dict:
                self.wc_dict[word] += wc_factor
                self.sent_dict[word] += sentiment_int
            else:
                self.wc_dict[word] = wc_factor
                self.sent_dict[word] = sentiment_int

    def wc_fit(self, regularize: bool, content: str, sentiment_int: int,
               **kwargs):
        sentence_split = str_splitter(content)

        if regularize:
            sentiment_int /= len(content)

        for word in sentence_split:
            if word in self.sent_dict_wcr:
                try:
                    self.sent_dict_wcr[word] += sentiment_int / self.wc_dict[word]
                except ZeroDivisionError:
                    # word may have negligible weight??
                    pass
            else:
                try:
                    self.sent_dict_wcr[word] = sentiment_int / self.wc_dict[word]
                except ZeroDivisionError:
                    # word may have negligible weight??
                    self.sent_dict_wcr[word] = 0

    def eval_sentiment_with_warnings(self, content: str,
                                     prewarned_list: list = [],
                                     **kwargs):
        sentiment_out = 0
        for word in str_splitter(content):
            if word not in prewarned_list:
                try:
                    sentiment_out += self.sent_dict[word]
                except KeyError:
                    warnings.warn(Warning("%s not found in sent_dict, assigning"
                                          " `neutral` sentiment".format(word)))
                    prewarned_list += [word]
                    continue

        return sentiment_out

    def eval_sentiment(self, sent_dict: dict, content: str, **kwargs):
        sentiment_out = 0
        for word in str_splitter(content):
            try:
                sentiment_out += sent_dict[word]
            except KeyError:
                continue

        return sentiment_out

    def evaluate(self, df: pd.DataFrame, sent_dict: dict):
        df_len = len(df)

        predicted_sentiment_vect = [0] * df_len # Preallocating should be
                                                 # more efficient
                                                 # than adding each element
                                                 # one at a time
        predicted_sentiment_vect_string = [0] * df_len

        for row in df.itertuples():
            num = row.Index
            if num == df_len:
                break

            predicted_sentiment_vect[num] = self.eval_sentiment(sent_dict,
                                                                row.content)

            predicted_sentiment_vect_string[num] = (
                'positive' if predicted_sentiment_vect[num] > 0 else (
                    'negative' if predicted_sentiment_vect[num] < 0 else 'neutral')
            )

        df['prediction_int'] = predicted_sentiment_vect
        df['prediction'] = predicted_sentiment_vect_string


def main(argv):
    regularize = argv[0]
    if regularize:
        print("Using regularization")
    else:
        print("Not regularizing")

    ## Read in the data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/evaluate.csv")

    # train_df.describe()
    # test_df.describe()

    # hopefully save some time by pregenerating word list
    blank_dict = word_to_dict_key(train_df)

    twitter_sentiment = Sentiment(wc_dict=blank_dict, sent_dict=blank_dict,
                                  sent_dict_wcr=blank_dict)

    # Convert possible sentiment output strings to numerical values
    train_df['sentiment_int'] = train_df['sentiment'].map(basic_dict)
    test_df['sentiment_int'] = test_df['sentiment'].map(basic_dict)

    # Fit into sklearn syntax
    X, y = train_df['content'], train_df['sentiment_int']

    kf = KFold(n_splits=n_folds, shuffle=True)  # Define the split - into 2
    # folds
    kf.get_n_splits(X)

    fold_accuracy = [0] * n_folds

    sentiment_list = [Sentiment(wc_dict=blank_dict, sent_dict=blank_dict,
                                  sent_dict_wcr=blank_dict)] * n_folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print('Working on fold {}'.format(fold+1))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reconstitute Sentiment structure
        subdf_train = pd.concat([X_train, y_train], axis=1)
        subdf_test = pd.concat([X_test, y_test], axis=1)

        iterator = subdf_train.copy().itertuples()
        # Train on training folds
        for row in iterator:
            sentiment_list[fold].fit(regularize=regularize,
                                     content=row.content,
                                     sentiment_int=row.sentiment_int)

        iterator = subdf_train.copy().itertuples()
        for row in iterator:
            sentiment_list[fold].wc_fit(regularize=regularize,
                                        content=row.content,
                                        sentiment_int=row.sentiment_int)

        # Evaluate on train then test folds
        sentiment_list[fold].evaluate(subdf_train,
                                      sentiment_list[fold].sent_dict_wcr)

        print("Fold {} accuracy{} on training set: {:0.2f}%".
                  format(fold+1, ' with regularization' if regularize else '',
                         accuracy(subdf_train)))

        sentiment_list[fold].evaluate(subdf_test,
                                      sentiment_list[fold].sent_dict_wcr)
        fold_accuracy[fold] = accuracy(subdf_test)
        print("Fold {} accuracy{} on test set: {:0.2f}%".
                  format(fold+1, ' with regularization' if regularize else '',
                         fold_accuracy[fold]))

    print("{} fold cross-validation accuracy: {:0.2f}".format(n_folds,
                                                              np.mean(fold_accuracy)))

    iterator = train_df.copy().itertuples()
    ## Train the full model
    for row in iterator:
        twitter_sentiment.fit(regularize=regularize,
                              content=row.content,
                              sentiment_int=row.sentiment_int)

    iterator = train_df.copy().itertuples()
    for row in iterator:
        twitter_sentiment.wc_fit(regularize=regularize,
                                 content=row.content,
                                 sentiment_int=row.sentiment_int)

    ## Evaluate
    twitter_sentiment.evaluate(train_df, twitter_sentiment.sent_dict_wcr)
    print("Accuracy{} on training set: {:0.2f}%".
              format(' with regularization' if regularize else '',
                     accuracy(train_df)))

    twitter_sentiment.evaluate(test_df, twitter_sentiment.sent_dict_wcr)
    print("Accuracy{} on test set: {:0.2f}%".
              format(' with regularization' if regularize else '',
                     accuracy(test_df)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
