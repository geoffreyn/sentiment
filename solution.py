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
                self.sent_dict_wcr[word] += sentiment_int / self.wc_dict[word]
            else:
                self.sent_dict_wcr[word] = sentiment_int / self.wc_dict[word]


    ## Evaluate the model
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


    def eval_sentiment(self, word_dict, content: str, **kwargs):
        sentiment_out = 0
        for word in str_splitter(content):
            try:
                sentiment_out += word_dict[word]
            except KeyError:
                continue

        return sentiment_out


    def evaluate(self, df, sent_dict):
        predicted_sentiment_vect = [0] * len(df) # Preallocating should be
                                                 # more efficient
                                                 # than adding each element
                                                 # one at a time
        predicted_sentiment_vect_string = [0] * len(df)

        for num, row in df.iterrows():
            predicted_sentiment_vect[num] = self.eval_sentiment(sent_dict,
                                                               **row)
            predicted_sentiment_vect_string[num] = (
                'positive' if predicted_sentiment_vect[num] > 0 else (
                    'negative' if predicted_sentiment_vect[num] < 0 else 'neutral')
            )

        df['prediction_int'] = predicted_sentiment_vect
        df['prediction'] = predicted_sentiment_vect_string

        return df


def main(argv):
    regularize = argv[0]
    if regularize:
        print("Using regularization")
    else:
        print("Not regularizing")

    twitter_sentiment = Sentiment()

    ## Read in the data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/evaluate.csv")

    # train_df.describe()
    # test_df.describe()

    # Convert possible sentiment output strings to numerical values
    train_df['sentiment_int'] = train_df['sentiment'].map(basic_dict)
    test_df['sentiment_int'] = test_df['sentiment'].map(basic_dict)

    # Fit into sklearn syntax
    X, y = train_df['content'], train_df['sentiment_int']

    kf = KFold(n_splits=n_folds, shuffle=True)  # Define the split - into 2
    # folds
    kf.get_n_splits(X)

    sentiment_list = [Sentiment()] * n_folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print('Working on fold {}'.format(fold+1))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reconstitute Sentiment structure
        subdf_train = pd.concat([X_train, y_train], axis=1)
        subdf_test = pd.concat([X_test, y_test], axis=1)

        # Train on training folds
        for num, row in subdf_train.iterrows():
            sentiment_list[fold].fit(regularize=regularize, **row)
            sentiment_list[fold].wc_fit(regularize=regularize, **row)

        # Evaluate on train then test folds
        subdf_train = sentiment_list[fold].evaluate(subdf_train,
                                                    sentiment_list[fold].sent_dict_wcr)
        print("Fold {} accuracy {} on test set: {:0.2f}%".
                  format(fold+1, 'with regularization' if regularize else '',
                         accuracy(subdf_train)))

        subdf_test = sentiment_list[fold].evaluate(subdf_test,
                                                    sentiment_list[fold].sent_dict_wcr)
        print("Fold {} accuracy {} on test set: {:0.2f}%".
                  format(fold+1, 'with regularization' if regularize else '',
                         accuracy(subdf_test)))

    ## Train the full model
    for num, row in train_df.iterrows():
        twitter_sentiment.fit(regularize=regularize, **row)
        twitter_sentiment.wc_fit(regularize=regularize, **row)

    ## evaluate
    train_df = twitter_sentiment.evaluate(train_df,
                                          twitter_sentiment.sent_dict_wcr)
    print("Accuracy {} on training set: {:0.2f}%".
              format('with regularization' if regularize else '',
                     accuracy(train_df)))

    test_df = twitter_sentiment.evaluate(test_df,
                                          twitter_sentiment.sent_dict_wcr) #
    print("Accuracy {} on test set: {:0.2f}%".
              format('with regularization' if regularize else '',
                     accuracy(test_df)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
