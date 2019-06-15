#!/usr/bin/env python3

r"""
Sentiment analysis on data in data/

Terminal Arguments:
    regularize (bool): Use sentence-length regularization
"""

import os
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import KFold

nltk.download('punkt')

# TODO word stemming and other techniques to reduce dimensionality of input space
# TODO try weighing word sentiment by length (longer words should carry more emotion)
# TODO replace words with synonyms to reduce dimensionality
# TODO correct typos and expand abbreviations
# TODO remove timestamps and URLs and other fluff
# TODO consider time of day's impact on emotion - create categorical time-of-day feature(e.g., morning)
# TODO consider other features and feed these features into a classification model, such as logistic regression
# TODO investigate case-sensitivity: sparser but capitlization can be indicative of emotional weight
str_splitter = lambda str_: [s.lower() for s in nltk.tokenize.word_tokenize(str_)]
accuracy = lambda df : (np.sum(df['prediction'] == df['sentiment']) /
                        len(df) * 100)

basic_dict = {'positive': +1, 'neutral': 0, 'negative': -1}

n_folds = 5


def word_to_dict_key(wordlist: pd.Series):
    print('Discovering words...', end=' ')
    blank_dict = {}
    for row in wordlist:
        for word in str_splitter(row):
            if word not in blank_dict:
                blank_dict[word] = 0

    print('Done!')
    return blank_dict


class Sentiment(object):

    # Initialize dicts
    def __init__(self,
                 wc_dict={},
                 sent_dict={},
                 sent_dict_wcr={},
                 regularize=0):
        self.wc_dict = wc_dict # for regularization by word frequency -
                               # requires two runs of fit-like functions
        self.sent_dict = sent_dict
        self.sent_dict_wcr = sent_dict_wcr
        self.rescale_factor = 1 # only affects results if non-regularized

    def fit(self, regularize: bool, content: str, sentiment_int: int,
            **kwargs):
        sentence_split = str_splitter(content)

        if len(sentence_split) == 0:
            print("bad")
        if regularize:
            sentiment_int /= len(sentence_split)
            wc_factor = 1 / len(sentence_split)
        else:
            wc_factor = 1

        for word in sentence_split:
            if word in self.sent_dict:
                self.wc_dict[word] += wc_factor
                self.sent_dict[word] += sentiment_int
            else:
                self.wc_dict[word] = wc_factor
                self.sent_dict[word] = sentiment_int

        if not regularize:
            self.rescale_factor = max(self.rescale_factor,
                                      abs(sentiment_int))

    def wc_fit(self, regularize: bool, content: str, sentiment_int: int,
               **kwargs):
        sentence_split = str_splitter(content)

        if regularize:
            sentiment_int /= len(sentence_split)

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

        self.rescale_factor = max(self.rescale_factor,
                                  abs(sentiment_int))

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

        return sentiment_out / self.rescale_factor

    def eval_sentiment(self, sent_dict: dict, content: str, **kwargs):
        sentiment_out = 0
        for word in str_splitter(content):
            try:
                sentiment_out += sent_dict[word]
            except KeyError:
                continue

        return sentiment_out / self.rescale_factor

    def evaluate(self, df: pd.DataFrame, sent_dict: dict):
        df_len = len(df)

        predicted_sentiment_vect = [0] * df_len # Preallocating should be
                                                # more efficient than
                                                # growing psv and psvs
        predicted_sentiment_vect_string = [0] * df_len

        for num, row in enumerate(df.itertuples()):
            predicted_sentiment_vect[num] = self.eval_sentiment(sent_dict,
                                                                row.content)

            predicted_sentiment_vect_string[num] = (
                'positive' if predicted_sentiment_vect[num] >= 0.5 else (
                    'negative' if predicted_sentiment_vect[num] <= -0.5
                         else 'neutral')
            )

        df['prediction_int'] = predicted_sentiment_vect
        df['prediction'] = predicted_sentiment_vect_string

    def save(self, name=''):
        os.makedirs('model', exist_ok=True)

        with open('model/sent_dict{name}.txt'.format(name=name), 'w') as f_out:
            json.dump(self.sent_dict, f_out)

        with open('model/wc_dict{name}.txt'.format(name=name), 'w') as f_out:
            json.dump(self.wc_dict, f_out)

        with open('model/sent_dict_wcr{name}.txt'.format(name=name), 'w') as f_out:
            json.dump(self.sent_dict_wcr, f_out)

    def load(self, name=''):
        with open('model/sent_dict{name}.txt'.format(name=name), 'r') as f_in:
            self.sent_dict = json.load(f_in)

        with open('model/wc_dict{name}.txt'.format(name=name), 'r') as f_in:
            self.wc_dict = json.load(f_in)

        with open('model/sent_dict_wcr{name}.txt'.format(name=name), 'r') as f_in:
            self.sent_dict_wcr = json.load(f_in)


def main(argv):
    regularize = (argv[0] == '1')
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

    # Fit into sklearn syntax
    X, y = train_df['content'], train_df['sentiment']

    kf = KFold(n_splits=n_folds, shuffle=True)  # Define the split - into 2
    # folds
    kf.get_n_splits(X)

    fold_train_accuracy = [0] * n_folds
    fold_test_accuracy = [0] * n_folds

    sentiment_list = [Sentiment(wc_dict=blank_dict.copy(),
                                sent_dict=blank_dict.copy(),
                                sent_dict_wcr=blank_dict.copy())] * n_folds
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print('Working on test fold {}: '.format(fold+1), end=' ')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reconstitute Sentiment structure
        subdf_train = pd.concat([X_train, y_train], axis=1)
        subdf_test = pd.concat([X_test, y_test], axis=1)

        # Convert possible sentiment output strings to numerical values
        subdf_train['sentiment_int'] = subdf_train['sentiment'].map(basic_dict)
        subdf_test['sentiment_int'] = subdf_test['sentiment'].map(basic_dict)

        if os.path.isfile('model/wc_dict_fold{n}.txt'.format(n=fold)):
            sentiment_list[fold].load(name='_fold{n}'.format(n=fold))
        else:
            iterator = subdf_train.copy().itertuples()
            # Train on training folds
            print('Fitting...', end=' ')
            for row in iterator:
                sentiment_list[fold].fit(regularize=regularize,
                                         content=row.content,
                                         sentiment_int=row.sentiment_int)

            print('Regularizing...', end=' ')
            iterator = subdf_train.copy().itertuples()
            for row in iterator:
                sentiment_list[fold].wc_fit(regularize=regularize,
                                            content=row.content,
                                            sentiment_int=row.sentiment_int)

            sentiment_list[fold].save(name='_fold{n}'.format(n=fold))

        # Evaluate on train then test folds
        sentiment_list[fold].evaluate(subdf_train,
                                      sentiment_list[fold].sent_dict_wcr)

        print("Evaluating...")
        fold_train_accuracy[fold] = accuracy(subdf_train)
        print("Fold {} accuracy{} on training set: {:0.2f}%".
                  format(fold+1, ' with regularization' if regularize else '',
                         fold_train_accuracy[fold]))

        sentiment_list[fold].evaluate(subdf_test,
                                      sentiment_list[fold].sent_dict_wcr)
        fold_test_accuracy[fold] = accuracy(subdf_test)
        print("Fold {} accuracy{} on test set: {:0.2f}%".
                  format(fold+1, ' with regularization' if regularize else '',
                         fold_test_accuracy[fold]))
        print('Done!')

    print("{} fold cross-validation average accuracy: {:0.2f}%".
              format(n_folds, np.mean(fold_test_accuracy)))

    ## Train the full model
    train_df['sentiment_int'] = train_df['sentiment'].map(basic_dict)
    test_df['sentiment_int'] = test_df['sentiment'].map(basic_dict)

    twitter_sentiment = Sentiment(wc_dict=blank_dict.copy(),
                                  sent_dict=blank_dict.copy(),
                                  sent_dict_wcr=blank_dict.copy())

    if os.path.isfile('model/wc_dict_full.txt'):
        twitter_sentiment.load(name='_full')
    else:
        iterator = train_df.copy().itertuples()
        for row in iterator:
            twitter_sentiment.fit(regularize=regularize,
                                  content=row.content,
                                  sentiment_int=row.sentiment_int)

        iterator = train_df.copy().itertuples()
        for row in iterator:
            twitter_sentiment.wc_fit(regularize=regularize,
                                     content=row.content,
                                     sentiment_int=row.sentiment_int)

        twitter_sentiment.save(name='_full')

    ## Evaluate
    twitter_sentiment.evaluate(train_df, twitter_sentiment.sent_dict_wcr)
    print("Accuracy{} on training set: {:0.2f}%".
              format(' with regularization' if regularize else '',
                     accuracy(train_df)))

    twitter_sentiment.evaluate(test_df, twitter_sentiment.sent_dict_wcr)
    print("Accuracy{} on test set: {:0.2f}%".
              format(' with regularization' if regularize else '',
                     accuracy(test_df)))

    # plt.boxplot([fold_train_accuracy, fold_test_accuracy],
    #             labels=['Training', 'Test'])
    # plt.title('Cross-validation Accuracy')
    # plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
