#!/usr/bin/env python3

import sys
import warnings

import numpy as np
import pandas as pd
import nltk

nltk.download('punkt')

# TODO investigate case-sensitive: sparser but capitlization can be indicative of emotional weight
str_splitter = lambda str_: [s.lower() for s in nltk.tokenize.word_tokenize(str_)]
accuracy = lambda df : (np.sum(df['prediction'] == df['sentiment']) /
                        len(df) * 100)

basic_dict = {'positive': +1, 'neutral': 0, 'negative': -1}

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
                    warnings.warn(Warning("%s not found in sent_dict, assigning "
                                          "`neutral` sentiment".format(word)))
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
        predicted_sentiment_vect = [0] * len(df) # Should be more efficient
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

    # TODO crossvalidation

    ## Train the model
    for num, row in train_df.iterrows():
        twitter_sentiment.fit(regularize=regularize, **row)
        twitter_sentiment.wc_fit(regularize=regularize, **row)

    ## evaluate
    twitter_sentiment.evaluate(train_df, twitter_sentiment.sent_dict_wcr)
    print("Accuracy with {} on training set: {:0.2f}%".
              format('regularization' if regularize else '', accuracy(train_df)))

    twitter_sentiment.evaluate(test_df, twitter_sentiment.sent_dict_wcr) #
    print("Accuracy with {} on test set: {:0.2f}%".
              format('regularization' if regularize else '', accuracy(test_df)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
