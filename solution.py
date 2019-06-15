#!/usr/bin/env python3

import sys
import warnings

import numpy as np
import pandas as pd
import nltk

nltk.download('punkt')

str_splitter = lambda str_: nltk.tokenize.word_tokenize(str_)
accuracy = lambda df : (np.sum(df['prediction'] == df['sentiment']) /
                        len(df) * 100)

basic_dict = {'positive': +1, 'neutral': 0, 'negative': -1}


def sentimize(sent_dict, wc_dict, regularize: bool, content: str,
              sentiment_int: int, **kwargs):
    sentence_split = str_splitter(content)

    if regularize:
        sentiment_int /= len(content)
        wc_factor = 1 / len(content)
    else:
        wc_factor = 1

    for word in sentence_split:
        if word in sent_dict:
            wc_dict[word] += wc_factor
            sent_dict[word] += sentiment_int
        else:
            wc_dict[word] = wc_factor
            sent_dict[word] = sentiment_int


def wc_sentimize(sent_dict_wcr, wc_dict, regularize: bool, content: str,
                 sentiment_int: int, **kwargs):
    sentence_split = str_splitter(content)

    if regularize:
        sentiment_int /= len(content)

    for word in sentence_split:
        if word in sent_dict_wcr:
            sent_dict_wcr[word] += sentiment_int / wc_dict[word]
        else:
            sent_dict_wcr[word] = sentiment_int / wc_dict[word]


## Evaluate the model
def eval_sentiment_with_warnings(content: str, prewarned_list: list = [],
                                 **kwargs):
    sentiment_out = 0
    for word in str_splitter(content):
        if word not in prewarned_list:
            try:
                sentiment_out += sent_dict[word]
            except KeyError:
                warnings.warn(Warning("%s not found in sent_dict, assigning "
                                      "`neutral` sentiment".format(word)))
                prewarned_list += [word]
                continue

    return sentiment_out


def eval_sentiment(word_dict, content: str, **kwargs):
    sentiment_out = 0
    for word in str_splitter(content):
        try:
            sentiment_out += word_dict[word]
        except KeyError:
            continue

    return sentiment_out


def evaluate(df, sent_dict):
    predicted_sentiment_vect = [0] * len(df) # Should be more efficient
                                             # than adding each element
                                             # one at a time
    predicted_sentiment_vect_string = [0] * len(df)

    for num, row in df.iterrows():
        predicted_sentiment_vect[num] = eval_sentiment(sent_dict, **row)
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

    # Initialize dicts
    wc_dict = {} # for regularization by word frequency - requires two runs
                 # of sentimize-like functions
    sent_dict = {}
    sent_dict_wcr = {}

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
        sentimize(sent_dict, wc_dict, regularize=regularize, **row)
        wc_sentimize(sent_dict_wcr, wc_dict, regularize=regularize, **row)

    ## evaluate
    evaluate(train_df, sent_dict_wcr) # pass by reference
    print("Accuracy on {} training set: {:0.2f}%".
              format('regularized' if regularize else '', accuracy(train_df)))

    evaluate(test_df, sent_dict_wcr) # pass by reference
    print("Accuracy on {} test set: {:0.2f}%".
              format('regularized' if regularize else '', accuracy(test_df)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
