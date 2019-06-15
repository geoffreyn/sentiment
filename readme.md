# Sentiment analysis of prerecorded tweets

## Usage:

Regularization on:

1.  python3 solution.py 1

Regularization off:

1.  python3 solution.py 0

<hr>

## Sample handling of existing model

```
import pandas as pd

regularize = 1
from solution import Sentiment, basic_dict, accuracy

twitter_sentiment = Sentiment()
twitter_sentiment.load(name='_full')
test_df = pd.read_csv("data/evaluate.csv")
test_df['sentiment_int'] = test_df['sentiment'].map(basic_dict)

twitter_sentiment.evaluate(test_df, twitter_sentiment.sent_dict_wcr)
print("Accuracy{} on test set: {:0.2f}%".
              format(' with regularization' if regularize else '',
                     accuracy(test_df)))
 ```
