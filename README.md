# Team 2 Group Project Report

## Team Members
- Νεόφυτος Ιωάννου
- Αντρέας Λαός Ξ.
- Γιώργος Παφίτης

## Project Overview
The project focuses on multiclass sentiment classification of tweets from Twitter. We developed a system that
classifies tweets as Positive, Negative or Neutral. To achieve this, we implement multiple custom features relevant to the task,
perform feature selection using a Random Forest classifier to identify the most useful features, and train a neural classifier, using Optuna to optimize its hyperparameters.

## Feature Types Used
### Lexical Features
- TF-IDF Features: Top 1000 unigrams and bigrams. Important words and phrases may be strongly associated with positive or negative sentiment
- avg_token_length:
- count_elongated_words:

### Syntactic Features
- pos_tag_counts

### Semantic Features
- TextBlob Polarity Score: Direct estimate of tweet's polarity.
- TextBlob Subjectivity Score: Higher subjectivity indicates feelings and opinions, while objective tweets reflect a neutral sentiment.
- VADER Sentiment Scores: Fine-grained polarity scores tailored for social media posts
- emoji_sentiment:
- count_positive_words:
- count_negative_words:

### Structural Features
- Log of Number of Tokens: Longer tweets may correlate with negative emotions.
- Number of Exclamation Marks: Shows emotional intensity.
- Number of Hashtag Marks: Number of exclamation marks (#)
- Number of Questions: Questions reflect uncertainty which could indicate negative emotions.

### Behavioural Features
- Number of Mentions: Counts how many users are mentioned in a tweet, which could carry targeted emotions.
- Contains URL (Boolean): If tweet contains a token starting with http:// or https://
- Number of Happy Emoticons: Directly signal positive emotions
- Number of Sad Emoticons: Directly signal negative emotions
- count_negation:
- negation_ratio:
- count_profanity:

## Model Implemented
## Key Results & Findings