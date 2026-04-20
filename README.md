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
- Average Token Length: Use of longer words may reflect stronger emotions.
- Number of Elongated Words: Elongated words (e.g. soooo) could indicate emphasis.

### Syntactic Features
- Number POS Tags: Count of each POS tags which can distinguish sentences carrying emotion. 

### Semantic Features
- TextBlob Polarity Score: Direct estimate of tweet's polarity.
- TextBlob Subjectivity Score: Higher subjectivity indicates feelings and opinions, while objective tweets reflect a neutral sentiment.
- VADER Sentiment Scores: Fine-grained polarity scores tailored for social media posts
- Count Of Positive and Negative Emoticons: Captures sentiment carried from emoticons
- Number of Positive Words: Direct correlation with positive sentiment
- Number of Negative Words: Direct correlation with negative sentiment

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
- Count of Negation Words: Negation words flip the polarity of sentences.
- Ratio of Negation Words: High negation ratio leads to ambiguity in polarity detection.
- Count of Profanity Words: Profanity often correlates with strong negative emotions

## Model Implemented
## Key Results & Findings