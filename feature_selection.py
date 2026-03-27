#NOTE: add feature selection functions here

from textblob import TextBlob
from ekphrasis.classes.tokenizer import SocialTokenizer
from math import log
import re



tokenizer = SocialTokenizer(lowercase=False).tokenize

def log_number_of_tokens(text: str) -> dict:
    """
    Computes the natural logarithm of the number of tokens in the cleaned email.
    Returns: {'log_tokens': log_tokens}
    """
    tokens=tokenizer(text)
    if len(tokens) == 0:
        return {'log_tokens': 0}
    return {'log_tokens': log(len(tokens))}
    
def polarity_score(text: str) -> dict:
    """
    Returns TextBlob polarity score.
    Returns: {'polarity': polarity}
    """
    polarity=TextBlob(text)
    return {'polarity': polarity.sentiment.polarity}
    
def subjectivity_score(text: str) -> dict:
    """
    Returns TextBlob subjectivity score.
    Returns: {'subjectivity': float}
    """
    return {'subjectivity': TextBlob(text).sentiment.subjectivity}

    
def exclamation_count(text: str) -> dict:
    """
    Counts the number of exclamation marks in the text.
    Returns: {'exclamation_count': count}
    """
    count=0
    for c in text:
       if c =='!':
          count+=1
    return {'exclamation_count': count}
    
def hashtag_count(text: str) -> dict:
    """
    Counts the number of hashtags in the text.
    Returns: {'hashtag_count': count}
    """
    count=0
    for c in text:
       if c =='#':
          count+=1
    return {'hashtag_count': count}

# Question mark count
def count_question(text: str) -> dict:
    """
    Counts question marks in a tweet.
    Returns: {'question_count': count}
    """
    return {'question_count': text.count('?')}
   
def mention_count(text: str) -> dict:
    """
    Counts the number of mentions in the text.
    Returns: {'mention_count': count}
    """
    count=0
    for c in text:
       if c =='@':
          count+=1
    return {'mention_count': count}

# URL presence (binary)
def has_url(text: str) -> dict:
    """
    Returns 1 if the tweet contains a URL, else 0.
    Returns: {'has_url': 0 or 1}
    """
    return {'has_url': 1 if re.search(r'https?://\S+', text) else 0}

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

# VADER sentiment scores
def vader_scores(text: str) -> dict:
    """
    Returns the 4 VADER sentiment scores for a certain tweet.
    Returns: {'vader_compound': float, 'vader_pos': float,
              'vader_neg': float, 'vader_neu': float}
    """
    scores = _vader.polarity_scores(text)
    return {
        'vader_compound': scores['compound'],
        'vader_pos':      scores['pos'],
        'vader_neg':      scores['neg'],
        'vader_neu':      scores['neu'],
    }
    
# Happy emoticons :) :D etc.
def count_happy_emoticons(text: str) -> dict:
    """
    Counts happy emoticons in a tweet.
    Returns: {'happy_emoticon_count': count}
    """
    pattern = re.compile(r':\)|=\)|:-\)|:-D|:D|=D|XD|xD')
    return {'happy_emoticon_count': len(pattern.findall(text))}


# Sad emoticons :( etc.
def count_sad_emoticons(text: str) -> dict:
    """
    Counts sad emoticons in a tweet.
    Returns: {'sad_emoticon_count': count}
    """
    pattern = re.compile(r':\(|=\(|:-\(|D:')
    return {'sad_emoticon_count': len(pattern.findall(text))}



# Negation word count
_NEGATION = {
    'not','no','never','nobody','nothing','nowhere','neither','nor',
    'barely','hardly','scarcely',"n't","don't","won't","can't",
    "isn't","doesn't","didn't","wouldn't","shouldn't","couldn't"
}
def count_negation(text: str) -> dict:
    """
    Counts negation words in a tweet.
    Returns: {'negation_count': count}
    """
    tokens = text.lower().split()
    return {'negation_count': sum(1 for t in tokens if t in _NEGATION)}

# Negation ratio — proportion of negation words relative to total tokens
def negation_ratio(text: str) -> dict:
    """
    Computes ratio of negation words over total tokens in a tweet.
    Returns: {'negation_ratio': float}
    """
    tokens = text.lower().split()
    n = len(tokens) or 1
    return {'negation_ratio': sum(1 for t in tokens if t in _NEGATION) / n}



# Emoji sentiment (positive, negative, neutral emoji counts)
_POSITIVE_EMOJI = set([
    '😀','😃','😄','😁','😆','😊','🙂','😍','🥰','😘','🤩','😎',
    '🥳','😇','🤗','👍','👏','🎉','🎊','❤️','💕','💯','🔥','✨',
    '💪','🙌','😂','🤣','😜','😉','😏','🤑','💚','💙','💜','🧡',
    '💛','❤','💖','💗','💓','💞','🌟','⭐','🌈','🎶','🎵'
])
_NEGATIVE_EMOJI = set([
    '😢','😭','😡','🤬','😠','😤','😞','😔','😟','😕','🙁','☹️',
    '😩','😫','🤮','🤢','💔','👎','😖','😣','😓','😰','😨','😱',
    '🥺','😿','💀','☠️'
])

# For simplicity, we just count positive and negative emojis. Neutral emojis are less common and harder to define.
def emoji_sentiment(text: str) -> dict:
    """
    Counts positive and negative emojis in a tweet.
    Returns: {'positive_emoji_count': int, 'negative_emoji_count': int}
    """
    pos = sum(1 for c in text if c in _POSITIVE_EMOJI)
    neg = sum(1 for c in text if c in _NEGATIVE_EMOJI)
    return {'positive_emoji_count': pos, 'negative_emoji_count': neg}

# Elongated words like "wowwwww" or "noooo" signal strong emotion
def count_elongated_words(text: str) -> dict:
    """
    Detects words with characters repeated 3+ times (e.g. 'sooo', 'noooo').
    Returns: {'elongated_word_count': count}
    """
    return {'elongated_word_count': len(re.findall(r'\b\w*(\w)\1{2,}\w*\b', text))}


# Longer words on average may indicate more formal or aggressive language
def avg_word_length(text: str) -> dict:
    """
    Computes mean character length across all words in a tweet.
    Returns: {'avg_word_length': float}
    """
    words = text.split()
    avg = sum(len(w) for w in words) / len(words) if words else 0
    return {'avg_word_length': avg}

import spacy
nlp = spacy.load("en_core_web_sm")

# Part-of-speech distribution using spaCy
def pos_tag_counts(text: str) -> dict:
    """
    Extracts POS tag counts and their normalized ratios per tweet.
    Returns counts for nouns, verbs, adjectives, adverbs, pronouns,
    interjections, auxiliary verbs, determiners, particles, and negations,
    plus a _ratio version of each normalized by total token count.
    Returns: {'num_nouns': int, ..., 'num_nouns_ratio': float, ...}
    """
    doc = nlp(text)
    n = len(doc) or 1

    counts = {
        'num_nouns': 0, 'num_verbs': 0, 'num_adjectives': 0,
        'num_adverbs': 0, 'num_pronouns': 0, 'num_interjections': 0,
        'num_aux_verbs': 0, 'num_determiners': 0, 'num_particles': 0,
        'num_negations': 0
    }

    tag_map = {
        'NOUN': 'num_nouns', 'VERB': 'num_verbs', 'ADJ': 'num_adjectives',
        'ADV': 'num_adverbs', 'PRON': 'num_pronouns', 'INTJ': 'num_interjections',
        'AUX': 'num_aux_verbs', 'DET': 'num_determiners', 'PART': 'num_particles'
    }

    negation_words = {'not', "n't", 'no', 'never', 'cannot', 'cant',
                      'doesnt', 'dont', 'didnt', 'wouldnt', 'shouldnt'}

    for token in doc:
        if token.pos_ in tag_map:
            counts[tag_map[token.pos_]] += 1
        if token.lower_ in negation_words:
            counts['num_negations'] += 1

    for key in list(counts.keys()):
        counts[key + '_ratio'] = counts[key] / n

    return counts

from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon', quiet=True)

POSITIVE_WORDS = set(opinion_lexicon.positive())
NEGATIVE_WORDS = set(opinion_lexicon.negative())


# Positive word count using NLTK Opinion Lexicon
def count_positive_words(text: str) -> dict:
    """
    Counts positive opinion words in a tweet using the NLTK Opinion Lexicon.
    Returns: {'positive_word_count': count}
    """
    tokens = [w.lower() for w in tokenizer(text)]
    return {'positive_word_count': sum(1 for t in tokens if t in POSITIVE_WORDS)}


# Negative word count using NLTK Opinion Lexicon
def count_negative_words(text: str) -> dict:
    """
    Counts negative opinion words in a tweet using the NLTK Opinion Lexicon.
    Returns: {'negative_word_count': count}
    """
    tokens = [w.lower() for w in tokenizer(text)]
    return {'negative_word_count': sum(1 for t in tokens if t in NEGATIVE_WORDS)}

from better_profanity import profanity
profanity.load_censor_words()

def count_profanity(text: str) -> dict:
    """
    Counts profanity words in a tweet using the better-profanity lexicon.
    Returns: {'profanity_count': count}
    """
    tokens = text.lower().split()
    return {'profanity_count': sum(1 for t in tokens if profanity.contains_profanity(t))}