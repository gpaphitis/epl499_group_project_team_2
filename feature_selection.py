#NOTE: add feature selection functions here

from textblob import TextBlob
from ekphrasis.classes.tokenizer import SocialTokenizer
from math import log

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
    
def polarity(text: str) -> dict:
    """
    Computes the natural logarithm of the number of tokens in the cleaned email.
    Returns: {'log_tokens': log_tokens}
    """
    polarity=TextBlob(text)
    return {'polarity': polarity.sentiment.polarity}
    
def subjectivity(text: str) -> dict:
    """
    Computes the natural logarithm of the number of tokens in the cleaned email.
    Returns: {'log_tokens': log_tokens}
    """
    polarity=TextBlob(text)
    return {'subjectivity': polarity.sentiment.subjectivity}
    
def exclamation_count(text: str) -> dict:
    """
    Computes the natural logarithm of the number of tokens in the cleaned email.
    Returns: {'log_tokens': log_tokens}
    """
    count=0
    for c in text:
       if c =='!':
          count+=1
    return {'exclamation_count': count}
    
def hashtag_count(text: str) -> dict:
    """
    Computes the natural logarithm of the number of tokens in the cleaned email.
    Returns: {'log_tokens': log_tokens}
    """
    count=0
    for c in text:
       if c =='#':
          count+=1
    return {'hashtag_count': count}
    
def mention_count(text: str) -> dict:
    """
    Computes the natural logarithm of the number of tokens in the cleaned email.
    Returns: {'log_tokens': log_tokens}
    """
    count=0
    for c in text:
       if c =='@':
          count+=1
    return {'mention_count': count}
