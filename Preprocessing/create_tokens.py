from nltk.tokenize import *
import re as regex

"""
Creates sentence tokens using NTLK tokenize using text
"""

def generate_sentence_tokens(game_review):
    """
    Generates tokens for a game_review using nltk.tokenize.sent_tokenize()
    :param game_review: A gamespot game review's text
    :type game_review: str
    :return: A list of sentences as tokens 
    :rtype: list : [[sentences]]
    """
    return sent_tokenize(game_review)

def apply_regex_1(game_review):
    """
    Removes numbers, extra newlines and makes text lowercase
    :param game_review: A game review from the dataset
    :type game_review: str
    :return: A game review with modifications
    :rtype: str
    """
    # Makes all text lower-case to begin with
    game_review = game_review.lower()

    # replaces all digits and digits followed by periods with nothiing and newlines and newlines following periods with ". "
    game_review = regex.sub(r'\d+.', '', game_review)
    game_review = regex.sub(r'\d+', '', game_review)
    game_review = regex.sub(r'\.\n', '. ', game_review)
    game_review = regex.sub(r'\n', '. ', game_review)
    return game_review



