from nltk.tokenize import *


"""
Removes bothersome errors from scraping, namely excess JS or NA strings
"""


def find_JS(tokens):
    """
    Takes a set of tokens and checks if the Javascript string is found within, signifying cleaning must occur
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: an index for the tokens and a boolean
    :rtype: (int, boolean)
    """
    # Initializes a boolean associated with presence of js related noise
    is_JS = False

    # Initializes a string that shows JS noise is in review
    JS_signifier_string = "you need a javascript"
    index = 0
    for token in tokens:
        if token.startswith(JS_signifier_string):
            is_JS = True
            break
        index += 1
    
    # Returns the index of the noise for use when noise exists 
    return index, is_JS

def find_NA(tokens):
    """
    Takes a set of tokens and checks if the NA string is found within, signifying that it must be removed 
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: a boolean
    :rtype: boolean
    """
    # Initializes a boolean associated with presence of NA related noise
    is_NA = False
    try:
        potential_NA = tokens[-1].split(".")
        if "n/a" in potential_NA:
            is_NA = True
        # Returns a boolean to determine whether there is NA noise or not
        return is_NA
    except IndexError:
        print("Index issue occured with the followng NA, ignoring NA")
        return is_NA


def remove_JS_or_NA(tokens):
    """
    Checks to see if there are JS or NA issues within the review, and removes them
    :param tokens: A list of sentence tokens
    :type tokens: List of strings
    :return: Tokens without NA or JS blurbs
    :rtype: List
    """
    potential_destructoid_noise = '[this review is based on a retail build of the dlc provided by the publisher.]'
    index1, has_JS = find_JS(tokens)
    has_NA = find_NA(tokens)
    index2, has_review = find_reviewed(tokens)

    if tokens[0].startswith("\nreviewed on:") or tokens[0].startswith("\nreviewed on"):
        split = tokens[0].split('\n')
        tokens[0] = split[-1]

    if has_JS:
        del tokens[index1:index1+8]
    if has_NA:
        del tokens[-1]
    if has_review:
        if '\n' in tokens[index2]:
            _, keep = tokens[index2].split('\n')
            tokens[index2] = keep
        else:
            del tokens[index2]
    if potential_destructoid_noise in tokens[-1]:
        del tokens[-1]
    return tokens

def find_reviewed(tokens):  
    """
    Checks to see if there is '[reviewed]' or '[review] is remaining in the text in order to remove it
    :param tokens: A list of sentence tokens
    :type tokens: List of strings
    :return: Both a boolean and an index to use for removing noise
    :rtype: touple of boolean and index
    """  
    reviewed_signifier = "[reviewed]"
    review_signifier = "[review]"
    index = 0
    is_review = False
    for token in tokens:
        if review_signifier in token or reviewed_signifier in token:
            is_review = True
            break
        index += 1
    
    # Returns the index of the noise for use when noise exists 
    return index, is_review
