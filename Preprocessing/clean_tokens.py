from nltk.corpus import stopwords
import re as regex
import string

stop_words = set(stopwords.words('english'))
contraction_with_original_form = {
    "enemies" : "enemy",
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

"""
Cleans up tokens by removing stopwords, punctuation and contractions
"""

def revert_contractions(tokens):
    """
    Reverts contractions into their normal forms
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: A list of sentence tokens
    :rtype: list
    """
    # Iterates through all sentences in the review, finding a contraction word and switching it for its complete form
    for index in range(len(tokens)):
        try:
            for word in tokens[index].split(" "):
                if word in contraction_with_original_form:
                    tokens[index] = regex.sub(word, contraction_with_original_form[word], tokens[index])
        except:
            print(f"error occured at {tokens[index]}")
    return tokens

def remove_all_punctuation(tokens):
    """
    Removes all puncatuation from a set of tokens
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: A list of sentence tokens
    :rtype: list
    """
    # Creates a translator containing string.punctuation characters
    translator = str.maketrans("", "", string.punctuation)

    # Creates exceptions for apostraphe and dash characters as those are delt with after more changes are made
    translator[ord('-')] = " "
    translator[ord("'")] = "'"

    # Applies translator to sentence tokens
    for index in range(len(tokens)):
        tokens[index] = tokens[index].translate(translator)
    return tokens

def remove_stop_words(tokens):
    """
    Removes all stopwords from tokens
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: A list of sentence tokens
    :rtype: list
    """
    # Iterates through all sentences in tokens removing words that occur in the nltk.corpus english stopwords set
    for index in range(len(tokens)):
        try:
            for word in tokens[index].split(" "):
                if word in stop_words:
                    word_remove = r"\b" + regex.escape(word) + r"\b"
                    tokens[index] = regex.sub(word_remove, "", tokens[index])
        except IndexError:
            print(f"{index} is a providing an index error ")
    return tokens

def remove_extra_whitespace(review_text):
    """
    Removes all extra whitespace from tokens
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: A list of sentence tokens
    :rtype: list
    """   
    # Uses a regex expression to remove any excess whitespace from the reviews
    review_text = regex.sub(r'\s+', " ", review_text)

    # Uses a regex expression to remove remaining period noise 
    review_text = regex.sub(r"\s+\.", ".", review_text)
    review_text = regex.sub(r"\.\.\s", ". ", review_text)
    review_text = regex.sub(r'(\w)\.(\w)', r'\1. \2', review_text)
    
    return review_text

def remove_apostraphes(tokens):
    """
    Removes all apostrophe s' from tokens
    :param tokens: A list of sentence tokens
    :type tokens: list
    :return: A list of sentence tokens
    :rtype: list
    """   
    # Uses regex expressions to remove any apostrophe / apostrophe s' including cases for different formats of apostrophe
    # … “” ’ —
    for index in range(len(tokens)):
        tokens[index]=regex.sub("—", '', tokens[index])
        tokens[index]=regex.sub("’", '', tokens[index])
        tokens[index]=regex.sub("“", '', tokens[index])
        tokens[index]=regex.sub("”", '', tokens[index])
        tokens[index]=regex.sub("…", '', tokens[index])
        tokens[index]=regex.sub("'s", '', tokens[index])
        tokens[index]=regex.sub("s'", "", tokens[index])
        tokens[index]=regex.sub("’s", "", tokens[index])
        tokens[index]=regex.sub("s’", "", tokens[index])
        tokens[index]=regex.sub("'", "", tokens[index])
        tokens[index]=regex.sub("'", "", tokens[index])
        tokens[index]=regex.sub("‘s", "", tokens[index])
        tokens[index]=regex.sub("s‘", "", tokens[index])
        tokens[index]=regex.sub('‘', "", tokens[index])
    return tokens
