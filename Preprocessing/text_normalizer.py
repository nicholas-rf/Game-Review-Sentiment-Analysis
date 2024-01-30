print("Compiling text_normalizer")
import os
print(os.getcwd())
import sys
sys.path.append('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Preprocessing')
import remove_bother # removes JS and NA tokens if applicable
import create_tokens # removes some stuff with regex prior to and after tokenization
import clean_tokens # cleans up tokens by removing whitespace, reverting contractions, removing punctuation, and stopwords
import pandas as pd

"""
Takes a corpus of text and pre-processes it for vectorization
"""
# … “” ’ —

# ()   ()
# (^ _ ^)
#  (    )*
#   () ()

# [This review is based on a retail build of the game provided by the publisher.] and 

def normalize_to_csv(review_text):

    """
    Normalizes the text in a review for vectorization
    :param review_text: A review
    :type review_text: str
    :return final_text: Returns a cleaned up version of the review
    :type final_text: str
    """
    if type(review_text) == float:
        return None
    # Takes text and removes numbers, newline characters and makes text lowercase
    review_text = create_tokens.apply_regex_1(review_text)

    # Separates the text into tokens
    review_as_tokens = create_tokens.generate_sentence_tokens(review_text)

    # Removes unnecesary noise leftover from html scraping
    tokens_without_bother = remove_bother.remove_JS_or_NA(review_as_tokens)

    # Expands contradictions into bigger form, ie don't -> do not
    tokens_without_contractions = clean_tokens.revert_contractions(tokens_without_bother)

    # Removes punctuation from the tokens
    tokens_without_punctuation = clean_tokens.remove_all_punctuation(tokens_without_contractions)

    # Removes common stop words to make dataset higher quality for vectorization
    tokens_without_stop_words = clean_tokens.remove_stop_words(tokens_without_punctuation)

    # Removes extra apostrohes and apostrophe s' for different types of '
    tokens_without_apostrophe_s = clean_tokens.remove_apostraphes(tokens_without_stop_words)

    # Joins the review back together by sentence
    full_text = ".".join(tokens_without_apostrophe_s)

    # Removes extra whitespace
    final_text = clean_tokens.remove_extra_whitespace(full_text)

    # Returns final text
    return final_text


def write_corpus(IS_WINDOWS=False):
    """
    Write corpus is only for first time use as it writes the complete review corpus to a txt file
    """
    if IS_WINDOWS:
        dataframe = pd.read_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/normalized_dataset_new.csv')
    else:
        dataframe = pd.read_csv('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/normalized_dataset_new.csv')

    list_of_all_reviews = dataframe['Review Text'].tolist()
    final_text = ""
    
    for review in list_of_all_reviews:
        if type(review) != float:
            final_text += review




    if IS_WINDOWS:
        with open('C:/Users/nicho/OneDrive/Desktop/Project/Data/dataset_corpus.txt', 'w',encoding='utf-8') as f:
            f.write(final_text)
    else:
        with open("/Users/nick/Desktop/dataset_corpus.txt", 'w', encoding='utf-8') as f:
            f.write(final_text)

def write_to_csv(normalize_func, IS_WINDOWS):
    """
    Applies the pre-processor to a dataset and then writes the dataset to a csv
    :param normalize_func:
    :type normalize_func:
    :return: None
    :rtype: None
    """
    if IS_WINDOWS:
        dataframe = pd.read_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/dataset.csv')
        dataframe["Review Text"] = dataframe["Review Text"].apply(normalize_func)
        dataframe = dataframe.dropna()
        dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/normalized_dataset_new.csv')
    else:
        dataframe = pd.read_csv("/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/dataset.csv")
        dataframe["Review Text"] = dataframe["Review Text"].apply(normalize_func)
        dataframe.to_csv('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/normalized_dataset_new.csv')




def main(IS_WINDOWS):
    write_to_csv(normalize_to_csv, IS_WINDOWS)
