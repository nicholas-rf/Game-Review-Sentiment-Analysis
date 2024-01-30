import bag_of_words as bow
import tf_idf as td
import pandas as pd
import os
import create_models
import csv
import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np   
import word2vec_tensor_flow as w2v
from tqdm import tqdm
import word2vec_gensim 
from gensim import utils
import gensim.models
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE

# normalized dataset new comes from the text_normalizer
# need to include a special case for enemies to be turned into enemy

"""
vectorize.py contains methods for vectorizing and then running models on the dataset based off of different hyperparameters
"""



def bow_tf_vectorizer(vocab_size, IS_WINDOWS=False): 
    """
    bow_tf_vectorizer gets dataframes for both bow and tf-idf vectorization techniques to use within machine learning models
    :param vocab_size: Vocab size is the total amount of vocab being used to generate vectors for bag of words and tf-idf
    :type vocab_size: int
    :return: two pd.DataFrame objects for use within models
    :rtype: pd.DataFrame, pd.DataFrame
    """
    # Creating potential filenames for datasets based off of vocab size
    if IS_WINDOWS:
        bowfname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/bow/bow_dataset' + str(vocab_size) + '.csv'
        tffname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/tfidf/tf_idf_dataset' + str(vocab_size) + '.csv'
    else:
        bowfname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/bow/bag_of_words_dataset' + str(vocab_size) + '.csv'
        tffname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/tfidf/tf_idf_dataset' + str(vocab_size) + '.csv'

    # Checking filenames to see if they already exist
    if os.path.exists(bowfname):
        print(f'A dataset for the vocab size of {vocab_size} already exists, fetching now')
        return pd.read_csv(bowfname), pd.read_csv(tffname)
    else:
        # Creating the word frequency distribution of vocab_size length
        list_dist = bow.create_word_distribution(vocab_size)

        # Creating a positional dictionary and frequency dictionary from the frequency distribution
        positional_dictionary, frequency_dictionary = bow.get_vocab(list_dist)

        # Applying the bag of words vectorizer to the dataframe and writing it to a csv
        if IS_WINDOWS:
            bowdataframe = pd.read_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/normalized_dataset_new.csv')
        else:
            bowdataframe = pd.read_csv('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/normalized_dataset_new.csv')

        bowdataframe["vector"] = bowdataframe["Review Text"].apply(lambda x: bow.create_vector(positional_dictionary, x, vocab_size))
        proccessed_bow = process_further(bowdataframe, vocab_size)
        proccessed_bow.to_csv(bowfname)


        # Applying the tf-idf vectorizer to the dataframe and writing it to a csv
        if IS_WINDOWS:
            tfdataframe = pd.read_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/normalized_dataset_new.csv')
        else:
            tfdataframe = pd.read_csv('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/normalized_dataset_new.csv')

        idf = td.generate_idf(frequency_dictionary)
        tfdataframe["vector"] = tfdataframe["Review Text"].apply(lambda x: td.create_tf_idf_vector(x, positional_dictionary, idf, vocab_size))

        proccessed_tf = process_further(tfdataframe, vocab_size)
        proccessed_tf.to_csv(tffname)
        return bowdataframe, tfdataframe

def create_lookup(fname, embedding_dim, vocab_size, IS_WINDOWS):
    """
    create_lookup takes a metadata and vector file generated from a tensor-flow word2vec model and returns a hashmap containing words and their embeddings as key:value pairs
    :param fname: A filename to write the lookup table to for potential analysis
    :type fname: str
    :return: A dictionary containing words and embeddings as key:value pairs
    :rtype: dict
    """

    if IS_WINDOWS:
        vname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/tensor_flow_vecs/vectors'+ str(vocab_size) + '_' + str(embedding_dim) +'.tsv'
        mname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/tensor_flow_vecs/metadata' + str(vocab_size) + '.tsv'
    else:
        vname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/Modelling/tensor_flow_vecs/vectors'+ str(vocab_size) + '_' + str(embedding_dim) +'.tsv'
        mname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/Modelling/tensor_flow_vecs/metadata'+ str(vocab_size) + '.tsv'


    with open(mname) as metadata, open(vname) as vector:
        # Make the data readable into a hashmap
        vectors = vector.read()
        words = metadata.read()
        groups = vectors.split('\n')
        words = words.split('\n')
        tsv_reader = csv.reader(groups, delimiter='\t')
        lookup_table = {}
        index = 0
        bar = tqdm(total = vocab_size, desc='creating table')
        # For every row associated with a row in the metadata, create a key : value pair
        for group in tsv_reader:
            lookup_table[words[index]] = group
            index += 1
            bar.update(1)
        bar.close()

        return lookup_table

def fetch_embeddings(text, lookup_table, vec_dim):
    """
    fetch_embeddings fetches embeddings from a lookup_table to vectorize reviews, this method does not standardize the embeddings
    :param text: a normalized review from the dataset
    :type text: str
    :param lookup_table: lookup table of word embeddings
    :type lookup_table: dict
    :param vec_dim: dimension of the vector space within word2vec model
    :type vec_dim: int
    :return: A list of embeddings
    :rtype: list
    """
    try:
        # Tokenize the review into words        
        review_tokens = word_tokenize(text)
        base_array = np.zeros((vec_dim,), dtype=int)
        review_vector = list(base_array)

        # For all words, if the word has embeddings fetch and add them to the review vector
        for word in review_tokens:
            if word in lookup_table:
                for index, number in enumerate(lookup_table[word]): # this is a list / vector of numbers 
                    review_vector[index] += float(number)

        return review_vector
    except:
        print(f"the following review was providing an error: {text}")
        return None

def word2vec_tf_vectorizer(embedding_dimensions, vocab_size, IS_WINDOWS):
    """
    This function reads in an embedding dimension size and a vocab size, checks to see if a model of those parameters has been ran before, and if not
    creates a new word2vec model with those specifications. Then the function returns a dataframe contianing review embeddings flattened across columns
    :param embedding_dimensions: The dimension of the vectors for the word2vec model
    :type embedding_dimensions: int
    :param vocab_size: The total amount of vocab used for training the tensor-flow word2vec model
    :type vocab_size: int
    :return: Dataframe of processed reviews for a machine learning model
    :rtype: pd.Datarame
    """

    # Start by creating a filename for the dataset with specific hyperparameters
    if IS_WINDOWS:
        fname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/tensor_flow_w2v/tf_word2vec_dim_' + str(embedding_dimensions) + 'vocab_' + str(vocab_size) + 'dataset.csv'
    else:
        fname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/tensor_flow_w2v/tf_word2vec_dim_' + str(embedding_dimensions) + 'vocab_' + str(vocab_size) + 'dataset.csv'

    # Check to see if the dataset with the specific hyperparameters already exists
    if os.path.exists(fname):
        print("Model already written for aforementioned parameters, fetching now")
        return pd.read_csv(fname)
    else:

        # From the created model two new .tsv files are created with the specific hyperparameters, we create a hash-table with them for use when creating vectors in the dataset
        if IS_WINDOWS:
            lookup_fname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/frequency_tables/tf_lookup' + str(vocab_size) + '_' + str(embedding_dimensions) + '.json'
        else:
            lookup_fname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/frequency_tables/lookup' + str(vocab_size) + '_' + str(embedding_dimensions) + '.json'
        
        print("Creating Lookup")
        lookup_table = (lookup_fname, embedding_dimensions, vocab_size, IS_WINDOWS)
        print("Lookup table created")
        if IS_WINDOWS:
            unprocessed_reviews = pd.read_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/normalized_dataset_new.csv')
        else:
            unprocessed_reviews = pd.read_csv('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/normalized_dataset_new.csv')
        
        print("Applying embeddings to vectors")
        unprocessed_reviews['vector'] = unprocessed_reviews['Review Text'].apply(lambda x : fetch_embeddings(x, lookup_table, embedding_dimensions))
        print("Embeddings added, flattening embeddings to columns now")

        # Process the reviews further by turning the vectors into embedding dimension length columns
        processed_reviews = process_further(unprocessed_reviews, embedding_dimensions)
        print("Columns flattened")
        # Write the processed dataframe into a csv and return dataframe for machine learning model use
        processed_reviews.to_csv(fname)
        return processed_reviews

def extract_gensim_embeddings(review_text, gensim_model, embedding_dim):
    """
    extract_gensim_embeddings takes in a gensim model and vectorizes a reviews text with the models word embeddings
    :param review_text: The text from a game review
    :type review_text: str
    :param gensim_model: A gensim word2vec model containing word embeddings
    :type gensim_model: A gensim word2vec model
    :param embedding_dim: The embedding dimensions of the gensim model
    :type embedding_dim: int
    :return: A vector containing the result of adding all word embedding vectors from a review
    :rtype: list
    """
    try:
        # Make review into word tokens
        tokens = word_tokenize(review_text)
        base_array = np.zeros((embedding_dim,), dtype=int)
        review_vector = list(base_array)

        # For all tokens apply the gensim models embeddings
        for token in tokens:
            try:
                embeddings = gensim_model.wv[token] 
                for index, number in enumerate(embeddings):
                    review_vector[index] += number
            except KeyError:
                pass
        return review_vector
    except:
        print(f'Issue with the following review: {review_text}')
        return None

def word2vec_gensim_vectorizer(embedding_dimensions, vocab_size, IS_WINDOWS):
    """
    word2vec_gensim_vectorizer applies the gensim word2vec model onto reviews using the cbow architecture instead of n-grams like the tensorflow version
    :param embedding_dimensions: The dimension of the vectors for the word2vec model
    :type embedding_dimensions: int
    :param vocab_size: The total amount of vocab used for training the tensor-flow word2vec model
    :type vocab_size: int
    :return: Dataframe of processed reviews for a machine learning model
    :rtype: pd.Datarame
    """ 
    # Check to see if a dataset of the specific hyperparameters already exists
    if IS_WINDOWS:
        fname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/gensim_w2v/word2vec_gensim_dim_' + str(embedding_dimensions) + 'vocab_' + str(vocab_size) + 'dataset.csv'
    else:
        fname = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/gensim_w2v/word2vec_gensim_dim_' + str(embedding_dimensions) + 'vocab_' + str(vocab_size) + 'dataset.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:

        # If it doesn't exist create a new gensim model with the new hyperparameters
        # model_name = word2vec_gensim.create_word2vec_gensim(embedding_dimensions, vocab_size, IS_WINDOWS)
        if IS_WINDOWS:
            model_name = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/gensim_models/word2vec_gensim_dim_' + str(embedding_dimensions) + 'vocab_' + str(vocab_size) + 'model.bin'
        else:
            model_name = '/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/Modelling/gensim_models/word2vec_gensim_dim_' + str(embedding_dimensions) + 'vocab_' + str(vocab_size) + 'model.bin'
        new_model = gensim.models.Word2Vec.load(model_name)
        
        if IS_WINDOWS:
            unprocessed_reviews = pd.read_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/normalized_dataset_new.csv')
        else:
            unprocessed_reviews = pd.read_csv('/Users/nick/Desktop/Current Classes/PSTAT 131/Project/Data/normalized_dataset_new.csv')

        # Apply the embeddings from the gensim model to the reviews
        print("Model Loaded, applying embeddins to vector now")
        unprocessed_reviews['vector'] = unprocessed_reviews['Review Text'].apply(lambda x : extract_gensim_embeddings(x, new_model, embedding_dimensions))
        print("Embeddings added, flattening embeddings to columns now")

        # Process the reviews further by turning the vectors into embedding dimension length columns
        processed_reviews = process_further(unprocessed_reviews, embedding_dimensions)

        # Write the processed dataframe into a csv and return dataframe for machine learning model use
        processed_reviews.to_csv(fname)
        return processed_reviews

def process_further(dataframe, size):
    """
    Processes the dataframe further as to only include review scores as well as vectors, also flattens out the vectors into columns
    :param dataframe: A dataframe containing vectors of review embeddings
    :type dataframe: pd.Dataframe
    :param size: The embedding size
    :type size: int
    :return: A dataframe with new columns for the number of indices in the review embeddings
    :rtype: pd.Dataframe
    """
    # Create column names for the number of elements in the vector of embeddings
    col_indices = ['col_' + str(x) for x in range(size)]

    # Clean up the dataframe to drop NaN values, stretch the vector into columns, and remove any excess potential issues
    dataframe = dataframe.dropna()
    dataframe[col_indices] = dataframe['vector'].apply(lambda x : pd.Series(x))
    dataframe = dataframe[dataframe['Review Score'].str.contains('issue with')==False]
    dataframe = dataframe[dataframe['Review Score'].str.contains('NAN')==False]
    dataframe['Review Score'] = dataframe['Review Score'].apply(lambda x : float(x))
    return dataframe 

def create_gensim_models(max_dimension, max_vocab_size, IS_WINDOWS):
    """
    create_gensim_models creates models in a batch with specific hyperparameters for easier runtime
    """
    prog_bar = tqdm(total=20, desc="Creating Gensim Datasets")
    for vocab_dim in range(10000, max_vocab_size+10000, 10000):
        for dimension in range(100, max_dimension+100, 100):           
            _ = word2vec_gensim_vectorizer(dimension, vocab_dim, IS_WINDOWS)
            prog_bar.update(1)
    prog_bar.close()

# what do we want to change we want it so training data is same across vocab size, so what we can do is just make it so that the training data is made in this function and then passed into the dimension bit

def create_tensor_flow_models(max_dimension, max_vocab_size, IS_WINDOWS):
    """
    create_tensor_flow_models creates models in a batch with specific hyperparameters for easier runtime
    """
    # w2v.create_model()
    prog_bar = tqdm(total=10, desc="Creating tf Datasets")

    for vocab_dim in range(20000, max_vocab_size+10000, 10000):
        for dimension in range(500, max_dimension+100, 100):
            _ = word2vec_tf_vectorizer(dimension, vocab_dim, IS_WINDOWS)
            prog_bar.update(1)
    prog_bar.close()

def create_bow_tf_datasets(max_vocab_size, IS_WINDOWS):
    """
    create_bow_tf_datasets creates a batch of datasets for specified vocab_size
    """
    for size in range(1000, max_vocab_size+1000, 1000):
        _, _ = bow_tf_vectorizer(size, IS_WINDOWS)
