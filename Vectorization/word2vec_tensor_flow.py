import re
import string
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
import keras.api._v2.keras as keras
from tqdm import tqdm

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
"""
Module that trains a word2vec tensor flow model for vectorization of dataset
"""

# cite
# https://www.tensorflow.org/text/tutorials/word2vec#setup
# https://www.tensorflow.org/text/guide/word_embeddings#using_the_embedding_layer
# https://stackoverflow.com/questions/39843584/gensim-doc2vec-vs-tensorflow-doc2vec

def file_updater(IS_WINDOWS=False):
    """
    Converts previous corpus.txt file into one readable by tf.data.TextLineDataset, which requires a file of sentences separated by newline characters
    """
    my_text = ""

    # Check if the program is running on windows to determine file storage location
    if IS_WINDOWS:

        # Write the corpus to a string
        with open('C:/Users/nicho/OneDrive/Desktop/Project/Data/dataset_corpus.txt', 'r', encoding='mac-roman') as f:
            for line in f:
                my_text += line

        # Write the string onto a file separated by newline characters
        with open('C:/Users/nicho/OneDrive/Desktop/Project/Data/corpus_newline2.txt', 'w', encoding='utf-8') as f:
            my_text_list = my_text.split(".")
            line = tqdm(desc="Writing Reviews into newline txt", total=len(my_text_list))
            for sentence in my_text_list:
                f.write(sentence + '\n')
                line.update(1)
            line.close()

    else:

        # Write the corpus to a string
        with open("/Users/nick/Desktop/dataset_corpus.txt", 'r', encoding='utf-8') as f:
            for line in f:
                my_text += line

        # Write the string onto a file separated by newline characters
        with open("/Users/nick/Desktop/corpus_newline.txt", 'w', encoding='utf-8') as f:
            for sentence in my_text.split("."):
                f.write(sentence + '\n')

class Word2Vec(keras.Model):
  """
  This is a word2vec class that inherits keras.Model class methods, it is used as a class to train with the data from the corpus
  Side note: under layers.Embedding, the input_length is set to be num_ns + 1, with num_ns being the number of negative samples for the skip grams
  """
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=4+1)

  def call(self, pair):
    target, context = pair
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    word_emb = self.target_embedding(target)
    context_emb = self.context_embedding(context)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    return dots
  
def tf_vectorizer(vocab_size=20000):
    """
    Creates a TensorFlow vectorizer layer to help with the creation of sequences for model training
    :param vocab_size: The vocabulary size of our sequences for the vectorization layer
    :type vocab_size: int
    :return: A TensorFlow Vectorizer layer
    :rtype: tf.TextVectorization
    """

    # A custom stnadardization function that normalizes the text for tensorflow word2vec standards
    def custom_standardization(input_data):
        """
        Standard standardization function from TensorFlow word2vec documentation
        """
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                '[%s]' % re.escape(string.punctuation), '')

    # Establish the vectorization layer with the custom standardization function along with other parameters
    sequence_length = 10
    vectorize_layer = keras.layers.TextVectorization(
        standardize = custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Return the vectorization layer
    return vectorize_layer

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  """
  Generates training data for the word2vec model by creating targets, contexts, and labels from training data
  """
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

def create_model(max_embedding_dim=300, max_vocab_size=20000, filepath="/Users/nick/Desktop/corpus_newline.txt", IS_WINDOWS=False):
    """
    Creates a word2vec model by taking a dataset of text, generating a vector layer with it, creating a set of vectors, and then turning the corpus into sequences.
    Sequences are then read into a function which generates training data, then training data is then placed into a word2vec model and several epochs are ran
    to improve accuracy of word embedding
    :param filepath: A filepath to a corpus of text
    :type filepath: str
    :return: A lookup table of words and their embeddings
    :rtype: dict {word:[embeddings]}
    """
    def create_text_dataset(vocab_size):
        """
        create_text_dataset creates training data to be used for the generation of word2vec embeddings of a specific vocab_size
        :param vocab_size: The number of words to be included in a models vocabulary
        :type vocab_size: int
        :return: A tensorflow dataset of training data for use within the model
        :rtype: Tensorflow.data.Dataset object
        """

        # A filepath is specified pointing to the corpus written in a format readable by tensor-flow       
        filepath = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/corpus_newline.txt'

        # A tf.data.TextLineDataset is generated from the dataset specified at filepath
        text_dataset = tf.data.TextLineDataset(filepath).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        # A vectorize layer is created with a vocab_size in mind to be used to generate training data
        vectorize_layer = tf_vectorizer(vocab_size)

        # The vectorize layer is then fed the text data which is batched to improve efficiency
        vectorize_layer.adapt(text_dataset.batch(2048))
        inverse_vocabulary = vectorize_layer.get_vocabulary()

        # The text vector datset is generated by mapping the vectorizer onto the dataset
        text_vector_dataset = text_dataset.batch(2048).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
        
        # The result of the vectorizer being placed on the data is a list of sequences used to generate training data
        sequences = list(text_vector_dataset.as_numpy_iterator())
        
        # Targets, contexts, and labels are generated from the generate training data function which takes in the sequences, a window size, the number
        # of negative samples, vocab size and a random seed to create consistency 
        targets, contexts, labels = generate_training_data(
            sequences=sequences,
            window_size=2,
            num_ns=4,
            vocab_size=vocab_size,
            seed=SEED)

        # The training data is placed into numpy arrays
        targets = np.array(targets)
        contexts = np.array(contexts)
        labels = np.array(labels)
        for seq in sequences[:5]:
            print(f"{seq} => {[inverse_vocabulary[i] for i in seq]}")

        
        BATCH_SIZE = 2048
        BUFFER_SIZE = 10000
        
        # The training data is then placed into a tensorflow tf.data.Dataset to be fed into the model, and then returned
        text_dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        text_dataset = text_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        text_dataset = text_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        return text_dataset, vectorize_layer, vocab_size


    # IF GENERATING TRAINING DATA #  

    # for data in text_datasets:
    #     text_dataset, _ ,vocab_size = data
    #     text_dataset.save('C:/Users/nicho/OneDrive/Desktop/Project/Data/' + str(vocab_size) + 'training_data')

    # IF USING TRAINING DATA TO CREATE EMBEDDINGS FOR MULTIPLE DIMENSIONS #
    for vocab_size in range(20000, 50000, 10000):
        
        # Specifiy a filename that points to the metadata for the specific vocab size
        mname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/vocab' + str(vocab_size) + '.tsv'
        
        # Generate a list of vocabulary words based off of the vocab size
        with open(mname) as metadata:
            words = metadata.read()
            vocab = words.split('\n')
        
        print(vocab[:5])

        # Load the text dataset that was written previously for the specific vocab size
        text_dataset = tf.data.Dataset.load('C:/Users/nicho/OneDrive/Desktop/Project/Data/' + str(vocab_size) + 'training_data')

        # Iterate through all dimensions from 100-1000 creating embeddings
        for dimension in range(100, 1100, 100):

            # Initialize a word2vec object of vocab size and dimension
            word2vec = Word2Vec(vocab_size, dimension)

            # Compile the model using base parameters
            word2vec.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")
            
            # Fit the model with the text dataset, a specific number of epochs, and a tensorboard callback
            word2vec.fit(text_dataset, epochs=20, callbacks=[tensorboard_callback])

            # Generate a summary of the model
            word2vec.summary()

            # Get the weights of the model and generate filenames to write the metadata along with the embeddings
            weights = word2vec.target_embedding.get_weights()[0]
            print(word2vec.target_embedding.get_weights()[0])
            vname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/tensor_flow_vecs/vectors'+ str(vocab_size) + '_' + str(dimension) +'.tsv'            
            # because metadata was already generated and then made into vocab so we will just use the big metadata
            # Open both files and write words with associated embeddings 
            mname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/tensor_flow_vecs/metadata'+ str(vocab_size) + '.tsv'
            print(len(vocab)) 
            print(len(weights))
            with open(vname, 'w', encoding='utf-8') as out_v, open(mname, 'w', encoding='utf-8') as out_m:
                for index, word in enumerate(vocab):
                    if index == 0:
                        continue
                    vec = weights[index]
                    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                    out_m.write(word + '\n')
