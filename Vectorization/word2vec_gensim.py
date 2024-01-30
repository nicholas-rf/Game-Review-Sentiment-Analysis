from gensim import utils
import gensim.models

class Corpus:
    def __iter__(self, IS_WINDOWS=False):
        corpus_path = '/Users/nick/Desktop/corpus_newline.txt'
        if IS_WINDOWS:
            corpus_path = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/corpus_newline.txt'
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)


def create_word2vec_gensim(embedding_dimension, vocab_size, IS_WINDOWS = False):
    sentences = Corpus()
    model = gensim.models.Word2Vec(
        sentences=sentences,
        min_count=20, 
        vector_size=embedding_dimension, 
        workers=4, 
        epochs=20,
        sg=0,
        max_vocab_size=vocab_size)
    if IS_WINDOWS:
        model = gensim.models.Word2Vec(
            sentences=sentences,
            min_count=20, 
            vector_size=embedding_dimension, 
            workers=8, 
            epochs=20,
            sg=0,
            max_vocab_size=vocab_size)
    

    

    fname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/gensim_models/word2vec_gensim_dim_' + str(embedding_dimension) + 'vocab_' + str(vocab_size) + 'model.bin'
    model.save(fname)
    return fname


    