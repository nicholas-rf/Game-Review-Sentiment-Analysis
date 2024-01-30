from nltk.tokenize import *
import bag_of_words as bow
import numpy as np
import math

"""
tf_idf module that includes methods for tf_idf vectorization
"""

# vocab_frequency is given at "/Users/nick/Desktop/corpus_vocabulary_frequency.txt"

SAMPLE = """
editor note pokemon scarlet violet expansion comes two parts cannot purchased individually. decided share impressions part one update full review hidden treasure area zero parts available. score attached review subject change based . fan reception scarlet violet launch varied wildly within community. every pokemon fan seems know best series happy share opinions scarlet violet anyone would listen. fans decried bugs struggling frame rate technical problems others appreciated new mechanics open world story enough make peace issues. whichever side fell community split alwayshowever get away release community cools . whove stuck around long term namely competitive players dug claws deep meta. pokemon best teal mask bolsters side series new returning pokemon tms moves welcome quality life changes. wrapped rich new region heartfelt story. teal mask much address scarlet violet poor technical performance changes additions offer exciting first part larger expansion. teal mask billed class field trip. handful students randomly selected study abroad new rural japan inspired region called kitakami. character among lucky whisked away countryside. punchy introduction gets new area quickly catching new returning pokemon right away. kitakami shares lot geographical features paldea rolling green hills serene lakes adorned waterfalls rugged mountain fixed center map sets apart rich culture. folks call kitakami home deep reverence loyal three trio legendary pokemon similar legendary birds dogs previous generations. story goes loyal three protected townsfolk fearsome ogre. student assignment explore kitakami region piece together tale loyal three ogre. simple setup effective way familiarize region history characters. heart soul teal mask two new characters carmine keiran. siblings born raised kitakami attend blueberry academy featured prominently second dlc indigo disk. explore region color discoveries personal experience local insight. quickly learn keiran feels like outsider carmine competitive fault. story provides excellent backdrop characters particularly keiran struggle acceptance mirrors plight loyal three ogre. story builds satisfying conclusion neatly ties things kitakami region also sowing seeds second expansion. touching story regularly took surprise. scarlet violet story exciting revelations sleeve mainline pokemon games often shackled handful recurring tropes gym challenge evil misunderstood faction rival. teal mask refreshing play semi self contained condensed story sheds lot tropes order shine brighter light characters. takes roughly four five hours get altogether took around hours factoring completing kitakami pokedex. wrapped story still reasons explore kitakami including handful secrets powerful new pokemon catch. good narrative chops biggest draw new returning pokemon. dipplin sinistcha loyal three creative additions fit thematically within region culture. meanwhile fan favorites like milotic chandelure gliscor welcome inclusions scarlet violet pokedex. hard say pokemon shake competitive scene new returning moves could certainly make waves grassy glide syrup bomb matcha gotcha. added tms could also make certain pokemon viable . instance scald toxic fairly common competitive teams past generations missing scarlet violet launch. least ev boosting resetting items far plentiful kitakami thanks ogre oustin minigame. makes much easier competitors train new returning pokemon put test. however real test scarlet violet offers tricky trainer battles none consist double battles. without battle tower something similar means way put new returning pokemon paces going online. online functionality decent always provide best space test different strategies builds. based know second expansion sounds like something similar battle tower could return scarlet violet still feel like missing core component endgame experience. teal mask apparent flaw one ultimately held back scarlet violet. performance graphical quality rough. new region plagued low frame rates poor looking textures visual bugs. hoping update might patch issues luck. minor fixes smoother pokemon box navigation hard appreciate given subpar game remains technical standpoint. nailing feelings pokemon scarlet violet past year difficult. review base game said one best mainline pokemon games years hours later still stand . terastallization best battle gimmick series introduced date open world design fits nicely serie themes exploration discovery competitive scene thrill follow participate . however scarlet violet triumphant highs still obscured technical issues said teal mask. scarlet violet core issues still persist lot chew hardcore competitive players fans want see pokemon world. pokemon scarlet violet first dlc expansion teal mask commendable step right direction held back familiar technical shortcomingsnew returning pokemon fit nicely paldea pokedex add interesting wrinkle competitive play. story characters shows heartfelt side pokemon rarely see. new items quality life changes make ev training easier faster. performance issues persist. provide much challenge lacks tools properly test new returning pokemon
"""

def get_vocab(frequency_size, IS_WINDOWS=False):
    """
    Takes a frequency distribution and creates a dictionary that contains words and their total occurences, then writes it to a file
    :param frequency_distribution: an NLTK frequency distribution list of the top 1000 words
    :type frequency_distribution: [(string, integer)]
    """
    frequency_distribution = bow.create_word_distribution(frequency_size)
    vocabulary_counts = {}
    i = 0
    for x, y in frequency_distribution:
        vocabulary_counts[x] = y
        i += 1
    
    if IS_WINDOWS:
        fname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/corpus_vocabulary_frequency_' + str(frequency_size) + '.txt'
    else:
        fname = "/Users/nick/Desktop/corpus_vocabulary_frequency_" + str(frequency_size) + ".txt"
    

    with open(fname, 'w') as f:
        f.write(str(vocabulary_counts))
    
    
def create_vocabulary_frequency_dictionary(frequency_size, IS_WINDOWS=False):
    """
    Creates a dictionary from a file including words along with total occurences
    # this can honestly just be gotten and then stored from the bag of words method, it can write two dictionaries 
    # since what we need for our tf-idf vectors is the n most common words + their indices, and also + their total occurences, which is built in the NLTK freq dist 
    :return: A dictionary of words as keys and their frequency amongst a corpus as values
    :rtype: {word:frequency}
    """
    if IS_WINDOWS:
        fname = 'C:/Users/nicho/OneDrive/Desktop/Project/Data/corpus_vocabulary_frequency_' + str(frequency_size) + '.txt'
    else:
        fname = "/Users/nick/Desktop/corpus_vocabulary_frequency_" + str(frequency_size) + ".txt"
        
    with open(fname, 'r') as f:
        frequency_dict = f.read()
    return eval(frequency_dict)

def generate_idf(vocab_count_dictionary):
    """
    Generates the idf for every word in the vocab_count_dictionary for usage when vectorizing
    :param vocab_count_dictionary: A dictionary of vocab words along with their counts
    :type vocab_count_dictionary: {word:frequency}
    :return: A dictionary containing words as keys and corresponding idf's as values
    :rtype: {word:idf}
    """
    idf_dict = {}
    for x, y in vocab_count_dictionary.items():
        idf_dict[x] = math.log(15413/(y+1))
    return idf_dict

def generate_tf_dictionary(text):
    """
    Generates the term frequency dictionary for a document, containing all words in the document as keys and tf's as values
    :param text: A videogame review
    :type text: str
    :return: A dictionary of terms along with their tf in the document
    :rtype: {term:tf}
    """
    try:
        list_of_words = word_tokenize(text)
        list_of_words.remove(".")
        count = len(list_of_words)
        term_dictionary = {}
        for word in list_of_words:
            if word in term_dictionary:
                term_dictionary[word] += 1
            else:
                term_dictionary[word] = 1 

        for item in term_dictionary.keys():
            term_dictionary[item] /= count
        
        return term_dictionary
    except:
        print(f"the following review was providing an error: {text}")



def create_tf_idf_vector(text, position_dictionary, idf_dictionary, vocab_size):
    """
    Creates a tf-idf vector for a given review using the position dictionary to track what word
    is where on the vector and the idf_dicitonary to get the corresponding words idf value
    :param text: A game review to run through the vectorizer
    :type text: string
    :param position_dictionary: A dictionary of the 1000 most common words in the corpus and their corresponding vector positions
    :type position_dictionary: {word:position}
    :param idf_dictionary: A dictionary of the 1000 most common words in the corpus and their corresponding idf values
    :type idf_dictionary: {word:idf}
    :return: A vector containing the tf-idf for every word in the data
    :rtype: [tf-idf values]
    """
    try:
        term_frequency_dictionary = generate_tf_dictionary(text)
        base_array = np.zeros((vocab_size,), dtype=int)
        review_vector = list(base_array)
        for word, frequency in term_frequency_dictionary.items():
            if word in idf_dictionary:
                review_vector[position_dictionary[word]] = frequency * idf_dictionary[word]
        return review_vector
    except:
        return None
