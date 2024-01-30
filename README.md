# GameReviewSentimentAnalysis
A project utilizing natural language processing to predict video game review scores from written reviews.

### Project Synopsis:

This project aims to predict the review score of a game based off of its review as a float variable. A dataset was created by webscraping data from 3 different gaming news outlets: PCGamer, Gamespot and Destructoid. Once a dataset was aggregated standard text normalization procedures were used to process the data for vectorization techniques. To normalize text, text was put into lowercase, contractions were expanded, stop words and punctuation were removed, and remaining whitespace was cleaned. After normalization bag of words and tf-idf vectors were generated with dimensionality 1000 and 2000. Both bag of words and tf-idf were implemented without the use of any libraries. Once tf-idf and bag of words vectors of different vocab sizes were finished, several word2vec models were trained with both CBOW and skip-gram architecture. CBOW was implemented using the GENSIM library wherease skip-gram was implemented using the Tensorflow keras library. After training was completed, 40 word2vec models were created, each representing a different embedding dimension ranging from 100-1000 for vocab sizes 10000 and 20000. After word embeddings were applied to reviews they were ran through linear regression, k nearest neighbors, boosted tree, and random forest models from sci-kit learn. Attempted hyperparameter tuning lead to higher MSE scores so models were left with default parameters. The best result from standard models was a GENSIM Word2Vec model trained with a size 20,000 vocab and embedding dimension of 700 paired with a gradient boosted tree leaving an MSE of 1.28. For the BERT model, distil-BERT was fine-tuned on the dataset into a regression task using the Hugging Face library. Additionally NVIDIA Cuda 11.8 and cuDNN were used to improve training times for Tensorflow and BERT models. Regarding overall results, the best model was BERT with an MSE of 1.11. Visualizations were created using matplotlib.

### Overall Project ecosystem

#### Webscraping:

Webscraping is done in 2 steps: urls for reviews are aggregated from outlets and then the reviews get scraped for review text and score.
For more detailed code explanations, check the gamespot scrapers. 

#### Preprocessing
Preprocessing was done in 3 main steps: excess noise from the reviews were removed, sentence tokens were created and then tokens were cleaned. In order to preprocess text, simply call the main function on a column of a pandas dataframe using the apply object method.

#### Vectorization
Vectorization techniques were each implemented separately with additional functions to automate the generation of word2vec models for a specifiable range of vocab sizes and dimensions.

#### Modelling
Modelling was done using sci-kit learn. 


