import numpy as np
from datasets import Dataset, Value, load_dataset, DatasetDict
import pandas as pd
import sys 
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, 'C:/Users/nicho/OneDrive/Desktop/Project')
from Preprocessing import clean_tokens, create_tokens, remove_bother

def prepare_for_bert(text):
    """
    prepare_for_bert takes a chunk of text and runs some of the text normalization functions on it to remove egregious noise
    :param text: A games review text
    :type text: str
    :return: Text without some extra noise
    :rtype: str
    """
    try:
        # Apply regex 1 which removes extra space and lowercases characters
        slightlyprocessed = create_tokens.apply_regex_1(text)

        # Remove excess noise left by review scraping
        sent_tokens = create_tokens.generate_sentence_tokens(slightlyprocessed)
        sent_tokens = remove_bother.remove_JS_or_NA(sent_tokens)

        # Join the tokens into a full review again
        full_text = ".".join(sent_tokens)

        # Remove excess whitespace and then return the review
        full_text = clean_tokens.remove_extra_whitespace(full_text)
        return full_text
    
    except Exception as e:
        print(e)
        return None

def create_bert_dataset():
    """
    create_bert_dataset takes the dataset and prepares it for a bert model
    """

    # Start by reading in the dataset and removing any rows containing messages related to errors
    dataset_fpath = 'C:\\Users\\nicho\\OneDrive\\Desktop\\Project\\Data\\dataset.csv'
    dataframe = pd.read_csv(dataset_fpath)
    dataframe = dataframe[dataframe['Review Score'].str.contains('issue with')==False]
    dataframe = dataframe[dataframe['Review Score'].str.contains('NAN')==False]
    
    # Clean up the review text portion of the bert dataset
    dataframe['Review Text'] = dataframe['Review Text'].apply(lambda x : prepare_for_bert(x))
    dataframe = dataframe.dropna()
    
    # Ensure that all scores are made into float values
    dataframe['Review Score'] = dataframe['Review Score'].apply(lambda x : float(x))
    
    # Remove all potentially missing rows
    missing_rows = dataframe['Review Text'].isnull() | (dataframe['Review Text'] == '')
    dataframe = dataframe.drop(dataframe[missing_rows].index)
    
    # Export the dataframe into a csv to be used for the bert model
    dataframe.to_csv('C:\\Users\\nicho\\OneDrive\\Desktop\\Project\\Data\\bert_dataset.csv')

def prepare_splits():
    """
    prepare_splits takes the BERT dataset and splits it into overall training and testing sets, and then splits the overall training into model training and testing
    so that the model can be trained, and then fed more testing data to generate predictions 
    :return: A dataset dict containing training and testing splits 
    :rtype: Dataset.DatasetDict
    """
    # Loading in the dataset and turning it into a csv
    bert_dataset_fpath = "C:\\Users\\nicho\\OneDrive\\Desktop\\Project\\Data\\bert_dataset.csv"
    dataframe = pd.read_csv(bert_dataset_fpath, dtype = {'Review Score' : float})

    # Drop unnesecary columns and rename review score to label for bert use
    dataframe = dataframe.drop(['Unnamed: 0.1', 'Unnamed: 0.2','Outlet'], axis=1)
    dataframe = dataframe.rename(columns={'Review Score': 'label'})

    # Drop columns containing scores that only appear 2 times so that a sratified split can occur
    value_counts = dataframe['label'].value_counts()
    values_appear_once = value_counts[value_counts < 3].index
    dataframe = dataframe[~dataframe['label'].isin(values_appear_once)]


    # Split the data into an overall train and overall test stratified on review score
    X = dataframe['Review Text'].copy()
    y = dataframe['label'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, stratify=y)
    train_overall_df = pd.concat([y_train, X_train], axis=1)
    test_overall_df = pd.concat([y_test, X_test], axis=1)

    # Save the overall test dataset into a csv to use on the model
    test_overall_df.to_csv("C:/Users/nicho/OneDrive/Desktop/Project/Data/BERT_Testing.csv")
    
    # Using the train_overall dataset to create a train/test split for the model during model training and evaluation 
    X = train_overall_df['Review Text'].copy()
    y = train_overall_df['label'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, stratify=y)
    train_df = pd.concat([y_train, X_train], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
   
    # Cast the BERT testing and training sets into Dataset objects
    training_dataset = Dataset.from_pandas(train_df)
    testing_dataset = Dataset.from_pandas(test_df)
    unsupervised_dataset = Dataset.from_pandas(train_overall_df)

    # Casting as a DatasetDict type for ease of use within model
    dataset = DatasetDict()
    dataset['test'] = testing_dataset
    dataset['train'] = training_dataset
    dataset['unsupervised'] = unsupervised_dataset

    # from here we return a couple things, a dataset dict containing testing and training datasets for the BERT model as a whole and an additional test_overall dataaset to use for BERT predictions
    return dataset


# Create the dataset dictionary object
# create_bert_dataset()

def create_bert_model():
    dataset = prepare_splits()

    print(dataset)

    # Initialize a tokenzier that matches that of the BERT dataset 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Create a preprocess_function that tokenizes the review text column
    def preprocess_function(examples):
        return tokenizer(examples['Review Text'], truncation=True)

    # Apply the tokenizer to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load('accuracy')

    # Create a compute metrics function that evaluates our model and returns evaluation statistics
    def compute_metrics(eval_pred):
        """
        compute_metrics takes in a eval prediction from the model and returns MSE and R^2 values from it
        :param eval_pred: Eval_pred has both predictions and associated actual values
        :type eval_pred: a touple
        :return: A dictionary containing statistics of model performance
        :rtype: {MSE, R^2}
        """
        # Unwrap predictions and their labels from the eval_pred
        predictions, labels = eval_pred

        # Calculate and then return the MSE and R2 from the predictions and labels
        mse = mean_squared_error(labels, predictions, squared=False)
        r2 = r2_score(labels, predictions)
        return {"mse": mse, 'r^2':r2}

    # Select the Bert model we will use along with num_labels set to 1 so that regression is accomplished  
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

        # ignore_mismatched_sizes = True,
        # problem_type = 'regression'

    # Initialize training hyperparameters for the BERT model
    training_args = TrainingArguments(
        output_dir='C:/Users/nicho/OneDrive/Desktop/Project/Data/Modelling/BERT_Model_tuned',
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=18,
        per_device_eval_batch_size=18,
        num_train_epochs=3,
        report_to="none",
        weight_decay=0.01,
        metric_for_best_model='mse')

    # Initialize the trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Fine tune the BERT model
    print("Starting Train")
    trainer.train()

    results = trainer.evaluate()

# metrics = results['eval_metrics']
# print("Metrics during evaluation:", metrics)

# {'eval_loss': 0.9828572273254395, 'eval_mse': 0.9913915395736694, 'eval_r^2': 0.599508648799492, 'eval_runtime': 301.2394, 'eval_samples_per_second': 35.809, 'eval_steps_per_second': 1.992, 'epoch': 1.0}
# {'loss': 3.3085, 'learning_rate': 7.029831387808041e-06, 'epoch': 1.95}
# {'eval_loss': 0.9357492923736572, 'eval_mse': 0.9673413038253784, 'eval_r^2': 0.6187040469593181, 'eval_runtime': 305.8042, 'eval_samples_per_second': 35.274, 'eval_steps_per_second': 1.962, 'epoch': 2.0}
# {'eval_loss': 0.9556600451469421, 'eval_mse': 0.9775786399841309, 'eval_r^2': 0.6105908808967458, 'eval_runtime': 302.5459, 'eval_samples_per_second': 35.654, 'eval_steps_per_second': 1.983, 'epoch': 3.0}
# {'train_runtime': 4570.4682, 'train_samples_per_second': 3.035, 'train_steps_per_second': 0.169, 'train_loss': 2.3782601956418206, 'epoch': 3.0}
# 100%|███████████████████████████████████████████████████████████████-█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 771/771 [1:16:10<00:00,  5.93s/it] 
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [05:01<00:00,  1.99it/s] 
# Traceback (most recent call last):
#   File "e:\bert_modelling copy.py", line 99, in <module>
#     metrics = results['eval_metrics']
#               ~~~~~~~^^^^^^^^^^^^^^^^
# KeyError: 'eval_metrics'
# PS C:\Users\nicho>
