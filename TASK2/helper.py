from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

# INITIALIZE GLOBAL PARAMS
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def prepare_data():
    # LOAD CSV FILES FOR TRAIN DATA AND TEST DATA 
    dataset = load_dataset("DeveloperOats/DBPedia_Classes")
   
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # SPLIT THE DF ACCORDING TO CLASSIFICATION LEVEL
    train_data = split_df(train_df)
    test_data = split_df(test_df)

    for i in range(len(train_data)):
        # COMBINE RESPECTIVE TRAIN AND TEST DATASET FOR EACH LEVEL
        data_dict = DatasetDict({"train":train_data[i], "test":test_data[i]})

        # TOKENIZE THE DATA
        tokens = data_dict.map(preprocess_function, batched=True)    

        # SAVE THESE DATASETS TO DISK
        tokens.save_to_disk("TASK2/tokens/data" + str(i+1))
    
    return None

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(data):
    return tokenizer(data["text"], truncation=True, padding="max_length")
    
def split_df(df):
    
    data = []
    levels = ['l1', 'l2', 'l3']
    for i in range(len(levels)):
         # SPLIT THE DATAFRAME FOR THREE LEVELS OF CLASSIFICATION
        dataframe = df[['text', levels[i]]]
        dataframe.rename(columns={levels[i]:'labels'}, inplace=True)

        # CHANGE THE VALUES OF LABELS FROM STRING TO INT FOR EACH DATAFRAME
        labeltxt = np.loadtxt("TASK2/label_vals/" + levels[i] + ".txt", dtype="str")
        labelint = list(range(len(labeltxt)))
        for i in range(len(labeltxt)):
            dataframe = dataframe.replace({'labels':labeltxt[i]}, {'labels':labelint[i]})

        # CONVERT INTO DATASET DICTIONARY FORM (TO MATCH THE INPUT TYPE USED IN DISTILBERT)
        dictionary = Dataset.from_dict(dataframe)
        data.append(dictionary)

    return data








