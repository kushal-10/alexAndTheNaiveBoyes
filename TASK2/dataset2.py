from datasets import Dataset, DatasetDict
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
    # READ CSV FILES FOR TRAIN DATA AND TEST DATA
    train_df = pd.read_csv("TASK2/data/DBPEDIA_train.csv")
    test_df = pd.read_csv("TASK2/data/DBPEDIA_test.csv")

    # SPLIT THE DF ACCORDING TO CLASSIFICATION LEVEL
    train1, train2, train3 = split_df(train_df)
    test1, test2, test3 = split_df(test_df)

    # COMBINE TRAIN AND TEST DATASET IN DICTIONARY FORM 
    data1 = DatasetDict({"train":train1, "test":test1})
    data2 = DatasetDict({"train":train2, "test":test2})
    data3 = DatasetDict({"train":train3, "test":test3})

    # TOKENIZE THE DATA
    tokens1 = data1.map(preprocess_function, batched=True)
    tokens2 = data2.map(preprocess_function, batched=True)
    tokens3 = data3.map(preprocess_function, batched=True)

    # SAVE THESE DATASETS IN DICTIONARY FORM TO DISK
    tokens1.save_to_disk("TASK2/tokens/data1")
    tokens2.save_to_disk("TASK2/tokens/data2")
    tokens3.save_to_disk("TASK2/tokens/data3")
    
    return None

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(data):
    return tokenizer(data["text"], truncation=True)
    
def split_df(df):
    
    # SPLIT THE DATAFRAME FOR THREE LEVELS OF CLASSIFICATION
    df1 = df[['text', 'l1']]
    df1.rename(columns={'l1':'label'}, inplace=True)
    df2 = df[['text', 'l2']]
    df2.rename(columns={'l2':'label'}, inplace=True)
    df3 = df[['text', 'l3']]
    df3.rename(columns={'l3':'label'}, inplace=True)

    # CONVERT INTO DATASET DICTIONARY FORM (TO MATCH THE INPUT TYPE USED IN TUTORIAL)
    t1 = Dataset.from_dict(df1)
    t2 = Dataset.from_dict(df2)
    t3 = Dataset.from_dict(df3)

    return t1, t2, t3








