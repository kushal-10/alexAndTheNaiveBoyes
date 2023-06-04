from datasets import Dataset, DatasetDict, load_dataset, ClassLabel
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
    train1, train2, train3 = split_df(train_df)
    test1, test2, test3 = split_df(test_df)

    # print(len(train1['label']))
    # print(train1.head(5))

    # COMBINE TRAIN AND TEST DATASET IN DICTIONARY FORM 
    data1 = DatasetDict({"train":train1, "test":test1})
    data2 = DatasetDict({"train":train2, "test":test2})
    data3 = DatasetDict({"train":train3, "test":test3})

    # TOKENIZE THE DATA
    tokens1 = data1.map(preprocess_function, batched=True)
    tokens2 = data2.map(preprocess_function, batched=True)
    tokens3 = data3.map(preprocess_function, batched=True)

    # tokens1 = data1.map(tokenize, batched=True)
    # tokens2 = data2.map(tokenize, batched=True)
    # tokens3 = data3.map(tokenize, batched=True)

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
    # ADD data['labels'] too (labels are 'str' and not 'int')
    return tokenizer(data["text"], truncation=True, padding="max_length")

# def tokenize(batch):
#     tokens = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)
#     tokens["label"] = [features["label"].str2int(label) if label is not None else None for label in batch["label"]]
#     return tokens

label = ClassLabel(num_classes=9)

def tokenize(batch):
    tokens = tokenizer(batch['text'], padding=True, truncation=True)
    tokens['label'] = label.str2int(batch['label'])
    return tokens
    
def split_df(df):
    
    # SPLIT THE DATAFRAME FOR THREE LEVELS OF CLASSIFICATION
    df1 = df[['text', 'l1']]
    df1.rename(columns={'l1':'label'}, inplace=True)
    # print(df1.head(3))
    l1 = ['Agent', 'Device', 'Event', 'Place', 'Species', 'SportsSeason', 'TopicalConcept', 'UnitOfWork', 'Work']
    l2 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(l1)):
        df1 = df1.replace({'label':l1[i]}, {'label':l2[i]})

    # print(df1.head(3))

    df2 = df[['text', 'l2']]
    df2.rename(columns={'l2':'label'}, inplace=True)
    df3 = df[['text', 'l3']]
    df3.rename(columns={'l3':'label'}, inplace=True)

    # CONVERT INTO DATASET DICTIONARY FORM (TO MATCH THE INPUT TYPE USED IN TUTORIAL)
    t1 = Dataset.from_dict(df1)
    t2 = Dataset.from_dict(df2)
    t3 = Dataset.from_dict(df3)

    return t1, t2, t3








