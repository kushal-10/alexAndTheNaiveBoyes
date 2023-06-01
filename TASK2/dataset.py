##### PREPARE THE DATASET (PREPROCESS/TOKENIZE TEXT) #####

# IMPORT NEESSARY LIBRARIES
import pandas as pd
from re import sub
import string
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def preprocess():
    # SAME AS PREPROCESS BUT JUST FOR TRAIN DATA, AS DONE IN NOTEBOOK

    # READ THE CSV FILES OF TRAIN DATA/VAL DATA/TEST DATA
    train = pd.read_csv('TASK2/data/DBPEDIA_train.csv')

    # CLEAN TRAIN AND TEST DATA
    # CONVERT TEXT TO LOWERCASE, REMOVE DIGITS, REMOVE PUNCTUATIONS, REMOVE MORE THAN ONE BLANKSPACE
    clean(train)

    # TOKENIZE DATA
    words = count(train)
    x_train = token(train, words)

    # ENCODE THE LABELS FOR TRAIN AND TEST DATA FOR EACH CLASSIFICATION LEVEL
    y_train_l1 = encode_labels(train.l1)
    y_train_l2 = encode_labels(train.l2)
    y_train_l3 = encode_labels(train.l3)

    CLASS_l1 = train['l1'].nunique()
    CLASS_l2 = train['l2'].nunique()
    CLASS_l3 = train['l3'].nunique()

    # NEW TRAIN TEST SPLIT
    x_train_l1, x_test_l1, y_train_l1, y_test_l1 = train_test_split(x_train, y_train_l1, test_size = .2)
    x_train_l2, x_test_l2, y_train_l2, y_test_l2 = train_test_split(x_train, y_train_l2, test_size = .2)
    x_train_l3, x_test_l3, y_train_l3, y_test_l3 = train_test_split(x_train, y_train_l3, test_size = .2)
 
    l1 = [x_train_l1, x_test_l1, y_train_l1, y_test_l1, CLASS_l1]
    l2 = [x_train_l2, x_test_l2, y_train_l2, y_test_l2, CLASS_l2]
    l3 = [x_train_l3, x_test_l3, y_train_l3, y_test_l3, CLASS_l3]

    return l1, l2, l3, words
    

def clean(data):

    # CONVERT TEXT TO LOWERCASE, REMOVE DIGITS, REMOVE PUNCTUATIONS, REMOVE MORE THAN ONE BLANKSPACE FOR TEST DATA 
    data.text = data.text.apply(lambda x: x.lower())
    data.text = data.text.apply(lambda x: sub('([0-9])','',x))
    data.text = data.text.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    data.text = data.text.apply(lambda x: sub(' +', ' ', x))
    
    return None

def token(data, words):

    tk = Tokenizer(words)
    tk.fit_on_texts(list(data.text))
    x = tk.texts_to_sequences(data.text.values)
    x = pad_sequences(x, maxlen = 100, padding='post')
    
    return x

def encode_labels(data):

    encoder = LabelEncoder()
    y = encoder.fit_transform(data)
    y = to_categorical(y)

    return y

def count(data):
    tk = Tokenizer()
    tk.fit_on_texts(data.text.values)
    minimo = 4
    cnt = 0
    tot_cnt = 0
    freq = 0
    tot_freq = 0

    for key,value in tk.word_counts.items():
        tot_cnt = tot_cnt + 1
        tot_freq = tot_freq + value
        if(value < minimo):
            cnt = cnt + 1
            freq = freq + value

    words = tot_cnt-cnt

    return words