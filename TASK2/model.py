##### MODEL ARCHITECTURES #####

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

REDUTOR = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
PARADA = EarlyStopping(monitor='val_loss',patience=2, min_delta=0.0001)
#HYPERPARAMETERS
DIM = 100
EPOCHS = 10
SIZE = 128

def model1(x_train_l1, x_test_l1, y_train_l1, y_test_l1, words, CLASS_l1):

    FILENAME_l1 = 'best_model_l1.h5'
    CHECK = ModelCheckpoint(FILENAME_l1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # MODEL ARCHITECTURE
    model_l1 = Sequential()
    model_l1.add(Embedding(words, DIM, input_length=x_train_l1.shape[1]))
    model_l1.add(Conv1D(128, 5, activation='relu'))
    model_l1.add(GlobalMaxPooling1D()) # added
    model_l1.add(Dense(128, activation='relu'))
    model_l1.add(Dropout(0.1))
    model_l1.add(Dense(32, activation='relu'))
    model_l1.add(Dense(CLASS_l1, activation='softmax')) 
    model_l1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_l1.summary()

    train(model_l1, x_train_l1, y_train_l1, x_test_l1, y_test_l1, CHECK)

    return None

def model2(x_train_l2, x_test_l2, y_train_l2, y_test_l2, words, CLASS_l2):

    FILENAME_l2 = 'best_model_l2.h5'
    CHECK = ModelCheckpoint(FILENAME_l2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model_l2 = Sequential()
    model_l2.add(Embedding(words, DIM, input_length=x_train_l2.shape[1]))
    model_l2.add(Conv1D(256, 5, activation='relu'))
    model_l2.add(GlobalMaxPooling1D()) # added
    model_l2.add(Dense(128, activation='relu'))
    model_l2.add(Dropout(0.1))
    model_l2.add(Dense(128, activation='relu'))
    model_l2.add(Dense(CLASS_l2, activation='softmax')) 
    model_l2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train(model_l2, x_train_l2, y_train_l2, x_test_l2, y_test_l2, CHECK)

    return None

def model3(x_train_l3, x_test_l3, y_train_l3, y_test_l3, words, CLASS_l3):

    FILENAME_l3 = 'best_model_l3.h5'
    CHECK = ModelCheckpoint(FILENAME_l3, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model_l3 = Sequential()
    model_l3.add(Embedding(words, DIM, input_length=x_train_l3.shape[1]))
    model_l3.add(Conv1D(256, 5, activation='relu'))
    model_l3.add(GlobalMaxPooling1D()) # added
    model_l3.add(Dense(256, activation='relu'))
    model_l3.add(Dropout(0.1))
    model_l3.add(Dense(256, activation='relu'))
    model_l3.add(Dense(CLASS_l3, activation='softmax')) 
    model_l3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train(model_l3, x_train_l3, y_train_l3, x_test_l3, y_test_l3, CHECK)

    return None

def train(model, x_train, y_train, x_test, y_test, C):

    history_l1 = model.fit(x_train, y_train, epochs=EPOCHS, verbose=1, validation_split=0.2, 
                            batch_size=SIZE,callbacks=[REDUTOR,PARADA,C])

    score_l1 = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score_l1[0])
    print('Test accuracy:', score_l1[1])

    return None