# FOR BASELINE
# from dataset import preprocess
# from model import model1, model2, model3

# FOR HUGGINGFACE
from dataset2 import prepare_data

import warnings
warnings.filterwarnings('ignore')

def main():

    ###############################FOR HUGGINGFACE###################################
    # UNCOMMENT THIS TO SAVE THE PROCESSED DATA TO DISK (approx 3 mins.)
    prepare_data()

    return None
    ###############################FOR BASELINE#######################################
    # # GET THE PROCESSED TRAINING AND TEST DATA FOR THE MODEL
    # l1, l2, l3, words = preprocess()

    # # TRAIN THE MODELS
    # # SELECT THE LEVEL OF CLASSIFIER [1, 2, 3, 'all']
    # level = 0

    # if level == 1 or level == 'all':
    #     print("Training Level 1 Classifier")
    #     x_train_l1, x_test_l1, y_train_l1, y_test_l1, CLASS_l1 = l1
    #     model1(x_train_l1, x_test_l1, y_train_l1, y_test_l1, words, CLASS_l1)  

    # if level == 2 or level == 'all':
    #     print("Training Level 2 Classifier")
    #     x_train_l2, x_test_l2, y_train_l2, y_test_l2, CLASS_l2 = l2
    #     model2(x_train_l2, x_test_l2, y_train_l2, y_test_l2, words, CLASS_l2)  
    
    # if level == 3 or level == 'all':
    #     print("Training Level 3 Classifier")
    #     x_train_l3, x_test_l3, y_train_l3, y_test_l3, CLASS_l3 = l3
    #     model3(x_train_l3, x_test_l3, y_train_l3, y_test_l3, words, CLASS_l3)  
    
    # return None


if __name__ == '__main__':
    
    main()
