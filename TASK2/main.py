from helper import prepare_data
from train import train
from predict import predictions
import argparse

import warnings
warnings.filterwarnings('ignore')

def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prepare', dest='prepare',
        help='Prepare the Training and Test data according to each classification Level and save the data to disk',
        action='store_true'
    )

    parser.add_argument(
        '--train1', dest='train1',
        help='Use this to train the model for Level 1 classification',
        action='store_true'
    )

    parser.add_argument(
        '--train2', dest='train2',
        help='Use this to train the model for Level 2 classification',
        action='store_true'
    )

    parser.add_argument(
        '--train3', dest='train3',
        help='Use this to train the model for Level 3 classification',
        action='store_true'
    )

    parser.add_argument(
        '--predict', dest='predict',
        help='Use this to predict and save the outputs for the test dataset',
        action='store_true'
    )

    args = parser.parse_args()

    if args.prepare:
        print("Splitting the Data according to each classification Level and saving to disk.....")
        prepare_data()

    elif args.train1:
        print("Training the model for Level 1 classification.....")
        train(9)

    # elif args.train2:
    #     print("Training the model for Level 2 classification.....")
    #     train(70)
    
    # elif args.train3:
    #     print("Training the model for Level 3 classification.....")
    #     train(219)

    else:
        print("Getting predictions and saving them.....")
        predictions()


    return None
    

if __name__ == '__main__':
    
    main()
