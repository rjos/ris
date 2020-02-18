from src.algorithm.instance_selection.classification.ris.ris import ris1, ris2, ris3, classify
from src.dataset.keel_dataset import load_from_file
from sklearn.metrics import accuracy_score
import numpy as np
import datetime
import time
import json
import sys
import os

BASE_DATA = 'data'
BASE_RESULT = 'results'

def log(msg, end='\n'):
    print(msg, end=end, flush=True)

def kfold(dataset, numFolds):
    # Set dataset path
    path = os.path.join(BASE_DATA, dataset)
    path = os.path.join(path, (f'{dataset}.dat'))

    # load dataset
    dataset = load_from_file(path)

    # split dataset in numFolds
    folders = dataset.get_folders(num_folders = numFolds, normalize=True)

    log(f'\tFold ', end='')

    # For each fold, save train and test sets
    for i, (train, test) in enumerate(folders):
        log(f'{(i + 1)} ', end='')
        train.save(os.path.join(os.path.dirname(path), train.name))
        test.save(os.path.join(os.path.dirname(path), test.name))

def metrics(y_true, y_pred):

    measures = dict()
    measures['accuracy'] = (100 * accuracy_score(y_true, y_pred))

    return measures

def ris(dataset, method, thresholds, fold, n_jobs = 4):
    path = os.path.join(BASE_DATA, dataset)

    results = dict()

    log(' \tThresholder ', end='')

    for t in thresholds:
        log(f'{t} ', end='')

        #Define results for folds
        results[t] = dict()

        # Set name for train and test datasets
        datatrainname = f'{dataset}-{numFolds}-{fold}tra.dat'
        datatestname  = f'{dataset}-{numFolds}-{fold}tst.dat'

        # Load train data
        datatrain = load_from_file(os.path.join(path, datatrainname))

        # Convert data to numpy array
        X_train, y_train = datatrain.get_data_target()
        X_train, y_train = X_train.astype('float'), y_train.astype(int)

        selection = None
        radius = None

        starttime = None
        endtime = None

        # Apply instance selection method
        if method == 'ris1':
            starttime = time.time()
            selection, radius = ris1(X_train, y_train, t, n_jobs=n_jobs)
            endtime = time.time()
        elif method == 'ris2':
            starttime = time.time()
            selection, radius = ris2(X_train, y_train, t, n_jobs=n_jobs)
            endtime = time.time()
        elif method == 'ris3':
            starttime = time.time()
            selection, radius = ris3(X_train, y_train, t, n_jobs=n_jobs)
            endtime = time.time()
        
        results[t]['time'] = (endtime - starttime)

        # When all samples are removed 
        # by instance selection algorithm
        if len(selection) == 0:
            results[t]['reduction'] = 100
            results[t]['validation'] = .0
            results[t]['test'] = .0
            continue

        # important instance from prototype selection method
        X_selection = X_train[selection]
    
        selection = np.array(selection)
        radius = np.concatenate(radius)

        # Compute reduction rate for instance method
        results[t]['reduction'] = (100 * (1 - (X_selection.shape[0] / X_train.shape[0])))

        # Apply train set as validation set
        # classify all instances in train set
        index = classify(X_train, X_selection, radius)
        y_validation = y_train[selection[index]]

        # Compute metrics for validation predictions
        results[t]['validation'] = metrics(y_train, y_validation)

        # Load test set
        datatest  = load_from_file(os.path.join(path, datatestname))

        # Convert data to numpy array
        X_test, y_test = datatest.get_data_target()
        X_test, y_test = X_test.astype('float'), y_test.astype(int)

        # classify all instances in test set
        index = classify(X_test, X_selection, radius)
        y_pred = y_train[selection[index]]

        # Compute metrics for test predictions
        results[t]['test'] = metrics(y_test, y_pred)

    log('')
    return results

if __name__ == '__main__':
    
    # Set available thresholds
    THRESHOLDERS = np.arange(0.0, 1.1, 0.1)
    THRESHOLDERS = np.round(THRESHOLDERS, decimals=1)
    n_jobs = 1

    # Set Folds number
    numFolds = 10

    # Set instance selection method
    methods = ['ris1', 'ris2', 'ris3']

    # Set datasets
    datasets = ['appendicitis']

    for dataset in datasets:
        log(f'{dataset} -> \t', end='')
        log(f'{datetime.datetime.now()}')

        # k-fold cross validation 
        # each time running this code
        # comment to avoid execute this method
        kfold(dataset, numFolds)

        for method in methods:
            log(f'{method}')
            
            try:
                results = dict()
                # Run instance selection method for all datasets
                for i in range(3, (numFolds + 1)):
                    log(f'Fold: {i}', end='')

                    result = ris(dataset, method, THRESHOLDERS, i, n_jobs)
                    results[i] = result

                log('')

                # Create a folder for each instance selection method
                directory = os.path.join(BASE_RESULT, method)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = f'{method}-{dataset}.json'
                with open(os.path.join(directory, filename), 'w') as outfile:
                    json.dump(results, outfile)

            except Exception as e:
                log(f'Error no dataset {dataset}')
                log(f'{e}')
