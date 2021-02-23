import numpy as np
import json
import os

def load(filepath):
    # Load json file and return a dictionary with results
    content = None
    with open(filepath, 'r') as input_file:
        content = json.load(input_file)
    return content

def select_best_results(content_file):

    # Get all folds
    folds = list(map(int, content_file.keys()))
    folds.sort()

    best_validation_threshold = None
    best_test_threshold = None

    # Store all reduction rate for each fold
    reductions = []

    # Store the all results for each fold
    accuracies_test_t_test = []
    accuracies_test_t_val = []
    accuracies_validation_t_val = []

    thresholds_test = []
    thresholds_validation = []

    for fold in folds:
        # Get results for each fold
        results_fold = content_file[str(fold)]

        for threshold, results in results_fold.items():
            # Set default threshold for both validation and test sets
            if (best_validation_threshold is None) and (best_test_threshold is None):
                best_validation_threshold = best_test_threshold = threshold
                continue
            
            # Get result for validation set
            validation_result = results['validation']['accuracy']

            # Get result for test set
            test_result = results['test']['accuracy']
            
            # Set the best threshold for validation set
            if validation_result >= results_fold[best_validation_threshold]['validation']['accuracy']:
                best_validation_threshold = threshold
            
            # Set the best threshold for test set
            if test_result >= results_fold[best_test_threshold]['test']['accuracy']:
                best_test_threshold = threshold

        # Get the result for test set using the best test threshold
        accuracies_test_t_test.append(results_fold[best_test_threshold]['test']['accuracy'])
        
        # Get the result for test set using the best validation threshold
        accuracies_test_t_val.append(results_fold[best_validation_threshold]['test']['accuracy'])
        
        # Get the result for validation set, using the best validation threshold
        accuracies_validation_t_val.append(results_fold[best_validation_threshold]['validation']['accuracy'])

        # Save best test and validation threshold
        thresholds_test.append(best_test_threshold)
        thresholds_validation.append(best_validation_threshold)

        # Get the reduction rate for the best validation threshold
        reductions.append(results_fold[best_validation_threshold]['reduction'])

    return (
        np.array(accuracies_test_t_test),
        np.array(accuracies_test_t_val),
        np.array(accuracies_validation_t_val), 
        np.array(reductions), 
        np.array(thresholds_test).astype(float), 
        np.array(thresholds_validation).astype(float)
    )

if __name__ == "__main__":
    # Path to result file
    filepath = os.path.join('results', 'ris1', 'ris1-appendicitis.json')

    # Load the result file
    content_file = load(filepath)

    # Get the best result for validation and test sets
    # for each fold
    acc_test_t_test, acc_test_t_val, acc_validation_t_val, reductions, t_test, t_val = select_best_results(content_file)