from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp

import os
join = os.path.join
import json

import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # Set the logging level to INFO
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format for the log messages
)

def getCount(df):
    df_relevant = df[['slideId', 'patternId', 'Upstage']]

    # Construct a new dataframe where each row is a unique slideId and each column is the proportion of different patternId
    pattern_counts = df_relevant.groupby(['slideId', 'patternId']).size().unstack(fill_value=0)
    
    upstage_info = df_relevant[['slideId', 'Upstage']].drop_duplicates().set_index('slideId')
    pattern_count_final = pattern_counts.merge(upstage_info, left_index=True, right_index=True)
    return pattern_count_final
def getProp(df):
    df_relevant = df[['slideId', 'patternId', 'Upstage']]

    # Construct a new dataframe where each row is a unique slideId and each column is the proportion of different patternId
    pattern_counts = df_relevant.groupby(['slideId', 'patternId']).size().unstack(fill_value=0)
    pattern_proportions = pattern_counts.div(pattern_counts.sum(axis=1), axis=0)

    # Merge with Upstage column, ensuring consistency
    upstage_info = df_relevant[['slideId', 'Upstage']].drop_duplicates().set_index('slideId')
    pattern_proportions_final = pattern_proportions.merge(upstage_info, left_index=True, right_index=True)
    return pattern_proportions_final


def augment_data(X, num_copies=5, noise_factor=0.01):
    """
    Augment the data by adding random Gaussian noise to the original dataset.
    
    Parameters:
    - X: The original dataset (numpy array or pandas DataFrame)
    - num_copies: Number of noisy copies to generate.
    - noise_factor: The standard deviation of the Gaussian noise relative to the data's range.
    
    Returns:
    - X_augmented: The original data combined with the noisy augmented data.
    """
    X = np.asarray(X)
    X_augmented = [X]  # Include the original data
    
    # Generate noisy copies of the data
    for _ in range(num_copies):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        X_augmented.append(X_noisy)
    
    # Stack original and augmented data
    return np.vstack(X_augmented)

def augment_labels(y, num_copies=5):
    """
    Duplicate the target labels to match the augmented data.
    
    Parameters:
    - y: Original labels (numpy array or pandas Series)
    - num_copies: Number of noisy copies created.
    
    Returns:
    - y_augmented: The original labels duplicated for each noisy copy.
    """
    return np.tile(y, num_copies + 1)

def train_random_forest(X, y, cv_type='kfold', refit_metric='roc_auc',
                        augment=True, num_augments=5, noise_factor=0.01):
    if augment:
        X = augment_data(X, num_copies=num_augments, noise_factor=noise_factor)
        y = augment_labels(y, num_copies=num_augments)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
    }
    # param_grid = {
    # 'n_estimators': [50, 100, 200],
    # 'max_depth': [None, 10, 20],
    # 'min_samples_split': [2, 5],
    # 'min_samples_leaf': [1, 2],
    # 'max_features': ['sqrt', 'log2'],  # 'auto' is equivalent to 'sqrt' in classification tasks
    # 'bootstrap': [True]
    # }

    if cv_type == 'kfold':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_type == 'leaveoneout':
        cv = LeaveOneOut()
    else:
        raise ValueError("cv_type must be 'kfold' or 'leaveoneout'")

    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    # Initialize lists to store the cross-validation results
    accuracy_list, precision_list, recall_list, roc_auc_list = [], [], [], []

    # Perform cross-validation manually to augment only the training set
    for train_index, test_index in cv.split(X, y):
        # Split the data into training and validation sets for the current fold
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        # Apply augmentation only on the training set
        if augment:
            X_train = augment_data(X_train, num_copies=num_augments, noise_factor=noise_factor)
            y_train = augment_labels(y_train, num_copies=num_augments)

        # Perform grid search on the augmented training set
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring=scoring, refit=refit_metric)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the best model on the validation set
        y_pred = best_model.predict(X_val)
        y_proba = best_model.predict_proba(X_val)[:, 1]

        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)

        # Append the results for this fold
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        roc_auc_list.append(roc_auc)

    # Calculate the mean and standard deviation of the performance metrics
    results = {
        'accuracy_mean': np.mean(accuracy_list),
        'accuracy_std': np.std(accuracy_list),
        'precision_mean': np.mean(precision_list),
        'precision_std': np.std(precision_list),
        'recall_mean': np.mean(recall_list),
        'recall_std': np.std(recall_list),
        'roc_auc_mean': np.mean(roc_auc_list),
        'roc_auc_std': np.std(roc_auc_list)
    }

    return best_model, grid_search.best_params_, results

def worker_single(cluster_path,process_func,num_augments):
    try:
        cluster_df = pd.read_csv(cluster_path)

        pattern_df = process_func(cluster_df)
        
        X = pattern_df.drop(columns=['Upstage'])
        y = pattern_df['Upstage']
        _, _, results = train_random_forest(X, y,num_augments=num_augments)

        
        return results['roc_auc_mean']
    except Exception as e:
        #print(e)
        return None
def main():
    parser = argparse.ArgumentParser(description="grid search the best classifier")
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-r','--roi_type',type=str,required=True,choices=['Duct', 'Layer','Niche'], 
                        help="The type of ROI. Must be one of 'Duct', 'Layer','Niche'")
    parser.add_argument('-pr', '--process_type', type=str, required=True,choices=['count', 'prop'])
    parser.add_argument('-p', '--num_processors', type=int, default=1, help='Number of processors to use')
    args = parser.parse_args()
    
    p = args.num_processors
    if args.process_type == 'count':
        process_func = getCount
    elif args.process_type == 'prop':
        process_func = getProp
    #ROI: Niche
    if args.roi_type == 'Niche':
        for stain in ['LAMP2b']:#'CA9','LAMP2b','CL'
            logging.info(f'Start processing {stain} Niche')

            # tasks = []
            # for thres in range(10,45,5):
            #     for cluster in range(3,11):
            #         cluster_path = os.path.join(args.input_dir , f'Niche/{stain}_{thres}/cluster{cluster}.csv')
                    
            #         tasks.append((cluster_path,process_func))
            # with mp.Pool(processes=p) as pool:
            #     roc_auc_list = pool.starmap(worker_single, tasks)
            #     roc_auc_array = np.array([x for x in roc_auc_list if x is not None])
            # print(f'Niche-{stain} best ROC AUC:{np.max(roc_auc_array)} at {tasks[np.argmax(roc_auc_array)][0]}')
            thres = 30
            cluster = 8
            cluster_path = os.path.join(args.input_dir , f'Niche/{stain}_{thres}/cluster{cluster}.csv')
            tasks = []
            for num_augments in range(1,6):
                tasks.append((cluster_path,process_func,num_augments))
            with mp.Pool(processes=p) as pool:
                roc_auc_list = pool.starmap(worker_single, tasks)
                roc_auc_array = np.array([x for x in roc_auc_list if x is not None])
            
            for i in range(5):
                roc_auc = roc_auc_array[i]
                logging.info(f'Niche-{stain} ROC AUC for augmenting copies {i}:{roc_auc} at {cluster_path}')
            
    # python classifier_fromCluster_multiscale_noise.py -i ../data/Features -r Niche -pr prop -p 10 &>> ../result/Niche_log_noise.txt &

    #ROI: Duct
    if args.roi_type == 'Duct':
        tasks = []
        
        for cluster in range(3,11):
            cluster_path = os.path.join(args.input_dir,f'Duct/cluster{cluster}.csv')
            
            tasks.append((cluster_path,process_func))
        with mp.Pool(processes=p) as pool:
            roc_auc_list = pool.starmap(worker_single, tasks)
            roc_auc_array = np.array([x for x in roc_auc_list if x is not None])
        print(f'Duct best ROC AUC:{np.max(roc_auc_array)} at {tasks[np.argmax(roc_auc_array)][0]}')
    # python classifier_fromCluster_multiscale.py -i ../data/Features -r Duct -pr prop -p 10 &>> ../result/Duct_log.txt

    #ROI: Layer
    if args.roi_type == 'Layer':
        for layer_type in ['hyp','oxi']:
            
            tasks = []
            
            for cluster in range(3,11):
                cluster_path = os.path.join(args.input_dir,f'Layer/{layer_type}/cluster{cluster}.csv')
                tasks.append((cluster_path,process_func))
            with mp.Pool(processes=p) as pool:
                roc_auc_list = pool.starmap(worker_single, tasks)
                roc_auc_array = np.array([x for x in roc_auc_list if x is not None])
            print(f'Layer-{layer_type} best ROC AUC:{np.max(roc_auc_array)} at {tasks[np.argmax(roc_auc_array)][0]}')
    # python classifier_fromCluster_multiscale.py -i ../data/Features -r Layer -pr prop -p 10 &>> ../result/Layer_log.txt

if __name__ == '__main__':
    main()

