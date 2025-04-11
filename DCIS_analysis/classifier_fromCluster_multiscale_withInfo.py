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

import pdb
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # Set the logging level to INFO
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format for the log messages
)
# Suppress the specific warning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


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

def addInfo(pattern_final,df_info):
    df_info = df_info[df_info['stain']=='HE']# use HE slide number
    df_info = df_info.set_index('slideId')
    features = ['age_at_core_biopsy', 'race_status', 'ethnicity_status', 'core_biopsy_dcis_path',
            'core_bx_dcis_er', 'core_bx_dcis_pr']

    X = df_info[features]
    
    # Apply one-hot encoding to categorical features
    X = pd.get_dummies(X, columns=['race_status', 'ethnicity_status', 
                                    'core_biopsy_dcis_path', 
                                    'core_bx_dcis_er', 'core_bx_dcis_pr'], drop_first=True)
    
    return pattern_final.merge(X, left_index=True, right_index=True)
def train_random_forest(X, y, cv_type='kfold', refit_metric='roc_auc'):
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

    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring=scoring, refit=refit_metric)
    grid_search.fit(X, y)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = cross_validate(best_model, X, y, cv=cv, scoring=scoring)

    results = {
        'accuracy_mean': np.mean(cv_results['test_accuracy']),
        'accuracy_std': np.std(cv_results['test_accuracy']),
        'precision_mean': np.mean(cv_results['test_precision']),
        'precision_std': np.std(cv_results['test_precision']),
        'recall_mean': np.mean(cv_results['test_recall']),
        'recall_std': np.std(cv_results['test_recall']),
        'roc_auc_mean': np.mean(cv_results['test_roc_auc']),
        'roc_auc_std': np.std(cv_results['test_roc_auc'])
    }

    return best_model, best_params, results

def worker_single(cluster_path,process_func,df_info):
    try:
        cluster_df = pd.read_csv(cluster_path)

        pattern_df = process_func(cluster_df)
        pattern_df = addInfo(pattern_df,df_info)
        
        X = pattern_df.drop(columns=['Upstage'])
        X.columns = X.columns.astype(str)
        y = pattern_df['Upstage']
        
        _, _, results = train_random_forest(X, y)

        return results['roc_auc_mean']
    except Exception as e:
        logging.error(f'Error in {cluster_path}: {e}')
        # print(e)
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

    info_df = pd.read_csv('../data/clinical_data.csv')

    #ROI: Niche
    if args.roi_type == 'Niche':
        # for stain in ['LAMP2b']:#'CA9','LAMP2b',
        #     tasks = []
        #     for thres in range(10,45,5):
        #         for cluster in range(3,11):
        #             cluster_path = os.path.join(args.input_dir , f'Niche/{stain}_{thres}/cluster{cluster}.csv')
                    
        #             tasks.append((cluster_path,process_func,info_df))
            
            
            # with mp.Pool(processes=p) as pool:
            #     roc_auc_list = pool.starmap(worker_single, tasks)
            #     roc_auc_array = np.array([x for x in roc_auc_list if x is not None])
            # logging.info(f'Niche-{stain} best ROC AUC:{np.max(roc_auc_array)} at {tasks[np.argmax(roc_auc_array)][0]}')
        
        stain = 'LAMP2b'
        thres = 30
        cluster = 8
        cluster_path = os.path.join(args.input_dir , f'Niche/{stain}_{thres}/cluster{cluster}.csv')
        roc_auc_mean = worker_single(cluster_path,process_func,info_df)
        logging.info(f'Niche-{stain} best ROC AUC:{roc_auc_mean} at {cluster_path}')

        stain = 'CA9'
        thres = 30
        cluster = 9
        cluster_path = os.path.join(args.input_dir , f'Niche/{stain}_{thres}/cluster{cluster}.csv')
        roc_auc_mean = worker_single(cluster_path,process_func,info_df)
        logging.info(f'Niche-{stain} best ROC AUC:{roc_auc_mean} at {cluster_path}')
    # python classifier_fromCluster_multiscale_withInfo.py -i ../data/Features -r Niche -pr prop -p 10 &>> ../result/Niche_log_withInfo.txt &

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

