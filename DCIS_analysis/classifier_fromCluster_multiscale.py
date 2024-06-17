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

def worker_single(cluster_path,out_dir,process_func):
    try:
        cluster_df = pd.read_csv(cluster_path)

        pattern_df = process_func(cluster_df)
        
        X = pattern_df.drop(columns=['Upstage'])
        y = pattern_df['Upstage']
        _, best_params, results = train_random_forest(X, y)

        params_output_path = join(out_dir, 'best_params.json')
        with open(params_output_path, 'w') as f:
            json.dump(best_params, f)
        results_output_path = join(out_dir, 'results.json')
        with open(results_output_path, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        print(e)
def main():
    parser = argparse.ArgumentParser(description="grid search the best classifier")
    
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
        for NA_type in ['noNAcol','imputed']:
            out_parent = f'/mnt/data10/shared/yujie/new_DCIS/{args.process_type}_classifier/Niche_{NA_type}'#use getCount() at line 94
            os.makedirs(out_parent,exist_ok=True)
            tasks = []
            for stain in ['CA9','LAMP2b']:
                for thres in range(10,45,5):
                    for cluster in range(3,11):
                        cluster_path = f'/mnt/data10/shared/yujie/new_DCIS/Features/Niche/{stain}_{thres}/{NA_type}/cluster{cluster}.csv'
                        #cluster_path = f'/mnt/data10/shared/yujie/new_DCIS/Features/Niche/{stain}_{thres}/noNAcol/cluster{cluster}.csv'
                        out_dir = join(out_parent,stain,f'thres{thres}',f'cluster{cluster}')
                        os.makedirs(out_dir,exist_ok=True)
                        tasks.append((cluster_path,out_dir,process_func))
            with mp.Pool(processes=p) as pool:
                pool.starmap(worker_single, tasks)
    #ROI: Duct
    if args.roi_type == 'Duct':
        
        out_parent = f'/mnt/data10/shared/yujie/new_DCIS/{args.process_type}_classifier/Duct'
        os.makedirs(out_parent,exist_ok=True)
        tasks = []
        
        for cluster in range(3,11):
            cluster_path = f'/mnt/data10/shared/yujie/new_DCIS/Features/Duct/cluster{cluster}.csv'
            out_dir = join(out_parent,f'cluster{cluster}')
            os.makedirs(out_dir,exist_ok=True)
            tasks.append((cluster_path,out_dir,process_func))
        with mp.Pool(processes=p) as pool:
            pool.starmap(worker_single, tasks)
    #ROI: Layer
    if args.roi_type == 'Layer':
        for subdir in ['noNAcol_oxi','noNAcol_hyp','imputed_oxi','imputed_hyp']:
            out_parent = f'/mnt/data10/shared/yujie/new_DCIS/{args.process_type}_classifier/Layer'
            os.makedirs(out_parent,exist_ok=True)
            tasks = []
            
            for cluster in range(3,11):
                cluster_path = f'/mnt/data10/shared/yujie/new_DCIS/Features/Layer/{subdir}/cluster{cluster}.csv'
                out_dir = join(out_parent,subdir,f'cluster{cluster}')
                os.makedirs(out_dir,exist_ok=True)
                tasks.append((cluster_path,out_dir,process_func))
            with mp.Pool(processes=p) as pool:
                pool.starmap(worker_single, tasks)
if __name__ == '__main__':
    main()

