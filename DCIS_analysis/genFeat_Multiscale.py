import os
import pandas as pd
import os
join = os.path.join
import argparse
from graph_util import *
def cal_graphMet_roi(df,roi_cols):
    group_keys = ['slideId'] + roi_cols
    results = []
    for key_tuple, group in df.groupby(group_keys):
        points = group[['X', 'Y']].values
        metrics=calculate_spat_features(points)
        metrics_dict = {key: value for key, value in zip(group_keys, key_tuple)}
        metrics_dict.update(metrics)
        results.append(metrics_dict)
    results_df = pd.DataFrame(results)
    return results_df

def main():
    parser = argparse.ArgumentParser(description="generate spatial features")
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    args = parser.parse_args()

    id_df = pd.read_csv('/mnt/data10/shared/yujie/new_DCIS/patientNum_slideId_upstage_biopsy.csv')
    #out_dir = '/mnt/data10/shared/yujie/new_DCIS/Features_includeH
    out_dir = args.out_dir
    roi_dict = {'Duct':['ductId'],
                'Layer':['ductId','layer'],
                'Niche':['ductId_IHC','clusterId']}
    for roi,roi_cols in roi_dict.items():
        feat_dir = join(out_dir,roi)
        os.makedirs(feat_dir,exist_ok=True)
        if roi =='Niche':
            for stain in ['CA9','LAMP2b']:
                for thres in range(10,45,5):
                    assignedCell_path = f'/mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/{stain}/{stain}_thres{thres}.csv'
                    assigned_cells_df = pd.read_csv(assignedCell_path)
                    assigned_cells_df[['X', 'Y']] = assigned_cells_df[['X', 'Y']].values*0.5022 #pixel to Micrometer,make sure all the dist features are micrometer units
            
                    cells_df = assigned_cells_df.dropna(subset = roi_cols,axis=0)#drop unassigned cells
                    print(f'original count:{len(assigned_cells_df)} ,no-NA count(HE cells within neighborhood):{len(cells_df)}')

                    out_path = os.path.join(feat_dir,f'graphFeat_{roi}_{stain}_{thres}.csv')
                    results_df = cal_graphMet_roi(cells_df, roi_cols)
                    results_df.to_csv(out_path,index=False)
                    
        else:
            stain='CA9'#anyone 
            thres=10#anyone
            assignedCell_path = f'/mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/{stain}/{stain}_thres{thres}.csv'
            assigned_cells_df = pd.read_csv(assignedCell_path)
            assigned_cells_df[['X', 'Y']] = assigned_cells_df[['X', 'Y']].values*0.5022 #pixel to Micrometer,make sure all the dist features are micrometer units
            
            out_path = os.path.join(feat_dir,f'graphFeat_{roi}.csv')
            results_df = cal_graphMet_roi(assigned_cells_df, roi_cols)
            results_df.to_csv(out_path,index=False)
if __name__ == '__main__':
    main()

'''
python genFeat_Multiscale.py -o /mnt/data10/shared/yujie/new_DCIS/Features_includeH >> log_genFeat_H.txt 2>&1 &
'''