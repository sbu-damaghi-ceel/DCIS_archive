import numpy as np
import pandas as pd
import os
import argparse
import pickle
from multiprocessing import Pool

from graph_util import *


def save_hulls_main(filtered_df, dist_thres, hulls_dir):
    
    hulls_path=hulls_dir+f'/thres{dist_thres}.pkl'
    dist_thres /= .5022 #micro to pixel

    region_dict = {}    
    for patient_num in filtered_df['patientNum'].unique():
        region_dict[patient_num] = {}
        patient_data = filtered_df[filtered_df['patientNum'] == patient_num]
        for duct_id_HE in patient_data['ductId_HE'].unique():
            duct_data = patient_data[patient_data['ductId_HE']==duct_id_HE]
        
            cells = duct_data[['X_HE', 'Y_HE']].to_numpy()
            regions = get_occupied_regions(cells, dist_thres)

            region_dict[patient_num][duct_id_HE] = {}
            for niche_id,region in enumerate(regions):
                region_dict[patient_num][duct_id_HE][niche_id] = region
    with open(hulls_path,'wb') as f:
        pickle.dump(region_dict,f)
    
def main():
    parser = argparse.ArgumentParser(description="Process distance threshold")
    parser.add_argument('-t','--dist_thres', type=int,nargs='+', required=True, help='Distance threshold value')
    #parser.add_argument('-p', '--num_processors', type=int, default=1, help='Number of processors to use')
    parser.add_argument('-s','--stain_type',type=str,required=True,choices=['C', 'L','CL'], 
                        help="The type of stain to use. Must be one of 'C', 'L','CL'.")
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()
    dist_thres_list = args.dist_thres

    pt_df = pd.read_csv('/mnt/data10/shared/yujie/new_DCIS/cell_data_processed/biopsy_CGL_warped_assignHEDuct.csv')
    id_df = pd.read_csv('/mnt/data10/shared/yujie/new_DCIS/patientNum_slideId_upstage_biopsy.csv')
    HE_dir = '/mnt/data10/shared/yujie/DCIS/biopsy_img_svs/HE'

    hulls_dir = args.output_dir
    os.makedirs(hulls_dir,exist_ok=True)

    match args.stain_type:
        
        case 'C':
            stain_classes = {'CA9': ['1+','2+','3+']}
            
        case 'L':
            stain_classes = {'LAMP2b': ['1+','2+','3+']}
            
        case 'CL':
            stain_classes = {'CA9': ['1+','2+','3+'],'LAMP2b': ['1+','2+','3+']}

    dfs = []
    for stain, classes in stain_classes.items():
        mask = (pt_df['stain'] == stain) & (pt_df['Class'].isin(classes))
        dfs.append(pt_df[mask])
    filtered_df = pd.concat(dfs, ignore_index=True)
    filtered_df['habitat'] = filtered_df['Parent'].str[-3:]
    filtered_df['ductId'] = filtered_df['Parent'].str.split('_', expand=True)[0].str.extract(r'Duct (\d+)')[0]

    args_list = [(filtered_df, dist_thres, hulls_dir) for dist_thres in dist_thres_list]
    with Pool(processes=len(dist_thres_list)) as pool:
        pool.starmap(save_hulls_main, args_list)
    

if __name__ == "__main__":
    
    main()
        
        
