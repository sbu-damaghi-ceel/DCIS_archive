import openslide
import matplotlib.pyplot as plt
import numpy as np

import cv2
import json
import sys
import pandas as pd
import pickle
import argparse
import multiprocessing as mp

from shapely.geometry import Polygon, MultiPolygon

import os
join = os.path.join 

from plot_util import plot_single
def plot_single_wrapper(pos_args, kw_args):
    return plot_single(*pos_args, **kw_args)


def main():
    parser = argparse.ArgumentParser(description="Process distance threshold")
    parser.add_argument('-l','--level', type=int,required=True, help='magnification level of the svs')
    parser.add_argument('-s','--stain_type',type=str,required=True,choices=['C', 'L','CL'], 
                        help="The type of stain to use. Must be one of 'C', 'L','CL'")
    parser.add_argument('-d','--parent_dir',type=str,required=True)
    parser.add_argument('-hu','--hulls_path',type=str,required=True)
    parser.add_argument('-o','--save_dir',type=str,required=True)
    parser.add_argument('-p', '--num_processors', type=int, default=1, help='Number of processors to use')
    args = parser.parse_args()
    level= args.level
    parent_dir = args.parent_dir
    hulls_path = join(parent_dir,args.hulls_path)
    save_dir = join(parent_dir,args.save_dir)
    os.makedirs(save_dir,exist_ok=True)
    p = args.num_processors

    match args.stain_type:
        case 'C':
            stain_classes = {'CA9': ['1+','2+','3+']}
            stain_colors = {'CA9': 'green'}
        case 'L':
            stain_classes = {'LAMP2b': ['1+','2+','3+']}
            stain_colors = {'LAMP2b': 'blue'}
        case 'CL':
            stain_classes = {'CA9': ['1+','2+','3+'],'LAMP2b': ['1+','2+','3+']}
            stain_colors = {'CA9': 'green','LAMP2b': 'blue'}

    
    pt_df = pd.read_csv('/mnt/data10/shared/yujie/DCIS/ANALYSIS/biopsy_CA9_Glut1_LAMP2b_warped.csv')
    id_df = pd.read_csv('/mnt/data10/shared/yujie/DCIS/patientNum_slideId_upstage_biopsy.csv')
    slide_dir = '/mnt/data10/shared/yujie/DCIS/biopsy_img_svs/HE'
    duct_dict_path = '/mnt/data10/shared/yujie/new_DCIS/cell_data_processed/HE_duct_geometries.pkl'

    
    with open(hulls_path,'rb') as f:
        hulls_dict=pickle.load(f)# in the format of {'patientNum':{'ductId':[]}} 
    
    tasks = []
    for patient_num in pt_df['patientNum'].unique():
        patient_df = pt_df[pt_df['patientNum'] == patient_num]
    
        if patient_df.empty: 
            print(f'patientNum {patient_num} no positive cells')
            continue
        slideId_HE = id_df[(id_df['patientNum']==patient_num)&(id_df['stain']=='HE')]['slideId'].iloc[0]
        slide_path = join(slide_dir,f'{slideId_HE}.svs')
        save_path = join(save_dir,f'{patient_num}_{slideId_HE}.png') 
        
        with open(duct_dict_path,'rb') as f:
            duct_dict = pickle.load(f)
        duct_slide = duct_dict[patient_num]
        with open(hulls_path,'rb') as f:
            hulls_dict = pickle.load(f)
        hulls_slide = hulls_dict.get(str(patient_num), {})

        tasks.append(((level,stain_colors, stain_classes,patient_df,patient_num,slide_path),
                    {'duct_slide': duct_slide, 'hulls_slide': hulls_slide, 'save_path': save_path}))
    with mp.Pool(processes=p) as pool:
        pool.starmap(plot_single_wrapper, [(pos_args, kw_args) for pos_args, kw_args in tasks])
    
if __name__ == '__main__':
    main()