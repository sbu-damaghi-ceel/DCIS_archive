"""
    This script is to assign HE cells to ROIs(duct,layer,community)
    Input:
        pt_csv: from ./process_raw_celldetection.ipynb <in micro unit>
        hull_json: regions defined by IHC(concave hulls) from ./connectedComponents.ipynb <in pixel unit>
    Output:
        adjusted_x,adjusted_y: in pixels
        roi labels: slideId, ductId, layer, ductId_IHC, cliusterId
"""
import pandas as pd
import json
from shapely.geometry import Point,Polygon
import re
import time
import os
import pickle

import argparse
import multiprocessing as mp

def assign_cell_to_community(args):
    row, slide_hulls = args
    x_adjusted = round(row['X'] / 0.5022) #micrometer to pixel
    y_adjusted = round(row['Y'] / 0.5022)
    cell_point = Point(x_adjusted, y_adjusted)

    nicheId = None
    ductId_IHC_HE = None
    found = False
    for ductId,duct_hulls in slide_hulls.items():
        for j,region in duct_hulls.items():#region can be either multiPolygon or Polygon
            if region.contains(cell_point):
                nicheId = j
                ductId_IHC_HE = ductId
                found = True
                break
        if found:
            break
    return {
        'Object ID': row['Object ID'],
        'ductId_IHC_HE':ductId_IHC_HE,
        'nicheId': nicheId
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("--pt_csv", "-p",type=str, required=True, help="path for the HE cell csv files.")
    parser.add_argument("--hull_path", '-hu',type=str, required=True, help="path for the hull pickle files.")
    parser.add_argument("--out_path", '-o',type=str, required=True, help="path for the output csv files.")
    parser.add_argument("--process_n", "-n",type=int, required=False, default=8, help="Number of parallel processes")
    args = parser.parse_args()
    out_directory = os.path.dirname(args.out_path)
    os.makedirs(out_directory,exist_ok=True)
    pt_df = pd.read_csv(args.pt_csv)
    with open(args.hull_path,'rb') as f:# in the format of {'patientNum':{'ductId':{'nicheId:Poly/MultiPoly}}} 
        hulls_dict = pickle.load(f)

    cell_data = []
    for patientNum in pt_df['patientNum'].unique():
        start_time = time.time()

        slide_hulls = hulls_dict.get(str(patientNum), {})
        slide_cells = pt_df[pt_df['patientNum'] == patientNum]

        with mp.Pool(args.process_n) as pool:
            results = pool.map(assign_cell_to_community, [(row, slide_hulls) for _, row in slide_cells.iterrows()])
        cell_data.extend(results)
        
        print(f'patientNum {patientNum} done in {time.time() - start_time:.2f} seconds')

    assigned_cells_df = pd.DataFrame(cell_data)
    assigned_cells_df.to_csv(args.out_path, index=False)

"""
for thres in 10 15 20 25 30 35 40; do
    python assignHE.py -p /mnt/data10/shared/yujie/new_DCIS/cell_data_processed/biopsy_HE_processed.csv -hu /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CA9/thres${thres}.pkl -o /mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/CA9/thres{thres}.csv -n 10
    python assignHE.py -p /mnt/data10/shared/yujie/new_DCIS/cell_data_processed/biopsy_HE_processed.csv -hu /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/LAMP2b/LAMP2b_123_thres${thres}.pkl -o /mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/LAMP2b/LAMP2b_thres{thres}.csv -n 10

done
"""