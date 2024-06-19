###Preprocessed IHC files from process_raw_cellDetection.ipynb

import pandas as pd
import numpy as np

from valis import registration, warp_tools

##cells in hyp and oxi
pt_df = pd.read_csv('/mnt/data10/shared/yujie/DCIS/cellDet_data/biopsy_CA9_Glut1_LAMP2b_processed.csv')
all_df=pt_df[(pt_df['Parent'].str.contains('hyp'))|(pt_df['Parent'].str.contains('oxi'))]
#print(all_df['Parent'].unique())


# Set the pyramid level from which the ROI coordinates originated. Usually 0 when working with slides.
COORD_LEVEL = 0
stains = ['CA9', 'Glut1', 'LAMP2b']#cells are identified only from IHC, not from HE


for patient_num in all_df['patientNum'].unique():
    patient_df = all_df[all_df['patientNum'] == patient_num]
    
    registrar_f = f'/mnt/data10/shared/yujie/DCIS/valis_regRef/{patient_num}/data/{patient_num}_registrar.pickle'
    registrar = registration.load_registrar(registrar_f)


    for stain in stains:
        # Filter the DataFrame for the current stain
        stain_df = patient_df[patient_df['stain'] == stain]
        if not stain_df.empty:
            slideId = stain_df['slideId'].iloc[0]
            stain_slide = registrar.get_slide(f'{slideId}_{stain}')
        
            # Convert x, y to NumPy array
            xy_array = stain_df[['X', 'Y']].to_numpy()
            xy_pixel_arr = (xy_array / 0.5022).astype(int)#from micrometer to pixel

            transformed_xy = stain_slide.warp_xy(xy_pixel_arr,
                                                    slide_level=COORD_LEVEL,
                                                    pt_level=COORD_LEVEL)
            transformed_xy_int = np.rint(transformed_xy).astype(int)
            all_df.loc[stain_df.index, f'X_HE'] = transformed_xy_int[:, 0]
            all_df.loc[stain_df.index, f'Y_HE'] = transformed_xy_int[:, 1]
    print(f'{patient_num} done.')
all_df.to_csv('/mnt/data10/shared/yujie/DCIS/ANALYSIS/biopsy_CA9_Glut1_LAMP2b_warped.csv',index=False)
print(len(all_df))
all_df.head()
