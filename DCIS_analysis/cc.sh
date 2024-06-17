#!/bin/bash

# python connectedComponents.py -t 10 11 12 13 14 15 20 -s C -o /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CA9
# python connectedComponents.py -t 10 11 12 13 14 15 20 -s L -o /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/LAMP2b
# python connectedComponents.py -t 10 11 12 13 14 15 20 -s CL -o /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CL

for thres in 10 11 12 13 14 15 20; do
    #python plot_connectedComponents.py -l 1 -s C -d /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CA9 -hu thres${thres}.pkl -o thres${thres} -p 10
    #python plot_connectedComponents.py -l 1 -s L -d /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/LAMP2b -hu thres${thres}.pkl -o thres${thres} -p 10
    python plot_connectedComponents.py -l 1 -s CL -d /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CL -hu thres${thres}.pkl -o thres${thres} -p 10
done

# for thres in 10 11 12 13 14 15 20; do
#     python assignHE.py -p /mnt/data10/shared/yujie/new_DCIS/cell_data_processed/biopsy_HE_processed.csv -hu /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CA9/thres${thres}.pkl -o /mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/CA9/thres${thres}.csv -n 10
#     python assignHE.py -p /mnt/data10/shared/yujie/new_DCIS/cell_data_processed/biopsy_HE_processed.csv -hu /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/LAMP2b/thres${thres}.pkl -o /mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/LAMP2b/thres${thres}.csv -n 10
#     python assignHE.py -p /mnt/data10/shared/yujie/new_DCIS/cell_data_processed/biopsy_HE_processed.csv -hu /mnt/data10/shared/yujie/new_DCIS/concaveHulls_cc/CL/thres${thres}.pkl -o /mnt/data10/shared/yujie/new_DCIS/HE_cellAssignment/CL/thres${thres}.csv -n 10      
# done