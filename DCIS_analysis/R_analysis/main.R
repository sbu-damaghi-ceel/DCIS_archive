setwd('/Users/jie/Desktop/dcis_temp/R_analysis')
source('./cal_spatFunc.R')
library(jsonlite)
library(tidyverse)
library(spatstat)
library(jsonlite)

directory_path <- "./exc44_L_100_5"
#assume the folder only contains one csv and one json file
cell_file <- list.files(directory_path, pattern = "\\.csv$", full.names = TRUE)
hull_file <- list.files(directory_path, pattern = "\\.json$", full.names = TRUE)

cells <- read.csv(cell_file)
hulls <- fromJSON(hull_file)

################################################################
#create ppp object

# Function to check and correct polygon orientation
ensure_correct_orientation <- function(poly_matrix) {
  if (sp::Polygon(poly_matrix)@area > 0) {#somehow >0 here will give <0 in owin()
    return(poly_matrix[nrow(poly_matrix):1, ])
  }
  return(poly_matrix)
}

ppp_objects <- list()
for (patient_num in names(hulls)) {
  for (cluster_id in names(hulls[[patient_num]])) {
    if (cluster_id==-1){ ##noise points are labelled -1 in DBSCAN 
      next
    }
    sub_df <- cells %>%
      filter(patientNum == patient_num & Cluster == as.numeric(cluster_id))
    
    if (nrow(sub_df) == 0) {
      next
    }
    
    points <- sub_df %>% select(X_HE, Y_HE) %>% as.matrix()
    marks <- as.factor(sub_df$stain)
    
    poly_matrix <- matrix(unlist(hulls[[patient_num]][[cluster_id]]), ncol = 2)
    poly_matrix <- ensure_correct_orientation(poly_matrix)
    
    window <- owin(poly = poly_matrix)
    ppp_obj <- ppp(points[,1], points[,2], marks = marks, window = window)
    
    ppp_objects[[paste(patient_num, cluster_id, sep = "_")]] <- ppp_obj
  }
}
saveRDS(ppp_objects, file.path(directory_path,"/ppp_objects.rds"))

################################################################
#calculate spatial function values

# Load the saved ppp objects
#ppp_objects <- readRDS(file.path(directory_path,"ppp_objects.rds"))
json_file=file.path(directory_path,'all_spatial_functions.json')
if (file.exists(json_file)){#for resuming task
  all_spatial_functions <- fromJSON(json_file)
} else{
  all_spatial_functions <- list()
}

for (key in names(ppp_objects)) {
  if (!(key %in% names(all_spatial_functions))) {
    ppp_obj <- ppp_objects[[key]]
    unitname(ppp_obj) <- list("µm", "µm", .5022)
    spatFunc_list <- calculate_spatial_functions(ppp_obj, key)
    all_spatial_functions[[key]] <- spatFunc_list 
  }
}
json_data <- toJSON(all_spatial_functions, pretty = TRUE)
write(json_data, file =json_file)


################################################################
#calculate count
# Load the saved ppp objects
ppp_objects <- readRDS(file.path(directory_path,"ppp_objects.rds"))
dfs_list <- list()
for (key in names(ppp_objects)) {
  ppp_obj <- ppp_objects[[key]]
  counts_df <- calculate_count(ppp_obj, key)
  dfs_list[[key]] <- counts_df
}
combined_df <- do.call(rbind, dfs_list)
count_file=file.path(directory_path,'ihc_counts.csv')
write.csv(combined_df, count_file, row.names = FALSE)

############################
#For visualization
# key = names(ppp_objects)[183]#183 is a huge one
# print(key)
# ppp_obj <- ppp_objects[[key]]
# plot(ppp_obj)
# print(table(marks(ppp_obj)))
# #leibovici_entropy_2_result <- leibovici(ppp_obj, ccdist=40)
# altieri_entropy <- altieri(ppp_obj,distbreak=c(20,40))
# G <- Gest(ppp_obj)
# plot(G)
# F <- Fest(ppp_obj)
# plot(F)
# L <- Lest(ppp_obj)
# plot(L)
# G_cross <- Gcross(ppp_obj, i='CA9',j='Glut1')
# plot(G_cross)
# K_cross <- Kcross(ppp_obj, i='CA9',j='Glut1')
# plot(K_cross)



