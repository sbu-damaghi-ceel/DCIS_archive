setwd('/Users/jie/Desktop/dcis_temp/R_analysis')
library(jsonlite)
library(tidyverse)
library(spatstat)

biopsy_cells <- read.csv("./biopsy_CL_100_5/biopsy_CL_123_123_eps100_min5.csv")
concave_hulls <- fromJSON("./biopsy_CL_100_5/biopsy_CL_123_123_eps100_min5.json")

# Function to check and correct polygon orientation
ensure_correct_orientation <- function(poly_matrix) {
  if (sp::Polygon(poly_matrix)@area > 0) {#somehow >0 here will give <0 in owin()
    return(poly_matrix[nrow(poly_matrix):1, ])
  }
  return(poly_matrix)
}

# Initialize a list to store ppp objects
ppp_objects <- list()

for (patient_num in names(concave_hulls)) {
  for (cluster_id in names(concave_hulls[[patient_num]])) {
    if (cluster_id==-1){ ##noise points are labelled -1 in DBSCAN 
      next
    }
    sub_df <- biopsy_cells %>%
      filter(patientNum == patient_num & Cluster == as.numeric(cluster_id))
    
    if (nrow(sub_df) == 0) {
      next
    }
    
    points <- sub_df %>% select(X_HE, Y_HE) %>% as.matrix()
    marks <- as.factor(sub_df$stain)
    
    poly_matrix <- matrix(unlist(concave_hulls[[patient_num]][[cluster_id]]), ncol = 2)
    poly_matrix <- ensure_correct_orientation(poly_matrix)
    
    window <- owin(poly = poly_matrix)
    ppp_obj <- ppp(points[,1], points[,2], marks = marks, window = window)
    
    ppp_objects[[paste(patient_num, cluster_id, sep = "_")]] <- ppp_obj
  }
}

# Save all ppp objects together
saveRDS(ppp_objects, "./biopsy_CL_100_5/ppp_objects.rds")
