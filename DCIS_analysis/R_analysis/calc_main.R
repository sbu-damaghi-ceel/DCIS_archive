setwd('/Users/jie/Desktop/dcis_temp/R_analysis')

library(spatstat)
library(SpatEntropy)
library(jsonlite)

calculate_entropies <- function(ppp_obj,key) {
  if (length(levels(marks(ppp_obj))) <= 1) {
    # If only one level, assign zero to all entropy measures
    return(data.frame(
      patient_cluster = key,
      shannon = 0,
      shannonZ = 0,
      leibovici_1 = 0,
      leibovici_2 = 0,
      altieri_1 = 0,
      altieri_2 = 0,
      altieri_3 = 0
    ))
  } else {
    # Calculate different types of entropy
    shannon_entropy <- shannon(ppp_obj)
    shannonZ_entropy <- shannonZ(ppp_obj)
    
    #entropy for paris with distance < ccdist
    leibovici_entropy_1_result <- try(leibovici(ppp_obj, ccdist=20,plotout=FALSE), silent = TRUE)
    if (class(leibovici_entropy_1_result) == "try-error") {
      leibovici_1 <- NA
    } else {
      leibovici_1 <- leibovici_entropy_1_result$rel.leib
    }
    leibovici_entropy_2_result <- try(leibovici(ppp_obj, ccdist=40,plotout=FALSE), silent = TRUE)
    if (class(leibovici_entropy_2_result) == "try-error") {
      leibovici_2 <- NA
    } else {
      leibovici_2 <- leibovici_entropy_2_result$rel.leib
    }
    
    altieri_entropy <- try(altieri(ppp_obj,distbreak=c(20,40),plotout=FALSE),silent=TRUE)
    if (class(altieri_entropy) == "try-error"){
      altieri_1 <- NA
      altieri_2 <- NA
      altieri_3 <- NA
    }else{
      altieri_1 = altieri_entropy$rel.SPI.terms[1]#SPI for less than distbreak[1]
      altieri_2 = altieri_entropy$rel.SPI.terms[2]#SPI for distbreak[1]-distbreak[2]
      altieri_3 = altieri_entropy$rel.SPI.terms[3]#SPI for larger than distbreak[2]
    }
    #The following needs partition of the data, could be quite useful in duct-level
    #batty_entropy <- batty(unmark(ppp_obj))
    #karlstrom_ceccato_entropy <- karlstrom(unmark(ppp_obj))
    
    
    return (data.frame(
      patient_cluster = key,
      shannon = shannon_entropy$rel.shann,#relative shannon, because the number of categories can be different
      shannonZ = shannonZ_entropy$rel.shannZ,#same reason
      leibovici_1 = leibovici_1,
      leibovici_2 = leibovici_2,
      altieri_1 = altieri_1,
      altieri_2 = altieri_2,
      altieri_3 = altieri_3
      #batty = batty_entropy,
      #karlstrom_ceccato = karlstrom_ceccato_entropy
    ))
  }
}
  

calculate_count <- function(ppp_obj, key) {
  total_points <- ppp_obj$n
  marks_counts <- table(marks(ppp_obj))

  counts_df <- data.frame(
    count_CA9 = 0,
    count_Glut1 = 0,
    count_LAMP2b = 0,
    total_count = total_points,
    patient_cluster = key,
    stringsAsFactors = FALSE
  )
  
  if ("CA9" %in% names(marks_counts)) {
    counts_df$count_CA9 <- marks_counts["CA9"]
  }
  if ("Glut1" %in% names(marks_counts)) {
    counts_df$count_Glut1 <- marks_counts["Glut1"]
  }
  if ("LAMP2b" %in% names(marks_counts)) {
    counts_df$count_LAMP2b <- marks_counts["LAMP2b"]
  }
  
  return(counts_df)
}




# ################calculate entropy
# entropy_results <- data.frame()
# #entropy_results <-read.csv('./excision_entropy_20_40.csv')##when resumed calculation
# for (key in names(ppp_objects)) {
#   if (!(key %in% entropy_results$patient_cluster)) {
#     ppp_obj <- ppp_objects[[key]]
#     entropy_df <- calculate_entropies(ppp_obj, key)
#     entropy_results <- rbind(entropy_results, entropy_df)
#   }
# }
# #print(entropy_results)
# write.csv(entropy_results, file = './excision_entropy_20_40.csv', row.names = FALSE)


############calculate spatial function values
source('./cal_spatFunc.R')

# Load the saved ppp objects
ppp_objects <- readRDS("./biopsy_CL_100_5/ppp_objects.rds")

library(jsonlite)
json_file='./biopsy_CL_100_5/biopsy_CL_123_123_all_spatial_functions.json'
if (file.exists(json_file)){
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

# Write the JSON data to a file
write(json_data, file = "./biopsy_CL_100_5/biopsy_CL_123_123_all_spatial_functions.json")


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

