setwd('/Users/jie/Desktop/dcis_temp/R_analysis')

library(spatstat)

library(jsonlite)

calculate_spatial_functions <- function(ppp_obj, key) {
  # All cells GFL
  G_obj <- Gest(ppp_obj, correction = 'km')
  F_obj <- Fest(ppp_obj, correction = 'km')
  L_obj <- Lest(ppp_obj, correction = 'iso')
  
  G <- list(r = G_obj$r, theo = G_obj$theo, val = G_obj$km)
  F <- list(r = F_obj$r, theo = F_obj$theo, val = F_obj$km)
  L <- list(r = L_obj$r, theo = L_obj$theo, val = L_obj$iso)
  
  ## single marker colon GFL
  sub_GFL <- list()
  stains <- c('CA9','Glut1','LAMP2b')
  for (stain in stains){
    if (stain %in% names(table(marks(ppp_obj))) && table(marks(ppp_obj))[stain]>=5){
      stain_ppp <- unmark(ppp_obj[ppp_obj$marks == stain])
      G_stain_obj <- Gest(stain_ppp, correction = 'km')
      F_stain_obj <- Fest(stain_ppp, correction = 'km')
      L_stain_obj <- Lest(stain_ppp, correction = 'iso')
      G_stain<- list(r = G_stain_obj$r, theo = G_stain_obj$theo, val = G_stain_obj$km)
      F_stain<- list(r = F_stain_obj$r, theo = F_stain_obj$theo, val = F_stain_obj$km)
      L_stain<- list(r = L_stain_obj$r, theo = L_stain_obj$theo, val = L_stain_obj$iso)
      sub_GFL[[stain]] <- list(subG = G_stain, subF=F_stain,subL = L_stain)
    }
  }
  ## cross functions
  cross_functions <- list()
  mark_pairs <- list(c('CA9', 'Glut1'), c('CA9', 'LAMP2b'), c('Glut1', 'LAMP2b'))
  for (pair in mark_pairs) {
    i <- pair[1]
    j <- pair[2]
    
    # Check if both marks exist and have counts >= 5
    if (i %in% names(table(marks(ppp_obj))) && j %in% names(table(marks(ppp_obj))) &&
        table(marks(ppp_obj))[i] >= 5 && table(marks(ppp_obj))[j] >= 5) {
      
      # Calculate cross functions
      G_cross_obj <- Gcross(ppp_obj, i = i, j = j, correction = 'km')
      #F_cross_obj <- Fcross(ppp_obj, i = i, j = j, correction = 'km')
      L_cross_obj <- Lcross(ppp_obj, i = i, j = j, correction = 'iso')
      
      G_cross <- list(r = G_cross_obj$r, theo = G_cross_obj$theo,val = G_cross_obj$km)
      #F_cross <- list(r = F_cross_obj$r, theo = F_cross_obj$theo,val = F_cross_obj$km)
      L_cross <- list(r = L_cross_obj$r, theo = L_cross_obj$theo,val = L_cross_obj$iso)
      
      
      # Store in list
      cross_functions[[paste(i, j, sep = "&")]] <- list(G_cross = G_cross, L_cross = L_cross)
    }
  }
  
  # Combine results
  spatial_functions <- list(
    patient_cluster = key,
    G = G,
    F = F,
    L = L,
    sub_GFL = sub_GFL,
    cross_functions = cross_functions
  )
  
  return(spatial_functions)
}

####same function to calculate G_singlestain,F_singlestain,L_singlestain
##(should include in the calculate_spatial_functions )
calculate_extra_spatial_functions <- function(ppp_obj, key) {
  stains <- c('CA9','Glut1','LAMP2b')
  sub_GFL <- list()
  for (stain in stains){
    if (stain %in% names(table(marks(ppp_obj))) && table(marks(ppp_obj))[stain]>=5){
      stain_ppp <- unmark(ppp_obj[ppp_obj$marks == stain])
      G_stain_obj <- Gest(stain_ppp, correction = 'km')
      F_stain_obj <- Fest(stain_ppp, correction = 'km')
      L_stain_obj <- Lest(stain_ppp, correction = 'iso')
      G_stain<- list(r = G_stain_obj$r, theo = G_stain_obj$theo, val = G_stain_obj$km)
      F_stain<- list(r = F_stain_obj$r, theo = F_stain_obj$theo, val = F_stain_obj$km)
      L_stain<- list(r = L_stain_obj$r, theo = L_stain_obj$theo, val = L_stain_obj$iso)
      sub_GFL[[stain]] <- list(subG = G_stain, subF=F_stain,subL = L_stain)
    }
  }

  # Combine results
  spatial_functions <- list(
    patient_cluster = key,
    sub_GFL = sub_GFL
  )
  
  return(spatial_functions)
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