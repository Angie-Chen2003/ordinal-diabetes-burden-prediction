# Use Pool-trained models to predict on each single country.
# Run this script locally after Pool's 04_collect_tuning.r
# has finished and trained_models.rds has been saved.

source("config.r")
source("unction.r")

library(readxl)

# Load Pool-trained models and preprocessing parameters

# pool_output_dir <- "~/Documents/Jeannette/HPC_code/Output/Pool"

pool_output_dir <- "~/Documents/Jeannette/HPC_code/Output/Balanced_Pool_200"

trained_models <- readRDS(file.path(pool_output_dir, "trained_models.rds"))

model_logit    <- trained_models$model_logit
model_forest   <- trained_models$model_forest
model_svm      <- trained_models$model_svm
model_knn      <- trained_models$model_knn
preProc_pool   <- trained_models$preProc
outcome_levels <- trained_models$outcome_levels

cat("Pool trained models loaded.\n")
cat("Outcome levels:", paste(outcome_levels, collapse = ", "), "\n")

# Define all single countries and their data paths

country_configs <- list(
  list(name = "CHARLS", path = "~/Documents/Jeannette/HPC_code/Dataset/CHARLS_final.xlsx"),
  list(name = "HRS",    path = "~/Documents/Jeannette/HPC_code/Dataset/HRS_final.xlsx"),
  list(name = "KLoSA",  path = "~/Documents/Jeannette/HPC_code/Dataset/KLoSA_final.xlsx"),
  list(name = "LASI",   path = "~/Documents/Jeannette/HPC_code/Dataset/LASI_final.xlsx"),
  list(name = "MAHS",   path = "~/Documents/Jeannette/HPC_code/Dataset/MAHS_final.xlsx"),
  list(name = "SA",     path = "~/Documents/Jeannette/HPC_code/Dataset/SA_final.xlsx"),
  list(name = "SHARE",  path = "~/Documents/Jeannette/HPC_code/Dataset/SHARE_final.xlsx"),
  list(name = "UK",     path = "~/Documents/Jeannette/HPC_code/Dataset/UK_final.xlsx")
)

# Mapping from single country data values to Pool's country_factor values
country_name_map <- c(
  "Mexio"          = "Mexico",
  "South Afirica"  = "South Africa",
  "European"       = "Europe",
  "United Kingdom" = "UK"
)

# All levels as they appear in Pool training data
pool_country_levels <- c("China", "United States", "Korea", "India",
                         "Mexico", "South Africa", "Europe", "UK")

# preprocess single country data to match Pool

preprocess_country_for_pool <- function(data_path, country_name) {
  
  dat <- read_xlsx(data_path)
  
  # Construct outcome
  dat <- dat %>%
    mutate(
      pain   = as.numeric(as.character(pain)),
      vision = as.numeric(as.character(vision))
    ) %>%
    filter(!is.na(pain), !is.na(vision)) %>%
    mutate(burden_score = pain + vision) %>%
    filter(burden_score %in% c(0, 1, 2))
  
  dat$burden_score_ord <- factor(
    dat$burden_score,
    levels = c(0, 1, 2),
    ordered = TRUE
  )
  
  # Remove outcome components to avoid leakage
  dat <- dat %>% dplyr::select(-pain, -vision)
  
  # Remove social_eco (not in Pool)
  if ("social_eco" %in% names(dat)) {
    dat <- dat %>% dplyr::select(-social_eco)
  }
  
  # Rename country to country_factor to match Pool
  if ("country" %in% names(dat)) {
    dat <- dat %>% rename(country_factor = country)
  }
  
  # Fix country name mismatches and set levels to match Pool
  if ("country_factor" %in% names(dat)) {
    dat$country_factor <- as.character(dat$country_factor)
    dat$country_factor <- dplyr::recode(dat$country_factor, !!!country_name_map)
    dat$country_factor <- factor(dat$country_factor, levels = pool_country_levels)
  }
  
  cat("Country:", country_name,
      "| Rows:", nrow(dat),
      "| country_factor value:", as.character(unique(dat$country_factor)),
      "| Outcome distribution:\n")
  print(table(dat$burden_score_ord))
  
  return(dat)
}

# apply Pool preprocessing and predict

predict_with_pool_model <- function(country_data, country_name) {
  
  y_true <- factor(
    country_data$burden_score_ord,
    levels = outcome_levels,
    ordered = TRUE
  )
  
  x_cols <- setdiff(
    names(country_data),
    c("burden_score", "burden_score_ord", "burden_score_int")
  )
  
  x_new <- country_data[, x_cols, drop = FALSE]
  
  # Apply Pool's standardization
  x_new_scaled <- predict(preProc_pool, x_new)
  
  results_list <- list()
  
  # Logit prediction
  results_list[["logit"]] <- tryCatch({
    pred     <- predict(model_logit, newdata = x_new_scaled, type = "class")
    prob_mat <- align_prob_matrix(
      predict(model_logit, newdata = x_new_scaled, type = "probs"),
      outcome_levels
    )
    pred_fac <- factor(pred, levels = outcome_levels, ordered = TRUE)
    cm       <- caret::confusionMatrix(pred_fac, y_true)
    metrics  <- calc_ordinal_metrics(y_true, prob_mat)
    
    eval_ordinal_model(
      model_name      = "Ordinal Logistic Regression",
      y_true          = y_true,
      y_pred          = pred_fac,
      cm_obj          = cm,
      ordinal_metrics = metrics
    ) %>%
      mutate(Country = country_name, Dataset = "Pool_to_Country")
  }, error = function(e) {
    message("Logit prediction error for ", country_name, ": ", e$message)
    NULL
  })
  
  # Forest prediction
  results_list[["forest"]] <- tryCatch({
    pred_out <- predict(model_forest, newdata = x_new_scaled)
    pred_fac <- factor(pred_out$ypred, levels = outcome_levels, ordered = TRUE)
    prob_mat <- align_prob_matrix(pred_out$classprobs, outcome_levels)
    cm       <- caret::confusionMatrix(pred_fac, y_true)
    metrics  <- calc_ordinal_metrics(y_true, prob_mat)
    
    eval_ordinal_model(
      model_name      = "Ordinal Forest",
      y_true          = y_true,
      y_pred          = pred_fac,
      cm_obj          = cm,
      ordinal_metrics = metrics
    ) %>%
      mutate(Country = country_name, Dataset = "Pool_to_Country")
  }, error = function(e) {
    message("Forest prediction error for ", country_name, ": ", e$message)
    NULL
  })
  
  # SVM prediction
  results_list[["svm"]] <- tryCatch({
    pred_tbl  <- predict(model_svm, new_data = x_new_scaled, type = "class")
    raw_tbl   <- predict(model_svm, new_data = x_new_scaled, type = "raw")
    pred_fac  <- factor(pred_tbl$.pred_class, levels = outcome_levels, ordered = TRUE)
    score_vec <- as.numeric(as.data.frame(raw_tbl)[[1]])
    cm        <- caret::confusionMatrix(pred_fac, y_true)
    metrics   <- calc_ordinal_metrics_from_score(y_true, score_vec)
    
    eval_ordinal_model(
      model_name      = "Ordinal SVM",
      y_true          = y_true,
      y_pred          = pred_fac,
      cm_obj          = cm,
      ordinal_metrics = metrics
    ) %>%
      mutate(Country = country_name, Dataset = "Pool_to_Country")
  }, error = function(e) {
    message("SVM prediction error for ", country_name, ": ", e$message)
    NULL
  })
  
  # KNN prediction
  results_list[["knn"]] <- tryCatch({
    pred     <- predict(model_knn, newdata = x_new_scaled, type = "raw")
    prob_mat <- align_prob_matrix(
      predict(model_knn, newdata = x_new_scaled, type = "prob"),
      outcome_levels
    )
    pred_fac <- factor(pred, levels = outcome_levels, ordered = TRUE)
    cm       <- caret::confusionMatrix(pred_fac, y_true)
    metrics  <- calc_ordinal_metrics(y_true, prob_mat)
    
    eval_ordinal_model(
      model_name      = "Ordinal KNN",
      y_true          = y_true,
      y_pred          = pred_fac,
      cm_obj          = cm,
      ordinal_metrics = metrics
    ) %>%
      mutate(Country = country_name, Dataset = "Pool_to_Country")
  }, error = function(e) {
    message("KNN prediction error for ", country_name, ": ", e$message)
    NULL
  })
  
  bind_rows(results_list)
}

# Run Pool-to-country validation for all 8 countries

all_country_results <- list()

for (cfg in country_configs) {
  cat("\n=============================================\n")
  cat("Processing country:", cfg$name, "\n")
  
  country_data <- tryCatch({
    preprocess_country_for_pool(cfg$path, cfg$name)
  }, error = function(e) {
    message("Data loading error for ", cfg$name, ": ", e$message)
    NULL
  })
  
  if (!is.null(country_data)) {
    result <- predict_with_pool_model(country_data, cfg$name)
    all_country_results[[cfg$name]] <- result
    cat("Done:", cfg$name, "\n")
  }
}

pool_to_country_results <- bind_rows(all_country_results)

# Format final table

model_order <- c(
  "Ordinal Logistic Regression",
  "Ordinal Forest",
  "Ordinal SVM",
  "Ordinal KNN"
)

country_order <- c("CHARLS", "HRS", "KLoSA", "LASI", "MAHS", "SA", "SHARE", "UK")

pool_to_country_table <- pool_to_country_results %>%
  dplyr::select(
    Country, Model,
    Accuracy, Macro_F1, MAE, MSE, MZOE, Macro_MAE, ORC, GC, ADC
  ) %>%
  arrange(
    factor(Country, levels = country_order),
    factor(Model, levels = model_order)
  )

cat("\n=============================================\n")
cat("Pool-to-country validation results:\n")
print(pool_to_country_table)

# Export to Excel

#output_path <- file.path(
#  pool_output_dir,
#  "Pool_to_country_validation.xlsx"
#)

output_path <- file.path(
  pool_output_dir,
  "Balanced_Pool_200_to_country_validation.xlsx"
)


writexl::write_xlsx(
  list(
    Pool_to_Country = pool_to_country_table,
    Raw_Results     = pool_to_country_results
  ),
  path = output_path
)

cat("\nResults exported to:", output_path, "\n")
