# Bootstrap CI for all performance metrics, HPC array job
# Each task runs one bootstrap replicate for all models
# Usage: Rscript 05_bootstrap_ci.R ${SLURM_ARRAY_TASK_ID}
# Submit: sbatch --array=1-1000 jobs/run_bootstrap_ci.sh

source("config.R")
source("00_functions_data.R")

args <- commandArgs(trailingOnly = TRUE)
b    <- as.integer(args[1])

cat("Running bootstrap CI, replicate:", b, "\n")

# Load tuning results from tuning.R

tuning_results <- readRDS(file.path(output_dir, "tuning_collect_results.rds"))

best_logit_formula  <- tuning_results$best_logit_formula
best_logit_spec     <- tuning_results$best_logit_spec
best_of_nsets       <- tuning_results$best_of_nsets
best_of_ntreeperdiv <- tuning_results$best_of_ntreeperdiv
best_of_ntreefinal  <- tuning_results$best_of_ntreefinal
best_of_mtry        <- tuning_results$best_of_mtry
best_svm_cost       <- tuning_results$best_svm_cost
best_svm_sigma      <- tuning_results$best_svm_sigma
best_knn_kmax       <- tuning_results$best_knn_kmax
best_knn_distance   <- tuning_results$best_knn_distance
best_knn_kernel     <- tuning_results$best_knn_kernel

# 2. Helper functions for bootstrap

metric_names_ci <- c(
  "Accuracy", "Macro_F1", "MAE", "MSE", "MZOE",
  "Macro_MAE", "ORC", "GC", "ADC"
)

make_stratified_boot_index <- function(y, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  y           <- factor(y, levels = levels(y), ordered = TRUE)
  idx_by_class <- split(seq_along(y), y)
  idx_boot    <- unlist(
    lapply(idx_by_class, function(idx) {
      sample(idx, size = length(idx), replace = TRUE)
    }),
    use.names = FALSE
  )
  idx_boot
}

extract_svm_score <- function(raw_pred) {
  raw_df <- as.data.frame(raw_pred)
  if (".pred" %in% names(raw_df)) return(as.numeric(raw_df$.pred))
  numeric_cols <- names(raw_df)[sapply(raw_df, is.numeric)]
  if (length(numeric_cols) == 0) {
    stop("No numeric raw-score column found in SVM prediction output.")
  }
  as.numeric(raw_df[[numeric_cols[1]]])
}

# Fit final models on full training data
# (train for train/validation objects, train+valid for test)

fit_and_predict <- function(fit_fun, pred_fun, train_data, eval_data,
                            metric_type, dataset_label) {
  scaled_dat  <- make_scaled_train_valid(train_data, eval_data)
  train_final <- scaled_dat$train_final
  eval_final  <- scaled_dat$valid_final
  
  y_eval         <- factor(eval_final$burden_score_ord,
                           levels = levels(train_final$burden_score_ord),
                           ordered = TRUE)
  outcome_levels <- levels(train_final$burden_score_ord)
  
  model    <- fit_fun(train_final)
  pred_out <- pred_fun(model, eval_final, outcome_levels)
  
  pred_obj <- list(
    model_name  = NULL,
    dataset     = dataset_label,
    y_true      = y_eval,
    pred_class  = factor(pred_out$pred, levels = outcome_levels, ordered = TRUE),
    metric_type = metric_type,
    eval_final  = eval_final,
    predictor_cols = setdiff(names(eval_final), "burden_score_ord")
  )
  
  if (metric_type == "prob") {
    pred_obj$prob_mat  <- pred_out$prob_mat
  } else {
    pred_obj$score_vec <- pred_out$score_vec
  }
  
  pred_obj
}

# Define fit and pred functions for each model
fit_logit  <- function(dat) MASS::polr(formula = best_logit_formula, data = dat, Hess = TRUE)
pred_logit <- function(model, newdat, outcome_levels) {
  list(
    pred     = predict(model, newdata = newdat, type = "class"),
    prob_mat = align_prob_matrix(
      predict(model, newdata = newdat, type = "probs"),
      outcome_levels
    )
  )
}

fit_forest  <- function(dat) {
  ordinalForest::ordfor(
    depvar       = "burden_score_ord",
    data         = dat,
    nsets        = best_of_nsets,
    ntreeperdiv  = best_of_ntreeperdiv,
    ntreefinal   = best_of_ntreefinal,
    mtry         = best_of_mtry,
    perffunction = "probability"
  )
}
pred_forest <- function(model, newdat, outcome_levels) {
  pred <- predict(model, newdata = newdat)
  list(
    pred     = pred$ypred,
    prob_mat = align_prob_matrix(pred$classprobs, outcome_levels)
  )
}

fit_svm  <- function(dat) {
  mildsvm::svor_exc(
    burden_score_ord ~ .,
    data    = dat,
    cost    = best_svm_cost,
    control = list(
      kernel    = "radial",
      sigma     = best_svm_sigma,
      scale     = FALSE,
      max_steps = 5000
    )
  )
}
pred_svm <- function(model, newdat, outcome_levels) {
  x_new    <- newdat %>% dplyr::select(-burden_score_ord)
  pred_tbl <- predict(model, new_data = x_new, type = "class")
  raw_tbl  <- predict(model, new_data = x_new, type = "raw")
  list(
    pred      = pred_tbl$.pred_class,
    score_vec = extract_svm_score(raw_tbl)
  )
}

fit_knn  <- function(dat) {
  kknn::train.kknn(
    burden_score_ord ~ .,
    data     = dat,
    kmax     = best_knn_kmax,
    distance = best_knn_distance,
    kernel   = best_knn_kernel,
    scale    = FALSE
  )
}
pred_knn <- function(model, newdat, outcome_levels) {
  list(
    pred     = predict(model, newdata = newdat, type = "raw"),
    prob_mat = align_prob_matrix(
      predict(model, newdata = newdat, type = "prob"),
      outcome_levels
    )
  )
}

model_specs <- list(
  list(name = "Ordinal Logistic Regression", fit = fit_logit,  pred = pred_logit,  metric_type = "prob"),
  list(name = "Ordinal Forest",              fit = fit_forest, pred = pred_forest, metric_type = "prob"),
  list(name = "Ordinal SVM",                 fit = fit_svm,    pred = pred_svm,    metric_type = "score"),
  list(name = "Ordinal KNN",                 fit = fit_knn,    pred = pred_knn,    metric_type = "prob")
)

# Fit all models once on original data

cat("Fitting models on original data...\n")

pred_objects <- list()

for (spec in model_specs) {
  # Train set evaluation (train on train, evaluate on train)
  pred_objects[[paste0(spec$name, "__Train")]] <- fit_and_predict(
    fit_fun      = spec$fit,
    pred_fun     = spec$pred,
    train_data   = train_ord,
    eval_data    = train_ord,
    metric_type  = spec$metric_type,
    dataset_label = "Train"
  )
  pred_objects[[paste0(spec$name, "__Train")]]$model_name <- spec$name
  
  # Validation set evaluation
  pred_objects[[paste0(spec$name, "__Validation")]] <- fit_and_predict(
    fit_fun      = spec$fit,
    pred_fun     = spec$pred,
    train_data   = train_ord,
    eval_data    = valid_ord,
    metric_type  = spec$metric_type,
    dataset_label = "Validation"
  )
  pred_objects[[paste0(spec$name, "__Validation")]]$model_name <- spec$name
  
  # Test set evaluation (train on train+valid, evaluate on test)
  pred_objects[[paste0(spec$name, "__Test")]] <- fit_and_predict(
    fit_fun      = spec$fit,
    pred_fun     = spec$pred,
    train_data   = train_plus_valid,
    eval_data    = test_ord,
    metric_type  = spec$metric_type,
    dataset_label = "Test"
  )
  pred_objects[[paste0(spec$name, "__Test")]]$model_name <- spec$name
}

cat("All models fitted. Running bootstrap replicate", b, "...\n")

# Run one bootstrap replicate for all models and datasets

compute_metrics_from_pred_obj <- function(pred_obj, index) {
  y_true <- pred_obj$y_true
  y_b    <- factor(y_true[index], levels = levels(y_true), ordered = TRUE)
  pred_b <- factor(pred_obj$pred_class[index], levels = levels(y_true), ordered = TRUE)
  
  cm_b <- caret::confusionMatrix(data = pred_b, reference = y_b)
  
  if (pred_obj$metric_type == "prob") {
    prob_b     <- pred_obj$prob_mat[index, , drop = FALSE]
    ordinal_b  <- calc_ordinal_metrics(y_true_ord = y_b, prob_mat = prob_b)
  } else {
    score_b    <- pred_obj$score_vec[index]
    ordinal_b  <- calc_ordinal_metrics_from_score(y_true_ord = y_b, score_vec = score_b)
  }
  
  eval_ordinal_model(
    model_name      = pred_obj$model_name,
    y_true          = y_b,
    y_pred          = pred_b,
    cm_obj          = cm_b,
    ordinal_metrics = ordinal_b
  ) %>%
    mutate(
      Dataset      = pred_obj$dataset,
      Bootstrap_ID = b
    )
}

boot_results_list <- list()

for (key in names(pred_objects)) {
  pred_obj <- pred_objects[[key]]
  
  idx_b <- make_stratified_boot_index(pred_obj$y_true, seed = 123 + b)
  
  result_b <- tryCatch({
    compute_metrics_from_pred_obj(pred_obj, index = idx_b)
  }, error = function(e) {
    message("Bootstrap error: ", key, ", b=", b, ", error=", e$message)
    NULL
  })
  
  if (!is.null(result_b)) {
    boot_results_list[[key]] <- result_b
  }
}

boot_results <- bind_rows(boot_results_list)

# Save this replicate's results

boot_dir  <- file.path(output_dir, "bootstrap_ci")
if (!dir.exists(boot_dir)) dir.create(boot_dir, recursive = TRUE)

save_path <- file.path(boot_dir, paste0("boot_ci_", b, ".rds"))
saveRDS(boot_results, save_path)

cat("Bootstrap replicate", b, "saved to:", save_path, "\n")