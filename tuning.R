### Tuning
# Ordinal Forest hyperparameter tuning, HPC array job
# Usage: Rscript 01_tuning_forest.R ${SLURM_ARRAY_TASK_ID}
# Submit: sbatch --array=1-108 jobs/run_forest.sh

source("config.R")
source("00_functions_data.R")

args <- commandArgs(trailingOnly = TRUE)
i    <- as.integer(args[1])

cat("Running Ordinal Forest tuning, parameter set:", i, "\n")

# Define tuning grid (3 x 3 x 3 x 3 = 108 combinations)

ordfor_grid <- tidyr::expand_grid(
  nsets       = c(20, 50, 100),
  ntreeperdiv = c(10, 20, 50),
  ntreefinal  = c(200, 300, 500),
  mtry        = c(1, 2, 3)
)

if (i < 1 || i > nrow(ordfor_grid)) {
  stop("Task ID ", i, " out of range. ordfor_grid has ", nrow(ordfor_grid), " rows.")
}

current_params <- ordfor_grid[i, ]

cat("Current parameters:",
    "nsets =", current_params$nsets,
    "| ntreeperdiv =", current_params$ntreeperdiv,
    "| ntreefinal =", current_params$ntreefinal,
    "| mtry =", current_params$mtry, "\n")

# Run this parameter combination

set.seed(123)

result_i <- tryCatch({
  run_train_valid_model(
    model_name     = "Ordinal Forest",
    model_stage    = "Tuned_Candidate",
    parameter_id   = paste0("OF_", i),
    parameter_text = paste0(
      "nsets=", current_params$nsets,
      "; ntreeperdiv=", current_params$ntreeperdiv,
      "; ntreefinal=", current_params$ntreefinal,
      "; mtry=", current_params$mtry
    ),
    train_data = train_ord,
    valid_data = valid_ord,
    fit_fun = function(dat) {
      ordinalForest::ordfor(
        depvar       = "burden_score_ord",
        data         = dat,
        nsets        = current_params$nsets,
        ntreeperdiv  = current_params$ntreeperdiv,
        ntreefinal   = current_params$ntreefinal,
        mtry         = current_params$mtry,
        perffunction = "probability"
      )
    },
    pred_fun = function(model, newdat, outcome_levels) {
      pred <- predict(model, newdata = newdat)
      list(
        pred     = pred$ypred,
        prob_mat = align_prob_matrix(pred$classprobs, outcome_levels)
      )
    },
    metric_type = "prob"
  )
}, error = function(e) {
  message("Error in parameter set ", i, ": ", e$message)
  return(NULL)
})

# Append parameter columns and save result

if (!is.null(result_i)) {
  result_i <- result_i %>%
    mutate(
      nsets       = current_params$nsets,
      ntreeperdiv = current_params$ntreeperdiv,
      ntreefinal  = current_params$ntreefinal,
      mtry        = current_params$mtry
    )
  
  tuning_dir <- file.path(output_dir, "tuning_forest")
  if (!dir.exists(tuning_dir)) dir.create(tuning_dir, recursive = TRUE)
  
  save_path <- file.path(tuning_dir, paste0("forest_", i, ".rds"))
  saveRDS(result_i, save_path)
  
  cat("Result saved to:", save_path, "\n")
  
} else {
  cat("Parameter set", i, "failed. No result saved.\n")
}

# Ordinal SVM hyperparameter tuning, HPC array job
# Usage: Rscript 02_tuning_svm.R ${SLURM_ARRAY_TASK_ID}
# Submit: sbatch --array=1-20 jobs/run_svm.sh

source("config.R")
source("00_functions_data.R")

args <- commandArgs(trailingOnly = TRUE)
i    <- as.integer(args[1])

cat("Running Ordinal SVM tuning, parameter set:", i, "\n")

# Define tuning grid (5 x 4 = 20 combinations)

svm_grid <- tidyr::expand_grid(
  cost  = c(0.01, 0.05, 0.1, 0.5, 1),
  sigma = c(0.001, 0.005, 0.01, 0.05)
)

if (i < 1 || i > nrow(svm_grid)) {
  stop("Task ID ", i, " out of range. svm_grid has ", nrow(svm_grid), " rows.")
}

current_params <- svm_grid[i, ]

cat("Current parameters:",
    "cost =", current_params$cost,
    "| sigma =", current_params$sigma, "\n")

# Run this parameter combination

set.seed(123)

result_i <- tryCatch({
  run_train_valid_model(
    model_name     = "Ordinal SVM",
    model_stage    = "Tuned_Candidate",
    parameter_id   = paste0("SVM_", i),
    parameter_text = paste0(
      "cost=", current_params$cost,
      "; sigma=", current_params$sigma,
      "; kernel=radial"
    ),
    train_data = train_ord,
    valid_data = valid_ord,
    fit_fun = function(dat) {
      mildsvm::svor_exc(
        burden_score_ord ~ .,
        data    = dat,
        cost    = current_params$cost,
        control = list(
          kernel    = "radial",
          sigma     = current_params$sigma,
          scale     = FALSE,
          max_steps = 5000
        )
      )
    },
    pred_fun = function(model, newdat, outcome_levels) {
      x_new    <- newdat %>% dplyr::select(-burden_score_ord)
      pred_tbl <- predict(model, new_data = x_new, type = "class")
      raw_tbl  <- predict(model, new_data = x_new, type = "raw")
      list(
        pred      = pred_tbl$.pred_class,
        score_vec = raw_tbl$.pred
      )
    },
    metric_type = "score"
  )
}, error = function(e) {
  message("Error in parameter set ", i, ": ", e$message)
  return(NULL)
})

# Append parameter columns and save result

if (!is.null(result_i)) {
  result_i <- result_i %>%
    mutate(
      cost  = current_params$cost,
      sigma = current_params$sigma
    )
  
  tuning_dir <- file.path(output_dir, "tuning_svm")
  if (!dir.exists(tuning_dir)) dir.create(tuning_dir, recursive = TRUE)
  
  save_path <- file.path(tuning_dir, paste0("svm_", i, ".rds"))
  saveRDS(result_i, save_path)
  
  cat("Result saved to:", save_path, "\n")
  
} else {
  cat("Parameter set", i, "failed. No result saved.\n")
}


# Ordinal KNN hyperparameter tuning, HPC array job
# Usage: Rscript 03_tuning_knn.R ${SLURM_ARRAY_TASK_ID}
# Submit: sbatch --array=1-48 jobs/run_knn.sh

source("config.R")
source("00_functions_data.R")

args <- commandArgs(trailingOnly = TRUE)
i    <- as.integer(args[1])

cat("Running Ordinal KNN tuning, parameter set:", i, "\n")

# Define tuning grid (6 x 2 x 4 = 48 combinations)

knn_grid <- tidyr::expand_grid(
  kmax     = c(10, 20, 30, 50, 70, 100),
  distance = c(1, 2),
  kernel   = c("rectangular", "triangular", "epanechnikov", "optimal")
)

if (i < 1 || i > nrow(knn_grid)) {
  stop("Task ID ", i, " out of range. knn_grid has ", nrow(knn_grid), " rows.")
}

current_params <- knn_grid[i, ]

cat("Current parameters:",
    "kmax =", current_params$kmax,
    "| distance =", current_params$distance,
    "| kernel =", current_params$kernel, "\n")

# Run this parameter combination

set.seed(123)

result_i <- tryCatch({
  run_train_valid_model(
    model_name     = "Ordinal KNN",
    model_stage    = "Tuned_Candidate",
    parameter_id   = paste0("KNN_", i),
    parameter_text = paste0(
      "kmax=", current_params$kmax,
      "; distance=", current_params$distance,
      "; kernel=", current_params$kernel
    ),
    train_data = train_ord,
    valid_data = valid_ord,
    fit_fun = function(dat) {
      kknn::train.kknn(
        burden_score_ord ~ .,
        data     = dat,
        kmax     = current_params$kmax,
        distance = current_params$distance,
        kernel   = current_params$kernel,
        scale    = FALSE
      )
    },
    pred_fun = function(model, newdat, outcome_levels) {
      pred <- predict(model, newdata = newdat, type = "raw")
      prob <- predict(model, newdata = newdat, type = "prob")
      list(
        pred     = pred,
        prob_mat = align_prob_matrix(prob, outcome_levels)
      )
    },
    metric_type = "prob"
  )
}, error = function(e) {
  message("Error in parameter set ", i, ": ", e$message)
  return(NULL)
})

# Append parameter columns and save result

if (!is.null(result_i)) {
  result_i <- result_i %>%
    mutate(
      kmax     = current_params$kmax,
      distance = current_params$distance,
      kernel   = current_params$kernel
    )
  
  tuning_dir <- file.path(output_dir, "tuning_knn")
  if (!dir.exists(tuning_dir)) dir.create(tuning_dir, recursive = TRUE)
  
  save_path <- file.path(tuning_dir, paste0("knn_", i, ".rds"))
  saveRDS(result_i, save_path)
  
  cat("Result saved to:", save_path, "\n")
  
} else {
  cat("Parameter set", i, "failed. No result saved.\n")
}

# Collect all tuning results from HPC, run baseline and logit
# tuning locally, select best parameters, run final test
# evaluation.
# Run this script locally after all HPC array jobs finish.

source("config.r")
source("00_functions_data.r")

# Baseline models

baseline_ord_logit <- tryCatch({
  run_train_valid_model(
    model_name     = "Ordinal Logistic Regression",
    model_stage    = "Baseline",
    parameter_id   = "LOGIT_BASE",
    parameter_text = "Full predictor set",
    train_data     = train_ord,
    valid_data     = valid_ord,
    fit_fun = function(dat) {
      MASS::polr(burden_score_ord ~ ., data = dat, Hess = TRUE)
    },
    pred_fun = function(model, newdat, outcome_levels) {
      list(
        pred     = predict(model, newdata = newdat, type = "class"),
        prob_mat = align_prob_matrix(
          predict(model, newdata = newdat, type = "probs"),
          outcome_levels
        )
      )
    },
    metric_type = "prob"
  )
}, error = function(e) {
  message("Logistic Regression baseline failed: ", e$message)
  NULL
})

baseline_ord_forest <- run_train_valid_model(
  model_name     = "Ordinal Forest",
  model_stage    = "Baseline",
  parameter_id   = "OF_BASE",
  parameter_text = "nsets=50; ntreeperdiv=20; ntreefinal=300; mtry=1",
  train_data     = train_ord,
  valid_data     = valid_ord,
  fit_fun = function(dat) {
    ordinalForest::ordfor(
      depvar       = "burden_score_ord",
      data         = dat,
      nsets        = 50,
      ntreeperdiv  = 20,
      ntreefinal   = 300,
      mtry         = 1,
      perffunction = "probability"
    )
  },
  pred_fun = function(model, newdat, outcome_levels) {
    pred <- predict(model, newdata = newdat)
    list(
      pred     = pred$ypred,
      prob_mat = align_prob_matrix(pred$classprobs, outcome_levels)
    )
  },
  metric_type = "prob"
)

baseline_ord_svm <- run_train_valid_model(
  model_name     = "Ordinal SVM",
  model_stage    = "Baseline",
  parameter_id   = "SVM_BASE",
  parameter_text = "cost=0.1; sigma=0.005; kernel=radial",
  train_data     = train_ord,
  valid_data     = valid_ord,
  fit_fun = function(dat) {
    mildsvm::svor_exc(
      burden_score_ord ~ .,
      data    = dat,
      cost    = 0.1,
      control = list(
        kernel    = "radial",
        sigma     = 0.005,
        scale     = FALSE,
        max_steps = 5000
      )
    )
  },
  pred_fun = function(model, newdat, outcome_levels) {
    x_new    <- newdat %>% dplyr::select(-burden_score_ord)
    pred_tbl <- predict(model, new_data = x_new, type = "class")
    raw_tbl  <- predict(model, new_data = x_new, type = "raw")
    list(
      pred      = pred_tbl$.pred_class,
      score_vec = raw_tbl$.pred
    )
  },
  metric_type = "score"
)

baseline_ord_knn <- run_train_valid_model(
  model_name     = "Ordinal KNN",
  model_stage    = "Baseline",
  parameter_id   = "KNN_BASE",
  parameter_text = "kmax=70; distance=1; kernel=triangular",
  train_data     = train_ord,
  valid_data     = valid_ord,
  fit_fun = function(dat) {
    kknn::train.kknn(
      burden_score_ord ~ .,
      data     = dat,
      kmax     = 70,
      distance = 1,
      kernel   = "triangular",
      scale    = FALSE
    )
  },
  pred_fun = function(model, newdat, outcome_levels) {
    pred <- predict(model, newdata = newdat, type = "raw")
    prob <- predict(model, newdata = newdat, type = "prob")
    list(
      pred     = pred,
      prob_mat = align_prob_matrix(prob, outcome_levels)
    )
  },
  metric_type = "prob"
)

baseline_results <- bind_rows(
  baseline_ord_logit,
  baseline_ord_forest,
  baseline_ord_svm,
  baseline_ord_knn
)

baseline_overfit_summary <- summarize_overfitting(baseline_results)

cat("Baseline models done.\n")
print(baseline_overfit_summary)

# Logit tuning (only 2 specs, fast, run locally)

logit_predictors_full <- get_predictor_cols(train_ord)

logit_tuning_specs <- list(
  list(
    id         = "LOGIT_FULL",
    label      = "Full predictor set",
    predictors = logit_predictors_full
  ),
  list(
    id         = "LOGIT_NO_AGE",
    label      = "Remove age if available",
    predictors = setdiff(logit_predictors_full, "age")
  )
)

logit_tuning_specs <- logit_tuning_specs[!duplicated(
  sapply(logit_tuning_specs, function(x) paste(x$predictors, collapse = ";"))
)]

set.seed(123)
logit_tuning_list <- list()

for (i in seq_along(logit_tuning_specs)) {
  current_spec       <- logit_tuning_specs[[i]]
  current_predictors <- current_spec$predictors
  current_formula    <- as.formula(
    paste("burden_score_ord ~", paste(current_predictors, collapse = " + "))
  )
  
  result_i <- tryCatch({
    run_train_valid_model(
      model_name     = "Ordinal Logistic Regression",
      model_stage    = "Tuned_Candidate",
      parameter_id   = current_spec$id,
      parameter_text = current_spec$label,
      train_data     = train_ord,
      valid_data     = valid_ord,
      fit_fun = function(dat) {
        MASS::polr(formula = current_formula, data = dat, Hess = TRUE)
      },
      pred_fun = function(model, newdat, outcome_levels) {
        list(
          pred     = predict(model, newdata = newdat, type = "class"),
          prob_mat = align_prob_matrix(
            predict(model, newdata = newdat, type = "probs"),
            outcome_levels
          )
        )
      },
      metric_type = "prob"
    )
  }, error = function(e) {
    message("Error in logit spec ", i, ": ", e$message)
    return(NULL)
  })
  
  if (!is.null(result_i)) {
    result_i <- result_i %>%
      mutate(
        Logit_Specification  = current_spec$label,
        Number_of_Predictors = length(current_predictors)
      )
    logit_tuning_list[[i]] <- result_i
  }
}

logit_tuning_results <- bind_rows(logit_tuning_list)

# Handle case where logit tuning completely failed

if (is.null(logit_tuning_results) || nrow(logit_tuning_results) == 0) {
  message("Logistic Regression tuning failed for all specs. Creating placeholder.")
  logit_tuning_results <- NULL
  logit_tuning_summary <- NULL
} else {
  logit_tuning_summary <- summarize_overfitting(logit_tuning_results)
}

cat("Logit tuning done.\n")

# Collect forest / svm / knn results from HPC

collect_rds_folder <- function(folder_path) {
  files <- list.files(folder_path, pattern = "\\.rds$", full.names = TRUE)
  if (length(files) == 0) {
    warning("No .rds files found in: ", folder_path)
    return(NULL)
  }
  results <- lapply(files, readRDS)
  bind_rows(results)
}

cat("Collecting Ordinal Forest tuning results...\n")
ordfor_tuning_results <- collect_rds_folder(
  file.path(output_dir, "tuning_forest")
)

cat("Collecting Ordinal SVM tuning results...\n")
svm_tuning_results <- collect_rds_folder(
  file.path(output_dir, "tuning_svm")
)

cat("Collecting Ordinal KNN tuning results...\n")
knn_tuning_results <- collect_rds_folder(
  file.path(output_dir, "tuning_knn")
)

ordfor_tuning_summary <- summarize_overfitting(ordfor_tuning_results)
svm_tuning_summary    <- summarize_overfitting(svm_tuning_results)
knn_tuning_summary    <- summarize_overfitting(knn_tuning_results)

cat("All tuning results collected.\n")

# Rank tuning candidates and select best parameters

all_tuning_summary <- bind_rows(
  logit_tuning_summary,
  ordfor_tuning_summary,
  svm_tuning_summary,
  knn_tuning_summary
)

ranked_tuning_summary <- all_tuning_summary %>%
  group_by(Model) %>%
  mutate(
    Rank_Accuracy  = min_rank(desc(Accuracy_Validation)),
    Rank_Macro_F1  = min_rank(desc(Macro_F1_Validation)),
    Rank_ORC       = min_rank(desc(ORC_Validation)),
    Rank_GC        = min_rank(desc(GC_Validation)),
    Rank_ADC       = min_rank(desc(ADC_Validation)),
    Rank_MAE       = min_rank(MAE_Validation),
    Rank_MSE       = min_rank(MSE_Validation),
    Rank_MZOE      = min_rank(MZOE_Validation),
    Rank_Macro_MAE = min_rank(Macro_MAE_Validation),
    Rank_Overfit   = min_rank(
      pmax(Accuracy_Gap, F1_Gap, ORC_Gap, na.rm = TRUE)
    ),
    Overall_Rank_Score =
      Rank_Accuracy + Rank_Macro_F1 + Rank_ORC + Rank_GC + Rank_ADC +
      Rank_MAE + Rank_MSE + Rank_MZOE + Rank_Macro_MAE + Rank_Overfit
  ) %>%
  ungroup() %>%
  arrange(Model, Overall_Rank_Score)

of_param_lookup <- ordfor_tuning_results %>%
  dplyr::select(Model, Parameter_ID, nsets, ntreeperdiv, ntreefinal, mtry) %>%
  distinct()

svm_param_lookup <- svm_tuning_results %>%
  dplyr::select(Model, Parameter_ID, cost, sigma) %>%
  distinct()

knn_param_lookup <- knn_tuning_results %>%
  dplyr::select(Model, Parameter_ID, kmax, distance, kernel) %>%
  distinct()

# Only add logit lookup if it exists
if (!is.null(logit_tuning_results)) {
  logit_param_lookup <- logit_tuning_results %>%
    dplyr::select(Model, Parameter_ID, Logit_Specification, Number_of_Predictors) %>%
    distinct()
  tuning_parameter_lookup <- bind_rows(
    logit_param_lookup,
    of_param_lookup,
    svm_param_lookup,
    knn_param_lookup
  )
} else {
  tuning_parameter_lookup <- bind_rows(
    of_param_lookup,
    svm_param_lookup,
    knn_param_lookup
  )
}

best_tuned_params <- ranked_tuning_summary %>%
  group_by(Model) %>%
  slice(1) %>%
  ungroup() %>%
  left_join(tuning_parameter_lookup, by = c("Model", "Parameter_ID")) %>%
  mutate(Model_Stage = "Best_Tuned")

cat("Best tuned parameters selected.\n")
print(best_tuned_params %>% dplyr::select(Model, Parameter_ID, Parameter_Text))

# Compare baseline vs best tuned on validation set

baseline_summary   <- baseline_overfit_summary %>% mutate(Model_Stage = "Baseline")
best_tuned_summary <- best_tuned_params %>% mutate(Model_Stage = "Best_Tuned")

pre_post_overfit_summary <- bind_rows(baseline_summary, best_tuned_summary) %>%
  arrange(Model, Model_Stage)

higher_better_metrics <- c("Accuracy", "Macro_F1", "ORC", "GC", "ADC")
lower_better_metrics  <- c("MAE", "MSE", "MZOE", "Macro_MAE")

validation_pre_post_wide <- pre_post_overfit_summary %>%
  dplyr::select(
    Model, Model_Stage, Parameter_ID, Parameter_Text, Overfitting_Status,
    Accuracy_Validation, Macro_F1_Validation, MAE_Validation, MSE_Validation,
    MZOE_Validation, Macro_MAE_Validation, ORC_Validation, GC_Validation, ADC_Validation
  ) %>%
  tidyr::pivot_longer(
    cols      = ends_with("_Validation"),
    names_to  = "Metric",
    values_to = "Validation_Value"
  ) %>%
  mutate(
    Metric = gsub("_Validation", "", Metric),
    Metric_Direction = case_when(
      Metric %in% higher_better_metrics ~ "Higher is better",
      Metric %in% lower_better_metrics  ~ "Lower is better",
      TRUE ~ NA_character_
    )
  ) %>%
  tidyr::pivot_wider(
    names_from  = Model_Stage,
    values_from = c(Validation_Value, Parameter_ID, Parameter_Text, Overfitting_Status)
  ) %>%
  mutate(
    Change = Validation_Value_Best_Tuned - Validation_Value_Baseline,
    Change_Direction = case_when(
      Metric %in% higher_better_metrics & Change > 0  ~ "Better",
      Metric %in% higher_better_metrics & Change < 0  ~ "Worse",
      Metric %in% lower_better_metrics  & Change < 0  ~ "Better",
      Metric %in% lower_better_metrics  & Change > 0  ~ "Worse",
      abs(Change) < 1e-10                             ~ "No change",
      TRUE ~ "Check manually"
    )
  ) %>%
  arrange(Model, Metric)

# Extract best parameter values as scalars

best_logit <- best_tuned_params %>% filter(Model == "Ordinal Logistic Regression")
best_of    <- best_tuned_params %>% filter(Model == "Ordinal Forest")
best_svm   <- best_tuned_params %>% filter(Model == "Ordinal SVM")
best_knn   <- best_tuned_params %>% filter(Model == "Ordinal KNN")

best_of_nsets       <- as.integer(best_of$nsets[[1]])
best_of_ntreeperdiv <- as.integer(best_of$ntreeperdiv[[1]])
best_of_ntreefinal  <- as.integer(best_of$ntreefinal[[1]])
best_of_mtry        <- as.integer(best_of$mtry[[1]])

best_svm_cost  <- as.numeric(best_svm$cost[[1]])
best_svm_sigma <- as.numeric(best_svm$sigma[[1]])

best_knn_kmax     <- as.integer(best_knn$kmax[[1]])
best_knn_distance <- as.numeric(best_knn$distance[[1]])
best_knn_kernel   <- as.character(best_knn$kernel[[1]])

# Only extract logit params if logit tuning succeeded
if (nrow(best_logit) > 0 && !is.null(logit_tuning_specs)) {
  best_logit_spec <- logit_tuning_specs[[
    which(sapply(logit_tuning_specs, function(x) x$id) == best_logit$Parameter_ID)
  ]]
  best_logit_formula <- as.formula(
    paste("burden_score_ord ~", paste(best_logit_spec$predictors, collapse = " + "))
  )
} else {
  best_logit_spec    <- NULL
  best_logit_formula <- NULL
}

# Final test evaluation using best tuned parameters

run_final_test_model <- function(model_name, model_stage, parameter_id,
                                 parameter_text, train_data, test_data,
                                 fit_fun, pred_fun,
                                 metric_type = c("prob", "score")) {
  metric_type <- match.arg(metric_type)
  
  scaled_dat  <- make_scaled_train_test(train_data, test_data)
  train_final <- scaled_dat$train_final
  test_final  <- scaled_dat$test_final
  
  y_test         <- factor(test_final$burden_score_ord,
                           levels = levels(train_final$burden_score_ord),
                           ordered = TRUE)
  outcome_levels <- levels(train_final$burden_score_ord)
  
  model      <- fit_fun(train_final)
  pred_out   <- pred_fun(model, test_final, outcome_levels)
  pred_class <- factor(pred_out$pred, levels = outcome_levels, ordered = TRUE)
  
  cm <- caret::confusionMatrix(pred_class, y_test)
  
  if (metric_type == "prob") {
    ordinal_metrics <- calc_ordinal_metrics(y_test, pred_out$prob_mat)
  } else {
    ordinal_metrics <- calc_ordinal_metrics_from_score(y_test, pred_out$score_vec)
  }
  
  eval_ordinal_model(model_name, y_test, pred_class, cm, ordinal_metrics) %>%
    mutate(
      Model_Stage    = model_stage,
      Dataset        = "Test",
      Parameter_ID   = parameter_id,
      Parameter_Text = parameter_text
    ) %>%
    dplyr::select(Model, Model_Stage, Dataset, Accuracy, Macro_F1, MAE, MSE,
                  MZOE, Macro_MAE, ORC, GC, ADC, Parameter_ID, Parameter_Text)
}

# Logistic Regression final test with tryCatch
final_test_logit <- tryCatch({
  if (is.null(best_logit_formula)) stop("Logistic Regression tuning failed, skipping.")
  run_final_test_model(
    model_name     = "Ordinal Logistic Regression",
    model_stage    = "Best_Tuned",
    parameter_id   = best_logit$Parameter_ID,
    parameter_text = best_logit$Parameter_Text,
    train_data     = train_plus_valid,
    test_data      = test_ord,
    fit_fun = function(dat) {
      MASS::polr(formula = best_logit_formula, data = dat, Hess = TRUE)
    },
    pred_fun = function(model, newdat, outcome_levels) {
      list(
        pred     = predict(model, newdata = newdat, type = "class"),
        prob_mat = align_prob_matrix(
          predict(model, newdata = newdat, type = "probs"),
          outcome_levels
        )
      )
    },
    metric_type = "prob"
  )
}, error = function(e) {
  message("Logistic Regression final test failed: ", e$message)
  NULL
})

final_test_forest <- run_final_test_model(
  model_name     = "Ordinal Forest",
  model_stage    = "Best_Tuned",
  parameter_id   = best_of$Parameter_ID,
  parameter_text = best_of$Parameter_Text,
  train_data     = train_plus_valid,
  test_data      = test_ord,
  fit_fun = function(dat) {
    ordinalForest::ordfor(
      depvar       = "burden_score_ord",
      data         = dat,
      nsets        = best_of_nsets,
      ntreeperdiv  = best_of_ntreeperdiv,
      ntreefinal   = best_of_ntreefinal,
      mtry         = best_of_mtry,
      perffunction = "probability"
    )
  },
  pred_fun = function(model, newdat, outcome_levels) {
    pred <- predict(model, newdata = newdat)
    list(
      pred     = pred$ypred,
      prob_mat = align_prob_matrix(pred$classprobs, outcome_levels)
    )
  },
  metric_type = "prob"
)

final_test_svm <- run_final_test_model(
  model_name     = "Ordinal SVM",
  model_stage    = "Best_Tuned",
  parameter_id   = best_svm$Parameter_ID,
  parameter_text = best_svm$Parameter_Text,
  train_data     = train_plus_valid,
  test_data      = test_ord,
  fit_fun = function(dat) {
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
  },
  pred_fun = function(model, newdat, outcome_levels) {
    x_new    <- newdat %>% dplyr::select(-burden_score_ord)
    pred_tbl <- predict(model, new_data = x_new, type = "class")
    raw_tbl  <- predict(model, new_data = x_new, type = "raw")
    list(
      pred      = pred_tbl$.pred_class,
      score_vec = raw_tbl$.pred
    )
  },
  metric_type = "score"
)

final_test_knn <- run_final_test_model(
  model_name     = "Ordinal KNN",
  model_stage    = "Best_Tuned",
  parameter_id   = best_knn$Parameter_ID,
  parameter_text = best_knn$Parameter_Text,
  train_data     = train_plus_valid,
  test_data      = test_ord,
  fit_fun = function(dat) {
    kknn::train.kknn(
      burden_score_ord ~ .,
      data     = dat,
      kmax     = best_knn_kmax,
      distance = best_knn_distance,
      kernel   = best_knn_kernel,
      scale    = FALSE
    )
  },
  pred_fun = function(model, newdat, outcome_levels) {
    pred <- predict(model, newdata = newdat, type = "raw")
    prob <- predict(model, newdata = newdat, type = "prob")
    list(
      pred     = pred,
      prob_mat = align_prob_matrix(prob, outcome_levels)
    )
  },
  metric_type = "prob"
)

final_test_summary_best_tuned <- bind_rows(
  final_test_logit,
  final_test_forest,
  final_test_svm,
  final_test_knn
)

cat("Final test evaluation done.\n")
print(final_test_summary_best_tuned)

# Save all intermediate results for downstream scripts

saveRDS(
  list(
    baseline_results              = baseline_results,
    baseline_overfit_summary      = baseline_overfit_summary,
    logit_tuning_results          = logit_tuning_results,
    logit_tuning_summary          = logit_tuning_summary,
    logit_tuning_specs            = logit_tuning_specs,
    ordfor_tuning_results         = ordfor_tuning_results,
    svm_tuning_results            = svm_tuning_results,
    knn_tuning_results            = knn_tuning_results,
    ranked_tuning_summary         = ranked_tuning_summary,
    best_tuned_params             = best_tuned_params,
    validation_pre_post_wide      = validation_pre_post_wide,
    final_test_summary_best_tuned = final_test_summary_best_tuned,
    best_logit_formula            = best_logit_formula,
    best_logit_spec               = best_logit_spec,
    best_of_nsets                 = best_of_nsets,
    best_of_ntreeperdiv           = best_of_ntreeperdiv,
    best_of_ntreefinal            = best_of_ntreefinal,
    best_of_mtry                  = best_of_mtry,
    best_svm_cost                 = best_svm_cost,
    best_svm_sigma                = best_svm_sigma,
    best_knn_kmax                 = best_knn_kmax,
    best_knn_distance             = best_knn_distance,
    best_knn_kernel               = best_knn_kernel
  ),
  file = file.path(output_dir, "tuning_collect_results.rds")
)

cat("All results saved to:", file.path(output_dir, "tuning_collect_results.rds"), "\n")

# Save trained models (only runs when country_label == "Pool")

if (country_label %in% c("Pool", "Balanced_Pool_200")) {
  
  cat(country_label, "dataset detected. Saving trained models...\n")
  
  scaled_for_save      <- make_scaled_train_test(train_plus_valid, test_ord)
  train_final_for_save <- scaled_for_save$train_final
  
  trained_model_logit <- tryCatch({
    if (is.null(best_logit_formula)) stop("Logistic Regression formula not available.")
    MASS::polr(formula = best_logit_formula, data = train_final_for_save, Hess = TRUE)
  }, error = function(e) {
    message("Logistic Regression model saving failed: ", e$message)
    NULL
  })
  
  trained_model_forest <- ordinalForest::ordfor(
    depvar       = "burden_score_ord",
    data         = train_final_for_save,
    nsets        = best_of_nsets,
    ntreeperdiv  = best_of_ntreeperdiv,
    ntreefinal   = best_of_ntreefinal,
    mtry         = best_of_mtry,
    perffunction = "probability"
  )
  
  trained_model_svm <- mildsvm::svor_exc(
    burden_score_ord ~ .,
    data    = train_final_for_save,
    cost    = best_svm_cost,
    control = list(
      kernel    = "radial",
      sigma     = best_svm_sigma,
      scale     = FALSE,
      max_steps = 5000
    )
  )
  
  trained_model_knn <- kknn::train.kknn(
    burden_score_ord ~ .,
    data     = train_final_for_save,
    kmax     = best_knn_kmax,
    distance = best_knn_distance,
    kernel   = best_knn_kernel,
    scale    = FALSE
  )
  
  saveRDS(
    list(
      model_logit    = trained_model_logit,
      model_forest   = trained_model_forest,
      model_svm      = trained_model_svm,
      model_knn      = trained_model_knn,
      preProc        = scaled_for_save$preProc,
      outcome_levels = levels(train_final_for_save$burden_score_ord)
    ),
    file = file.path(output_dir, "trained_models.rds")
  )
  
  cat("Trained models saved to:", file.path(output_dir, "trained_models.rds"), "\n")
  
} else {
  cat("Skipping model saving (not Pool dataset).\n")
}
