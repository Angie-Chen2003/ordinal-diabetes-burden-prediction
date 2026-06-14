# Collect all bootstrap results, generate all final tables,
# and export to Excel.
# Run this script locally after all HPC bootstrap jobs finish.

source("config.R")
source("function.R")

# Load tuning results from tuning.R

tuning_results <- readRDS(file.path(output_dir, "tuning_collect_results.rds"))

baseline_results              <- tuning_results$baseline_results
baseline_overfit_summary      <- tuning_results$baseline_overfit_summary
logit_tuning_results          <- tuning_results$logit_tuning_results
logit_tuning_summary          <- tuning_results$logit_tuning_summary
logit_tuning_specs            <- tuning_results$logit_tuning_specs
ordfor_tuning_results         <- tuning_results$ordfor_tuning_results
svm_tuning_results            <- tuning_results$svm_tuning_results
knn_tuning_results            <- tuning_results$knn_tuning_results
ranked_tuning_summary         <- tuning_results$ranked_tuning_summary
best_tuned_params             <- tuning_results$best_tuned_params
validation_pre_post_wide      <- tuning_results$validation_pre_post_wide
final_test_summary_best_tuned <- tuning_results$final_test_summary_best_tuned
best_logit_formula            <- tuning_results$best_logit_formula
best_logit_spec               <- tuning_results$best_logit_spec
best_of_nsets                 <- tuning_results$best_of_nsets
best_of_ntreeperdiv           <- tuning_results$best_of_ntreeperdiv
best_of_ntreefinal            <- tuning_results$best_of_ntreefinal
best_of_mtry                  <- tuning_results$best_of_mtry
best_svm_cost                 <- tuning_results$best_svm_cost
best_svm_sigma                <- tuning_results$best_svm_sigma
best_knn_kmax                 <- tuning_results$best_knn_kmax
best_knn_distance             <- tuning_results$best_knn_distance
best_knn_kernel               <- tuning_results$best_knn_kernel

cat("Tuning results loaded.\n")

# Collect bootstrap CI results

cat("Collecting bootstrap CI results...\n")

boot_ci_dir   <- file.path(output_dir, "bootstrap_ci")
boot_ci_files <- list.files(boot_ci_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(boot_ci_files) == 0) {
  stop("No bootstrap CI files found in: ", boot_ci_dir)
}

cat("Found", length(boot_ci_files), "bootstrap CI replicates.\n")

boot_ci_all <- bind_rows(lapply(boot_ci_files, readRDS))

# Collect bootstrap importance results

cat("Collecting bootstrap importance results...\n")

boot_imp_dir   <- file.path(output_dir, "bootstrap_importance")
boot_imp_files <- list.files(boot_imp_dir, pattern = "\\.rds$", full.names = TRUE)

if (length(boot_imp_files) == 0) {
  stop("No bootstrap importance files found in: ", boot_imp_dir)
}

cat("Found", length(boot_imp_files), "bootstrap importance replicates.\n")

boot_imp_all <- bind_rows(lapply(boot_imp_files, readRDS))

# format CI text

format_ci <- function(est, lower, upper, digits = 3) {
  paste0(
    round(est, digits),
    " [",
    round(lower, digits),
    ", ",
    round(upper, digits),
    "]"
  )
}

metric_names_ci       <- c("Accuracy", "Macro_F1", "MAE", "MSE", "MZOE",
                           "Macro_MAE", "ORC", "GC", "ADC")
higher_better_metrics <- c("Accuracy", "Macro_F1", "ORC", "GC", "ADC")
lower_better_metrics  <- c("MAE", "MSE", "MZOE", "Macro_MAE")

# Compute CI from bootstrap distribution

compute_ci_wide <- function(boot_data, dataset_filter = NULL) {
  
  if (!is.null(dataset_filter)) {
    boot_data <- boot_data %>% filter(Dataset %in% dataset_filter)
  }
  
  # Observed estimates (Bootstrap_ID == any, just take mean across replicates
  # as the point estimate is already stored per replicate consistently)
  observed <- boot_data %>%
    group_by(Model, Dataset) %>%
    summarise(
      across(all_of(metric_names_ci), ~mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  # CI from bootstrap distribution
  ci_long <- boot_data %>%
    group_by(Model, Dataset) %>%
    summarise(
      across(
        all_of(metric_names_ci),
        list(
          CI_Lower = ~quantile(.x, 0.025, na.rm = TRUE),
          CI_Upper = ~quantile(.x, 0.975, na.rm = TRUE)
        ),
        .names = "{.col}__{.fn}"
      ),
      B_Successful = n(),
      .groups = "drop"
    )
  
  # Join observed and CI
  result <- observed %>%
    left_join(ci_long, by = c("Model", "Dataset")) %>%
    mutate(Country = country_label)
  
  # Add CI text columns
  for (m in metric_names_ci) {
    result[[paste0(m, "_CI")]] <- format_ci(
      result[[m]],
      result[[paste0(m, "__CI_Lower")]],
      result[[paste0(m, "__CI_Upper")]]
    )
  }
  
  result
}

ci_train_valid <- compute_ci_wide(boot_ci_all, dataset_filter = c("Train", "Validation"))
ci_test        <- compute_ci_wide(boot_ci_all, dataset_filter = "Test")

cat("Bootstrap CI computed.\n")

# Compute importance CI from bootstrap distribution

# Observed importance (mean across replicates per predictor per model)
observed_importance <- boot_imp_all %>%
  group_by(Country, Model, Predictor, Index) %>%
  summarise(
    Importance        = mean(Importance, na.rm = TRUE),
    Base_ORC          = mean(Base_ORC, na.rm = TRUE),
    Permuted_ORC_Mean = mean(Permuted_ORC_Mean, na.rm = TRUE),
    N_Repeats         = mean(N_Repeats, na.rm = TRUE),
    B_Successful      = n(),
    .groups = "drop"
  )

importance_ci <- boot_imp_all %>%
  group_by(Model, Predictor) %>%
  summarise(
    Importance_CI_Lower       = quantile(Importance, 0.025, na.rm = TRUE),
    Importance_CI_Upper       = quantile(Importance, 0.975, na.rm = TRUE),
    Importance_Bootstrap_Mean = mean(Importance, na.rm = TRUE),
    Importance_Bootstrap_SD   = sd(Importance, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(observed_importance, by = c("Model", "Predictor")) %>%
  mutate(
    Importance_CI = format_ci(
      Importance,
      Importance_CI_Lower,
      Importance_CI_Upper,
      digits = 4
    )
  ) %>%
  arrange(
    factor(Model, levels = c(
      "Ordinal Logistic Regression", "Ordinal Forest",
      "Ordinal SVM", "Ordinal KNN"
    )),
    desc(Importance)
  ) %>%
  group_by(Model) %>%
  mutate(Rank = row_number()) %>%
  ungroup() %>%
  dplyr::select(
    Country, Model, Predictor, Index,
    Importance, Importance_CI, Importance_CI_Lower, Importance_CI_Upper,
    Importance_Bootstrap_Mean, Importance_Bootstrap_SD,
    Base_ORC, Permuted_ORC_Mean, N_Repeats, B_Successful, Rank
  )

cat("Importance CI computed.\n")

#Train / Validation Table

model_order <- c(
  "Ordinal Logistic Regression", "Ordinal Forest",
  "Ordinal SVM", "Ordinal KNN"
)

select_ci_cols <- function(ci_data) {
  ci_data %>%
    dplyr::select(
      Country, Model, Dataset,
      all_of(metric_names_ci),
      paste0(metric_names_ci, "_CI"),
      paste0(metric_names_ci, "__CI_Lower"),
      paste0(metric_names_ci, "__CI_Upper")
    ) %>%
    rename_with(
      ~ gsub("__CI_Lower", "_CI_Lower", .x),
      ends_with("__CI_Lower")
    ) %>%
    rename_with(
      ~ gsub("__CI_Upper", "_CI_Upper", .x),
      ends_with("__CI_Upper")
    )
}

table_01_train_validation_with_CI <- select_ci_cols(ci_train_valid) %>%
  arrange(
    factor(Model, levels = model_order),
    factor(Dataset, levels = c("Train", "Validation"))
  )

# Test performance with bootstrap CI

table_02_test_performance_with_CI <- select_ci_cols(ci_test) %>%
  arrange(factor(Model, levels = model_order))

# Parameter table

parameter_for_model_fill <- best_tuned_params %>%
  mutate(
    Selected_Parameter = case_when(
      Model == "Ordinal Logistic Regression" ~ Parameter_Text,
      Model == "Ordinal Forest" ~ paste0(
        "nsets=", best_of_nsets,
        "; ntreeperdiv=", best_of_ntreeperdiv,
        "; ntreefinal=", best_of_ntreefinal,
        "; mtry=", best_of_mtry
      ),
      Model == "Ordinal SVM" ~ paste0(
        "cost=", best_svm_cost,
        "; sigma=", best_svm_sigma,
        "; kernel=radial"
      ),
      Model == "Ordinal KNN" ~ paste0(
        "kmax=", best_knn_kmax,
        "; distance=", best_knn_distance,
        "; kernel=", best_knn_kernel
      ),
      TRUE ~ Parameter_Text
    )
  ) %>%
  dplyr::select(
    Model, Parameter_ID, Selected_Parameter,
    Overall_Rank_Score, Overfitting_Status,
    Accuracy_Validation, Macro_F1_Validation,
    MAE_Validation, MSE_Validation, MZOE_Validation,
    Macro_MAE_Validation, ORC_Validation, GC_Validation, ADC_Validation
  ) %>%
  arrange(factor(Model, levels = model_order))

# Add validation CI to parameter table
validation_ci_for_parameter <- ci_train_valid %>%
  filter(Dataset == "Validation") %>%
  dplyr::select(
    Model,
    paste0(metric_names_ci, "_CI"),
    paste0(metric_names_ci, "__CI_Lower"),
    paste0(metric_names_ci, "__CI_Upper")
  ) %>%
  rename_with(~ gsub("__CI_Lower", "_CI_Lower", .x), ends_with("__CI_Lower")) %>%
  rename_with(~ gsub("__CI_Upper", "_CI_Upper", .x), ends_with("__CI_Upper"))

table_03_parameter_with_CI <- parameter_for_model_fill %>%
  left_join(validation_ci_for_parameter, by = "Model")

# CV robustness (mean and SD across 5 folds)

cat("Running 5-fold CV for robustness table...\n")

set.seed(123)
cv_folds_final <- caret::createFolds(
  train_plus_valid$burden_score_ord,
  k = 5,
  returnTrain = FALSE
)

run_cv_fixed_model <- function(model_name, train_data, folds,
                               fit_fun, pred_fun,
                               metric_type = c("prob", "score")) {
  metric_type    <- match.arg(metric_type)
  cv_results     <- list()
  outcome_levels <- levels(train_data$burden_score_ord)
  
  for (i in seq_along(folds)) {
    valid_idx  <- folds[[i]]
    train_idx  <- setdiff(seq_len(nrow(train_data)), valid_idx)
    fold_train <- train_data[train_idx, ]
    fold_valid <- train_data[valid_idx, ]
    
    scaled_dat  <- make_scaled_train_valid(fold_train, fold_valid)
    train_final <- scaled_dat$train_final
    valid_final <- scaled_dat$valid_final
    
    y_valid <- factor(valid_final$burden_score_ord,
                      levels = outcome_levels, ordered = TRUE)
    
    model    <- fit_fun(train_final)
    pred_out <- pred_fun(model, valid_final, outcome_levels)
    
    pred_class <- factor(pred_out$pred, levels = outcome_levels, ordered = TRUE)
    cm         <- caret::confusionMatrix(pred_class, y_valid)
    
    if (metric_type == "prob") {
      ordinal_metrics <- calc_ordinal_metrics(
        y_true_ord = y_valid,
        prob_mat   = pred_out$prob_mat
      )
    } else {
      ordinal_metrics <- calc_ordinal_metrics_from_score(
        y_true_ord = y_valid,
        score_vec  = pred_out$score_vec
      )
    }
    
    cv_results[[i]] <- eval_ordinal_model(
      model_name      = model_name,
      y_true          = y_valid,
      y_pred          = pred_class,
      cm_obj          = cm,
      ordinal_metrics = ordinal_metrics
    ) %>%
      mutate(Fold = i)
  }
  
  bind_rows(cv_results)
}

cv_logit <- tryCatch({
  if (is.null(best_logit_formula)) stop("Logistic Regression formula not available.")
  run_cv_fixed_model(
    model_name  = "Ordinal Logistic Regression",
    train_data  = train_plus_valid,
    folds       = cv_folds_final,
    fit_fun     = function(dat) MASS::polr(formula = best_logit_formula, data = dat, Hess = TRUE),
    pred_fun    = function(model, newdat, outcome_levels) {
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
  message("Logistic Regression CV failed: ", e$message)
  NULL
})


cv_forest <- run_cv_fixed_model(
  model_name  = "Ordinal Forest",
  train_data  = train_plus_valid,
  folds       = cv_folds_final,
  fit_fun     = function(dat) {
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
  pred_fun    = function(model, newdat, outcome_levels) {
    pred <- predict(model, newdata = newdat)
    list(
      pred     = pred$ypred,
      prob_mat = align_prob_matrix(pred$classprobs, outcome_levels)
    )
  },
  metric_type = "prob"
)

cv_svm <- run_cv_fixed_model(
  model_name  = "Ordinal SVM",
  train_data  = train_plus_valid,
  folds       = cv_folds_final,
  fit_fun     = function(dat) {
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
  pred_fun    = function(model, newdat, outcome_levels) {
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

cv_knn <- run_cv_fixed_model(
  model_name  = "Ordinal KNN",
  train_data  = train_plus_valid,
  folds       = cv_folds_final,
  fit_fun     = function(dat) {
    kknn::train.kknn(
      burden_score_ord ~ .,
      data     = dat,
      kmax     = best_knn_kmax,
      distance = best_knn_distance,
      kernel   = best_knn_kernel,
      scale    = FALSE
    )
  },
  pred_fun    = function(model, newdat, outcome_levels) {
    pred <- predict(model, newdata = newdat, type = "raw")
    prob <- predict(model, newdata = newdat, type = "prob")
    list(
      pred     = pred,
      prob_mat = align_prob_matrix(prob, outcome_levels)
    )
  },
  metric_type = "prob"
)

cv_all_results <- bind_rows(cv_logit, cv_forest, cv_svm, cv_knn)

table_04_cv_robustness <- cv_all_results %>%
  group_by(Model) %>%
  summarise(
    across(
      all_of(metric_names_ci),
      list(
        mean = ~round(mean(.x, na.rm = TRUE), 3),
        sd   = ~round(sd(.x, na.rm = TRUE), 3)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  ) %>%
  arrange(factor(Model, levels = model_order))

cat("CV robustness table done.\n")

# Variable importance with CI

table_05_variable_importance_with_CI <- importance_ci

# Original tables without CI (for reference)

table_1_train_validation_no_ci <- bind_rows(
  logit_tuning_results,
  ordfor_tuning_results,
  svm_tuning_results,
  knn_tuning_results
) %>%
  semi_join(
    best_tuned_params %>% dplyr::select(Model, Parameter_ID),
    by = c("Model", "Parameter_ID")
  ) %>%
  dplyr::select(
    Model, Dataset,
    Accuracy, Macro_F1, MAE, MSE, MZOE, Macro_MAE, ORC, GC, ADC
  ) %>%
  arrange(
    factor(Model, levels = model_order),
    factor(Dataset, levels = c("Train", "Validation"))
  )

table_2_test_no_ci <- final_test_summary_best_tuned %>%
  mutate(Country = country_label) %>%
  dplyr::select(
    Country, Model,
    Accuracy, Macro_F1, MAE, MSE, MZOE, Macro_MAE, ORC, GC, ADC
  ) %>%
  arrange(factor(Model, levels = model_order))

# Export all tables to Excel

output_path <- file.path(
  output_dir,
  paste0(country_label, "_all_final_tables.xlsx")
)

writexl::write_xlsx(
  list(
    "01_Train_Validation_With_CI"    = table_01_train_validation_with_CI,
    "02_Test_Performance_With_CI"    = table_02_test_performance_with_CI,
    "03_Parameter_With_CI"           = table_03_parameter_with_CI,
    "04_CV_Robustness"               = table_04_cv_robustness,
    "05_Variable_Importance_With_CI" = table_05_variable_importance_with_CI,
    "Validation_Before_After"        = validation_pre_post_wide,
    "Ranked_Tuning_Summary"          = ranked_tuning_summary,
    "Best_Tuned_Params_Raw"          = best_tuned_params,
    "CV_Fold_Level_Results"          = cv_all_results,
    "Original_Train_Valid_No_CI"     = table_1_train_validation_no_ci,
    "Original_Test_No_CI"            = table_2_test_no_ci,
    "Original_Parameter_Table"       = parameter_for_model_fill,
    "Bootstrap_CI_Raw"               = boot_ci_all,
    "Bootstrap_Importance_Raw"       = boot_imp_all
  ),
  path = output_path
)

cat("All tables exported to:", output_path, "\n")