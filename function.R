# align probability matrix

align_prob_matrix <- function(prob_mat, outcome_levels) {
  prob_mat <- as.matrix(prob_mat)
  
  if (!is.null(colnames(prob_mat))) {
    prob_mat <- prob_mat[, outcome_levels, drop = FALSE]
  } else {
    if (ncol(prob_mat) == length(outcome_levels)) {
      colnames(prob_mat) <- outcome_levels
    } else {
      stop("Probability matrix columns do not match outcome levels.")
    }
  }
  
  return(prob_mat)
}

# Ordinal metrics from probability matrix

calc_ordinal_metrics <- function(y_true_ord, prob_mat) {
  if (!is.ordered(y_true_ord)) warning("y_true_ord is not an ordered factor.")
  if (!is.matrix(prob_mat)) prob_mat <- as.matrix(prob_mat)
  if (ncol(prob_mat) != length(levels(y_true_ord))) {
    stop("Number of columns in prob_mat must match number of outcome levels.")
  }
  
  y     <- as.numeric(y_true_ord)
  prob  <- prob_mat
  K     <- ncol(prob)
  score <- as.numeric(prob %*% (1:K))
  
  pairwise_c <- matrix(NA, K, K)
  for (a in 1:(K - 1)) {
    for (b in (a + 1):K) {
      idx <- which(y %in% c(a, b))
      if (length(idx) > 1 && sum(y[idx] == a) > 0 && sum(y[idx] == b) > 0) {
        y_ab    <- ifelse(y[idx] == b, 1, 0)
        roc_obj <- roc(response = y_ab, predictor = score[idx], quiet = TRUE)
        pairwise_c[a, b] <- as.numeric(auc(roc_obj))
      }
    }
  }
  
  pair_vals        <- pairwise_c[upper.tri(pairwise_c)]
  pair_vals_non_na <- pair_vals[!is.na(pair_vals)]
  ORC <- ifelse(length(pair_vals_non_na) > 0, mean(pair_vals_non_na, na.rm = TRUE), NA)
  
  tab <- table(factor(y, levels = 1:K))
  num <- 0; den <- 0
  for (a in 1:(K - 1)) {
    for (b in (a + 1):K) {
      if (!is.na(pairwise_c[a, b])) {
        w_ab <- as.numeric(tab[a] * tab[b])
        num  <- num + w_ab * pairwise_c[a, b]
        den  <- den + w_ab
      }
    }
  }
  GC <- ifelse(den > 0, num / den, NA)
  
  adc_vals <- c()
  for (j in 1:(K - 1)) {
    y_bin <- ifelse(y > j, 1, 0)
    p_bin <- rowSums(prob[, (j + 1):K, drop = FALSE])
    if (length(unique(y_bin)) == 2) {
      roc_obj  <- roc(response = y_bin, predictor = p_bin, quiet = TRUE)
      adc_vals <- c(adc_vals, as.numeric(auc(roc_obj)))
    }
  }
  ADC <- ifelse(length(adc_vals) > 0, mean(adc_vals, na.rm = TRUE), NA)
  
  return(list(ORC = ORC, GC = GC, ADC = ADC,
              pairwise_c = pairwise_c,
              ADC_each_cutpoint = adc_vals,
              score = score))
}

# Ordinal metrics from continuous score (for SVM)

calc_ordinal_metrics_from_score <- function(y_true_ord, score_vec) {
  if (!is.ordered(y_true_ord)) warning("y_true_ord is not an ordered factor.")
  
  y     <- as.numeric(y_true_ord)
  score <- as.numeric(score_vec)
  K     <- nlevels(y_true_ord)
  
  if (length(score) != length(y)) stop("score_vec and y_true_ord must have the same length.")
  
  pairwise_c <- matrix(NA, K, K)
  for (a in 1:(K - 1)) {
    for (b in (a + 1):K) {
      idx <- which(y %in% c(a, b))
      if (length(idx) > 1 && sum(y[idx] == a) > 0 && sum(y[idx] == b) > 0) {
        y_ab    <- ifelse(y[idx] == b, 1, 0)
        roc_obj <- roc(response = y_ab, predictor = score[idx], quiet = TRUE)
        pairwise_c[a, b] <- as.numeric(auc(roc_obj))
      }
    }
  }
  
  pair_vals        <- pairwise_c[upper.tri(pairwise_c)]
  pair_vals_non_na <- pair_vals[!is.na(pair_vals)]
  ORC <- ifelse(length(pair_vals_non_na) > 0, mean(pair_vals_non_na, na.rm = TRUE), NA)
  
  tab <- table(factor(y, levels = 1:K))
  num <- 0; den <- 0
  for (a in 1:(K - 1)) {
    for (b in (a + 1):K) {
      if (!is.na(pairwise_c[a, b])) {
        w_ab <- as.numeric(tab[a] * tab[b])
        num  <- num + w_ab * pairwise_c[a, b]
        den  <- den + w_ab
      }
    }
  }
  GC <- ifelse(den > 0, num / den, NA)
  
  adc_vals <- c()
  for (j in 1:(K - 1)) {
    y_bin <- ifelse(y > j, 1, 0)
    if (length(unique(y_bin)) == 2) {
      roc_obj  <- roc(response = y_bin, predictor = score, quiet = TRUE)
      adc_vals <- c(adc_vals, as.numeric(auc(roc_obj)))
    }
  }
  ADC <- ifelse(length(adc_vals) > 0, mean(adc_vals, na.rm = TRUE), NA)
  
  return(list(ORC = ORC, GC = GC, ADC = ADC,
              pairwise_c = pairwise_c,
              ADC_each_cutpoint = adc_vals,
              score = score))
}

# Unified evaluation function

eval_ordinal_model <- function(model_name, y_true, y_pred, cm_obj, ordinal_metrics) {
  y_true_fac <- factor(y_true, levels = levels(y_true), ordered = TRUE)
  y_pred_fac <- factor(y_pred, levels = levels(y_true), ordered = TRUE)
  
  y_true_num <- as.numeric(y_true_fac)
  y_pred_num <- as.numeric(y_pred_fac)
  
  MAE  <- mean(abs(y_true_num - y_pred_num), na.rm = TRUE)
  MSE  <- mean((y_true_num - y_pred_num)^2, na.rm = TRUE)
  MZOE <- mean(y_true_num != y_pred_num, na.rm = TRUE)
  
  classes <- levels(y_true_fac)
  MAE_by_class <- sapply(classes, function(cls) {
    idx <- y_true_fac == cls
    mean(abs(y_true_num[idx] - y_pred_num[idx]), na.rm = TRUE)
  })
  Macro_MAE <- mean(MAE_by_class, na.rm = TRUE)
  
  if (is.matrix(cm_obj$byClass)) {
    Macro_F1 <- mean(cm_obj$byClass[, "F1"], na.rm = TRUE)
  } else {
    Macro_F1 <- cm_obj$byClass["F1"]
  }
  
  data.frame(
    Model     = model_name,
    Accuracy  = round(as.numeric(cm_obj$overall["Accuracy"]), 3),
    Macro_F1  = round(as.numeric(Macro_F1), 3),
    MAE       = round(MAE, 3),
    MSE       = round(MSE, 3),
    MZOE      = round(MZOE, 3),
    Macro_MAE = round(Macro_MAE, 3),
    ORC       = round(ordinal_metrics$ORC, 3),
    GC        = round(ordinal_metrics$GC, 3),
    ADC       = round(ordinal_metrics$ADC, 3)
  )
}

# Model runner: train + validation evaluation

run_train_valid_model <- function(model_name, model_stage, parameter_id,
                                  parameter_text, train_data, valid_data,
                                  fit_fun, pred_fun,
                                  metric_type = c("prob", "score")) {
  metric_type <- match.arg(metric_type)
  
  scaled_dat  <- make_scaled_train_valid(train_data, valid_data)
  train_final <- scaled_dat$train_final
  valid_final <- scaled_dat$valid_final
  
  y_train        <- factor(train_final$burden_score_ord, ordered = TRUE)
  y_valid        <- factor(valid_final$burden_score_ord, levels = levels(y_train), ordered = TRUE)
  outcome_levels <- levels(y_train)
  
  model <- fit_fun(train_final)
  
  pred_train_out <- pred_fun(model, train_final, outcome_levels)
  pred_valid_out <- pred_fun(model, valid_final, outcome_levels)
  
  pred_train <- factor(pred_train_out$pred, levels = outcome_levels, ordered = TRUE)
  pred_valid <- factor(pred_valid_out$pred, levels = outcome_levels, ordered = TRUE)
  
  cm_train <- caret::confusionMatrix(pred_train, y_train)
  cm_valid <- caret::confusionMatrix(pred_valid, y_valid)
  
  if (metric_type == "prob") {
    ordinal_train <- calc_ordinal_metrics(y_train, pred_train_out$prob_mat)
    ordinal_valid <- calc_ordinal_metrics(y_valid, pred_valid_out$prob_mat)
  } else {
    ordinal_train <- calc_ordinal_metrics_from_score(y_train, pred_train_out$score_vec)
    ordinal_valid <- calc_ordinal_metrics_from_score(y_valid, pred_valid_out$score_vec)
  }
  
  train_eval <- eval_ordinal_model(model_name, y_train, pred_train, cm_train, ordinal_train) %>%
    mutate(Dataset = "Train")
  valid_eval <- eval_ordinal_model(model_name, y_valid, pred_valid, cm_valid, ordinal_valid) %>%
    mutate(Dataset = "Validation")
  
  bind_rows(train_eval, valid_eval) %>%
    mutate(
      Model_Stage    = model_stage,
      Parameter_ID   = parameter_id,
      Parameter_Text = parameter_text
    ) %>%
    dplyr::select(Model, Dataset, Accuracy, Macro_F1, MAE, MSE, MZOE, Macro_MAE,
                  ORC, GC, ADC, Model_Stage, Parameter_ID, Parameter_Text)
}

# Overfitting summary

summarize_overfitting <- function(overfit_results) {
  overfit_results %>%
    dplyr::select(Model, Model_Stage, Parameter_ID, Parameter_Text, Dataset,
                  Accuracy, Macro_F1, MAE, MSE, MZOE, Macro_MAE, ORC, GC, ADC) %>%
    tidyr::pivot_wider(
      names_from  = Dataset,
      values_from = c(Accuracy, Macro_F1, MAE, MSE, MZOE, Macro_MAE, ORC, GC, ADC)
    ) %>%
    mutate(
      Accuracy_Gap = Accuracy_Train - Accuracy_Validation,
      F1_Gap       = Macro_F1_Train - Macro_F1_Validation,
      ORC_Gap      = ORC_Train      - ORC_Validation,
      GC_Gap       = GC_Train       - GC_Validation,
      ADC_Gap      = ADC_Train      - ADC_Validation,
      Overfitting_Status = case_when(
        Accuracy_Gap > 0.10 | F1_Gap > 0.10 | ORC_Gap > 0.10 ~ "Potential severe overfitting",
        Accuracy_Gap > 0.05 | F1_Gap > 0.05 | ORC_Gap > 0.05 ~ "Possible mild overfitting",
        TRUE ~ "No severe overfitting"
      )
    )
}


#Create output directory if it does not exist

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("00_functions_data.R loaded. Country:", country_label, "\n")
cat("Data dimensions:", nrow(charls_model), "rows x", ncol(charls_model), "cols\n")
cat("Outcome distribution:\n")
print(table(charls_model$burden_score_ord))
