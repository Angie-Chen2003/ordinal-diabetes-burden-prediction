###Data Preprocessing for Model
library(caret)
library(dplyr)
library(readxl)

data_model = read_xlsx("~/Data_final.xlsx")

# If pain and vision exist, construct burden_score from them (single country datasets)
# If burden_score_ord already exists (Pool dataset), skip construction
if ("pain" %in% names(data_model) && "vision" %in% names(data_model)) {
  
  data_model <- data_model %>%
    mutate(
      pain   = as.numeric(as.character(pain)),
      vision = as.numeric(as.character(vision))
    ) %>%
    filter(!is.na(pain), !is.na(vision)) %>%
    mutate(burden_score = pain + vision)
  
  # Keep only valid ordinal categories
  data_model <- data_model %>%
    filter(burden_score %in% c(0, 1, 2))
  
  # Ordered factor outcome
  data_model$burden_score_ord <- factor(
    data_model$burden_score,
    levels = c(0, 1, 2),
    ordered = TRUE
  )
  
  # Integer version, useful for some ordinal methods
  data_model$burden_score_int <- as.integer(data_model$burden_score_ord) - 1
  
  # Remove outcome components to avoid leakage
  data_model <- data_model %>%
    dplyr::select(-pain, -vision)
  
} else {
  
  # Pool dataset: outcome already constructed, just ensure correct types
  data_model <- charls_model %>%
    filter(burden_score %in% c(0, 1, 2))
  
  data_model$burden_score_ord <- factor(
    data_model$burden_score_ord,
    levels = c(0, 1, 2),
    ordered = TRUE
  )
  
  data_model$burden_score_int <- as.integer(data_model$burden_score_ord) - 1
  
  # Rename country to country_factor if needed (Balanced Pool uses 'country')
  if ("country" %in% names(data_model) && !"country_factor" %in% names(data_model)) {
    data_model <- data_model %>% rename(country_factor = country)
  }
  
}

#Train / Validation / Test split

set.seed(123)
trainIndex <- createDataPartition(
  data_model$burden_score_ord,
  p = 0.8,
  list = FALSE
)

train_ord_full <- data_model[trainIndex, ]
test_ord       <- data_model[-trainIndex, ]

set.seed(123)
validIndex <- createDataPartition(
  train_ord_full$burden_score_ord,
  p = 0.8,
  list = FALSE
)

train_ord        <- train_ord_full[validIndex, ]
valid_ord        <- train_ord_full[-validIndex, ]
train_plus_valid <- bind_rows(train_ord, valid_ord)

# get predictor columns

outcome_cols <- c("burden_score", "burden_score_ord", "burden_score_int")

get_predictor_cols <- function(dat) {
  setdiff(names(dat), outcome_cols)
}

# standardization

make_scaled_train_valid <- function(train_data, valid_data) {
  x_cols <- get_predictor_cols(train_data)
  
  x_train <- train_data[, x_cols, drop = FALSE]
  x_valid <- valid_data[, x_cols, drop = FALSE]
  
  y_train <- train_data$burden_score_ord
  y_valid <- valid_data$burden_score_ord
  
  preProc <- caret::preProcess(x_train, method = c("center", "scale"))
  
  x_train_scaled <- predict(preProc, x_train)
  x_valid_scaled <- predict(preProc, x_valid)
  
  train_final <- data.frame(burden_score_ord = y_train, x_train_scaled)
  valid_final <- data.frame(burden_score_ord = y_valid, x_valid_scaled)
  
  return(list(
    train_final    = train_final,
    valid_final    = valid_final,
    preProc        = preProc,
    predictor_cols = names(x_train_scaled)
  ))
}

make_scaled_train_test <- function(train_data, test_data) {
  x_cols <- get_predictor_cols(train_data)
  
  x_train <- train_data[, x_cols, drop = FALSE]
  x_test  <- test_data[, x_cols, drop = FALSE]
  
  y_train <- train_data$burden_score_ord
  y_test  <- test_data$burden_score_ord
  
  preProc <- caret::preProcess(x_train, method = c("center", "scale"))
  
  x_train_scaled <- predict(preProc, x_train)
  x_test_scaled  <- predict(preProc, x_test)
  
  train_final <- data.frame(burden_score_ord = y_train, x_train_scaled)
  test_final  <- data.frame(burden_score_ord = y_test,  x_test_scaled)
  
  return(list(
    train_final    = train_final,
    test_final     = test_final,
    preProc        = preProc,
    predictor_cols = names(x_train_scaled)
  ))
}
