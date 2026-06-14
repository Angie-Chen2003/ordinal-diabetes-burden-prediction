###Model Package Preparation
#Logistic Regression
library(MASS)
library(caret)
library(pROC)
train_polr = data.frame(
  burden_score_ord = y_train_ord,
  x_train_ord)

valid_polr = data.frame(
  burden_score_ord = y_valid_ord,
  x_valid_ord)

test_polr = data.frame(
  burden_score_ord = y_test_ord,
  x_test_ord)

#Ordinal Forest
library(ordinalForest)
library(caret)

# train / valid / test data
train_of = data.frame(
  burden_score_ord = y_train_ord,
  x_train_ord)

test_of = data.frame(
  burden_score_ord = y_test_ord,
  x_test_ord)

valid_of = data.frame(
  burden_score_ord = y_valid_ord,
  x_valid_ord)

train_of = as.data.frame(train_of)
valid_of  = as.data.frame(valid_of)
test_of  = as.data.frame(test_of)

#SVM
library(mildsvm)
library(caret)
library(pROC)

# build clean train/valid/test data
train_svm = data.frame(
  burden_score_ord = y_train_ord,
  x_train_ord)

valid_svm = data.frame(
  burden_score_ord = y_valid_ord,
  x_valid_ord)

test_svm = data.frame(
  burden_score_ord = y_test_ord,
  x_test_ord)

train_svm = as.data.frame(train_svm)
valid_svm = as.data.frame(valid_svm)
test_svm  = as.data.frame(test_svm)

predictor_cols = setdiff(
  names(train_svm),
  "burden_score_ord")

#knn
library(kknn)
library(caret)

# build clean train/valid/test data
train_knn = data.frame(
  burden_score_ord = y_train_ord,
  x_train_ord)

test_knn = data.frame(
  burden_score_ord = y_test_ord,
  x_test_ord)

valid_knn = data.frame(
  burden_score_ord = y_valid_ord,
  x_valid_ord)

train_knn = as.data.frame(train_knn)
test_knn  = as.data.frame(test_knn)
valid_knn  = as.data.frame(valid_knn)