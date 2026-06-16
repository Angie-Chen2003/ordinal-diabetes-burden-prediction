# ordinal-diabetes-burden-prediction
Reproducible analytical pipeline for comparing ordinal statistical and machine learning models to predict diabetes-related complication burden across international aging cohorts.

--- 
## Overview 

This repository contains the complete analytical pipeline used to evaluate and compare ordinal statistical and machine learning models for predicting diabetes-related complication burden among older adults with diabetes across multiple international aging cohorts. 
The framework was designed to investigate: 
- Cross-country heterogeneity in diabetes-related burden;
- Generalizability of prediction models across countries;
- Differences between country-specific and pooled training strategies;
- Predictor importance across populations;
- Reproducible implementation of ordinal machine learning pipelines.

---

## Study Population

Eight nationally representative aging cohorts were included:

| Dataset | Country/Region |
|----------|----------------|
| [HRS](https://hrsdata.isr.umich.edu/data-products/public-survey-data) | United States |
| [MHAS](https://www.mhasweb.org/DataProducts/Home.aspx) | Mexico |
| [LASI](https://www.iipsindia.ac.in/content/LASI-data) | India |
| [CHARLS](https://charls.charlsdata.com/pages/data/111/en.html) | China |
| [SHARE](https://share-eric.eu/data/become-a-user) | Europe |
| [ELSA](https://datacatalogue.ukdataservice.ac.uk/series/series/200011#abstract) | United Kingdom |
| [KLoSA](https://survey.keis.or.kr/eng/klosa/klosa01.jsp) | South Korea |
| [HAALSI](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TW84UI) | South Africa |

Participants were eligible if they:

- were aged ≥ 55 years;
- reported physician-diagnosed diabetes.

---

## Repository Structure


```text
project/
├── data/
│ ├── raw/
│ ├── harmonized/
│ └── processed/
│
├── R/
│ ├── harmonization.R
│ ├── descriptive_analysis.R
│ ├── imputation.R
│ ├── preprocessing.R
│ ├── model_preparing.R
│ ├── function.R
│ ├── tuning.R
│ ├── model_ci.R
│ ├── importance_ci.R
│ └── results_summary.R
│
├── python/
│ ├── metrics.py
│ ├── utils.py
│ ├── model.py
│ ├── cv.py
│ ├── config.py
│ ├── tuning.py
│ ├── uncertainty.py
│ └── run_all_results.py
│
├── outputs/
│ ├── descriptive_analysis.xlsx/
│ ├── country_performance.xlsx/
│ ├── pool_performance.xlsx/
│ ├── importance.zip/
│ ├── cv.xlsx/
│ ├── diabetes_distribution.xlsx/
│ ├── parameters.xlsx/
│ └── tunning_validation.xlsx/
│
├── figures/
│ ├── importance.png/
│ └── workflow.png
│
└── README.md
```
---
## Key R Scripts

| File | Description |
|--------|-------------|
| [harmonization.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/4eeeafd00f3bd404894ea42e14e8f619f54d86bc/harmonization.R) | Variable harmonization across cohorts |
| [descriptive_analysis.R]([R/Descriptive.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/Descriptive.R)) | Generate Table 1 descriptive statistics |
| [imputation.R]([R/imputation.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/imputation.R)) | Missing data handling |
| [preprocessing.R]([R/preprocessing.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/preprocessing.R)) | Outcome construction and data preprocessing |
| [model_preparing.R]([R/model_prepare.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/model_prepare.R)) | Prepare datasets for modeling |
| [function.R]([R/function.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/function.R)) | Utility functions and evaluation metrics |
| [tuning.R]([R/tuning.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/tuning.R)) | Hyperparameter tuning |
| [model_ci.R]([R/metric_ci.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/metric_CI.R)) | Bootstrap confidence intervals for model performance |
| [importance_ci.zip](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/importance.zip) | Bootstrap confidence intervals for variable importance |
| [results_summary.R]([R/Results_summary.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/Results_summary.R)) | Summarize final results |

---
## Key Python Scripts

| File | Description |
|--------|-------------|
| [metrics.py]([python/metrics.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/metrics.py)) | Performance metric calculation |
| [utils.py]([python/data_utils.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/data_utils.py)) | Helper functions |
| [model.py]([python/models.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/models.py)) | Ordinal XGBoost and CORAL models |
| [cv.py]([python/cv_robustness.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/cv_robustness.py)) | Cross-validation procedures |
| [config.py]([python/config.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/config.py)) | Model configurations |
| [tuning.py]([python/tuning.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/tuning.py)) | Hyperparameter tuning |
| [uncertainty.py]([python/uncertainty.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/uncertainty.py)) | Bootstrap uncertainty estimation |
| [run_all_results.py]([python/run_pipeline.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/run_pipeline.py)) | Run complete analytical pipeline |

---
## Analytical Workflow

### Step 1. Download Raw Data

Download original datasets from the official cohort websites and obtain all required permissions.

---

### Step 2. Data Harmonization

Variables were harmonized based on cohort-specific codebooks.

Procedures included:

- selecting corresponding variables;
- standardizing coding schemes;
- converting "Don't Know" and "Refused" responses into missing values.

---

### Step 3. Study Sample Selection

Participants were restricted to:

- Age ≥ 55 years;
- Diabetes = Yes.

---

### Step 4. Descriptive Analysis

Descriptive statistics were generated separately for each country.

Categorical variables:

- Frequency (n);
- Percentage (%).

Continuous variables:

- Mean;
- Standard deviation (SD);
- Median;
- Interquartile range (IQR).
[Get Descriptive Analysis](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Table%201.xlsx)
[Diabetes Distribution Results](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Diabetes%20Distribution%20for%20Each%20Country.xlsx)
---

### Step 5. Variable Transformation

Examples include:

- Oral diabetes medication + insulin → Diabetes treatment;
- Social welfare + pension → Socioeconomic status;
- Conversion of categorical variables into binary indicators when appropriate.

**Note:** In country-specific analyses, SES was calculated using each country's original pension and social welfare income variables. In pooled analyses, these measures were converted to a PPP-adjusted indicator (`social_eco_ppp`) to facilitate meaningful comparisons across countries with different currencies and economic contexts.
[Website](https://data.worldbank.org/indicator/PA.NUS.PPP)
[Code to Harmonization](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/67b1bd251fb276814f55c8ad587a982557704230/GDP_PPP.rmd)

---

### Step 6. Missing Data Handling

Procedures included:

1. Excluding observations with missing outcome information;
2. Evaluating missingness patterns;
3. Performing imputation for predictors when appropriate.

> **Note:** Most cohorts used multiple imputation by chained equations (MICE) for predictor imputation. In contrast, for the SHARE dataset, categorical variables were imputed using the mode and continuous variables using the mean due to cohort-specific data characteristics and preprocessing requirements.

---

### Step 7. Outcome Construction and Data Splitting

The ordinal outcome was derived from:

- Vision impairment;
- Pain severity.

These variables were combined into a burden score.

Data splitting:

- Training set: 80%
- Test set: 20%

Within the training set:

- Validation subsets were used for hyperparameter tuning;
- Five-fold cross-validation was used for robustness assessment.

The test set remained untouched until final evaluation.

---

### Step 8. Evaluation Functions

Functions were implemented to calculate:

#### Ordinal discrimination metrics

- ORC (Ordinal C-index)
- GC (Generalized C-index)
- ADC (Average Dichotomous C-index)

#### Classification metrics

- Accuracy
- Macro-F1

#### Error metrics

- MAE
- MSE
- MZOE
- Macro-MAE

Additional functions included:

- Overfitting assessment;
- Train–validation gap calculation;
- Cross-validation evaluation.

---

## Models Evaluated

Six ordinal prediction models were compared.

| Model | Software |
|---------|----------|
| Ordinal Logistic Regression | R |
| Ordinal Forest | R |
| K-Nearest Neighbors (KNN) | R |
| Support Vector Machine (SVM) | R |
| Ordinal XGBoost | Python |
| CORAL | Python |

---

## Hyperparameter Tuning

Hyperparameters were selected using the training data only.

Procedure:

1. Train model on training subset;
2. Evaluate on validation subset;
3. Select optimal hyperparameters;
4. Assess train–validation gap.

The test set was never used during tuning.
[Get Tuning Results](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Train_Validation.xlsx)
[Get Parameters](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/c83d36f61adbfe913a4dbb84eb38691b18f6aa9b/Paramenters.zip)

---

## Final Model Evaluation

After hyperparameter selection:

1. Refit the model using the full training set;
2. Evaluate once using the held-out test set.

Performance metrics included:

- Accuracy;
- Macro-F1;
- ORC;
- GC;
- ADC;
- MAE;
- MSE;
- MZOE;
- Macro-MAE.
[Get Each Country Results](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Table%202.%20Model%20Performance%20for%20Each%20Country.xlsx)
---

## Predictor Effects and Variable Importance

Predictor effects were estimated differently according to model type.

### Ordinal Logistic Regression

- Regression coefficients;
- Odds ratios.

### Ordinal Forest

Permutation importance using ORC.

### KNN

Permutation importance using ORC.

### SVM

Permutation importance using ORC.

### Ordinal XGBoost

Permutation importance using ORC.

### CORAL

Permutation importance using ORC.

---

## Bootstrap Uncertainty Estimation

Bootstrap resampling was performed on the held-out test set.

### Performance metrics

Bootstrap replicates:

B = 1000


Outputs:

- Point estimates;
- 95% confidence intervals.

### Variable importance

Bootstrap replicates:

B = 1000

Outputs:

- Importance estimates;
- 95% confidence intervals.

[Get Outputs Table](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Variable%20Importance.zip)
[Get Outputs Figure](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/7437ed57e5c225e7b5a6d5a86b1ba777df0c2713/Figure3_variable_importance.png)

---

## Cross-Validation Robustness

Five-fold cross-validation was conducted using the selected hyperparameters.

Outputs included:

- Fold-level performance;
- Mean performance;
- Standard deviations.
[Get Outputs](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Cross-Validation%20Robustness%20Test.xlsx)

---

## Pooled Dataset Analysis

To evaluate transferability and generalizability, two pooled datasets were created.

### Full Pooled Dataset

Included all eligible participants from all countries.

### Balanced Pooled Dataset

Randomly sampled:

- 200 participants × 8 countries
- Total sample size: N = 1,600

Country membership was included as a factor variable.

Country = factor(Country)

Socioeconomic status (SES) was represented using purchasing power parity-adjusted SES (ses_ppp) to improve comparability across countries with different currencies and economic contexts.

The same analytical workflow was applied to both pooled datasets.

Comparisons included:
- Full pooled vs balanced pooled;
- Pooled models evaluated on each country's test set;
- Country-specific vs pooled training strategies.
[Get Results](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/0d9c4e79621a4ec9b16d4badf56f0c13f9f675b8/Table%203.%20Pooled%20Data%20Performance.xlsx)
---

## Main Outputs

### Country-Specific Analyses
- Final test performance with 95% CIs;
- Hyperparameter summaries;
- Cross-validation summaries;
- Variable importance estimates.

### Full Pooled Analyses
- Final test performance with 95% CIs;
- Hyperparameter summaries;
- Cross-validation summaries;
- Variable importance estimates.

### Balanced Pooled Analyses
- Final test performance with 95% CIs;
- Hyperparameter summaries;
- Cross-validation summaries;
- Variable importance estimates.

### Transfer Analyses

Comparison of:
- Country-specific models;
- Full pooled models;
- Balanced pooled models.
- Transfer code: [transfer.R](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/transfer.R), [transfer.py](https://github.com/Angie-Chen2003/ordinal-diabetes-burden-prediction/blob/6d8e7dee505c2610bcba678c64a4884574d6c508/run_pipeline.py)

---
## Software
### R

Version: R 4.4.2

Used for:
- Data harmonization;
- Ordinal logistic regression;
- Ordinal forest;
- KNN;
- SVM;
- Evaluation and visualization.

### Python

Version: Python 3.13.12

Used for:
- Ordinal XGBoost;
- CORAL.

---

## Reproducibility

All analyses were conducted using fixed random seeds whenever applicable. The analytical framework was designed to facilitate adaptation to additional aging cohorts with similar data structures. Researchers may modify the harmonization procedures while preserving the overall modeling framework.

---

## Citation

If you use this repository, please cite:

Chen Y, Wang X, Shao W, Beasley J, Shu H.

Global Transportability and Shared Architecture of Ordinal Machine Learning Frameworks for Predicting Diabetes Complication Burden: A Multinational Harmonized Cohort Study.

---

## License

This repository is intended for academic and non-commercial use.

Please ensure compliance with each cohort's individual data use agreement before reproducing analyses.
