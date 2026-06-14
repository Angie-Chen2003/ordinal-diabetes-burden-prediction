### Imputation
## Predictors and Outcome variable selection
harmonized_db = read_xlsx("~/harmonized_db.xlsx")
db_diabetes55 = harmonized_db %>%
  filter(age >= 55, diabetes == 1)

data_cleaned = db_diabetes55 %>%
  mutate(
    diabetes_treat = case_when(
      diabetes_med == 1 | insulin == 1 ~ 1,
      is.na(diabetes_med) & is.na(insulin) ~ NA_real_,
      TRUE ~ 0)) %>%
  mutate(
    female = case_when(
      sex == 1 ~ 0,
      sex == 2 ~ 1
    )) %>%
  mutate(
    social_eco = case_when(
      is.na(social_welfare) & is.na(individual_pension) ~ NA_real_,
      TRUE ~ social_welfare + individual_pension
    )
  )

target_vars = c("female", "age", "ever_smoke", "pain", 
                "vision", "hypertension_med", "diabetes_treat",
                "social_eco", "medical_insurance")

data_model = data_cleaned %>%
  select(all_of(target_vars))

head(data_model)
write_xlsx(data_model, "~/Data_model.xlsx")

##Missing Data Handling
library(naniar)
library(ggplot2)
data_model = read_xlsx("~/Data_model.xlsx")

#filter outcome missing value
data_cleaned = data_model %>% 
  filter(!is.na(vision)) %>%
  filter(!is.na(pain))

#predictor missing analysis
gg_miss_var(data_cleaned, show_pct = TRUE) + 
  labs(title = "Predictor Missing Ratio")

gg_miss_upset(data_cleaned, nsets = 10)

#missing ~ covariates logistic regression
data_cleaned$miss_eco = is.na(hrs_cleaned$social_eco)
summary(glm(miss_eco ~ age + female + hypertension_med, 
            data = hrs_cleaned, family = binomial))

##Missing Value Imputation MAR
install.packages(mice)
library(mice)
vars_for_impute = c(
  "age", "female", "ever_smoke", "diabetes_treat",
  "hypertension_med", "medical_insurance",
  "social_eco","pain", "vision")

imp_data = data_cleaned[, vars_for_impute]
meth = make.method(data_imp_data)
pred = make.predictorMatrix(data_imp_data)

meth["age"]                = ""       # No NA
meth["female"]                = ""       # No NA
meth["ever_smoke"]         = "logreg"
meth["diabetes_treat"]     = "logreg"
meth["hypertension_med"]   = "logreg"
meth["medical_insurance"]  = "logreg"
meth["social_eco"]         = "pmm"
meth["pain"]               = ""       # No NA
meth["vision"]             = ""       # No NA

imp = mice(
  imp_data,
  m = 5,
  method = meth,
  predictorMatrix = pred,
  seed = 123,
  maxit = 20)

plot(imp)
densityplot(imp)
stripplot(imp, pch = 20, cex = 1.2)

##Output Five Datasets
data_1 = complete(imp, 1)
data_2 = complete(imp, 2)
data_3 = complete(imp, 3)
data_4 = complete(imp, 4)
data_5 = complete(imp, 5)

write_xlsx(data_1, "~/data_1.xlsx")
write_xlsx(data_2, "~/data_2.xlsx")
write_xlsx(data_3, "~/daat_3.xlsx")
write_xlsx(data_4, "~/data_4.xlsx")
write_xlsx(data_5, "~/data_5.xlsx")

##final Construct MAR
imputed_list = list(data_1, data_2, data_3, data_4, data_5)
continuous_vars = c("age", "social_eco")
categorical_vars = c("female", "ever_smoke", "diabetes_treat",
                     "hypertension_med", "medical_insurance", "pain",
                     "vision")

data_final = data_1

#continuous variable for mean
for (v in continuous_vars) {
  data_final[[v]] = Reduce(`+`, lapply(imputed_list, function(df) df[[v]])) / 5}

#binary variable for mode
get_mode = function(x) {
  ux = unique(x)
  ux[which.max(tabulate(match(x, ux)))]}

for (v in categorical_vars) {
  temp = do.call(cbind, lapply(imputed_list, function(df) df[[v]]))
  data_final[[v]] = apply(temp, 1, get_mode)}

write_xlsx(data_final, "~/Data_final.xlsx")