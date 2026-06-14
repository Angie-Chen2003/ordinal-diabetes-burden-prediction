### Descriptive Analysis
harmonized_db = read_xlsx("~/harmonized_data.xlsx")

db55 = harmonized_db %>%
  filter(age >= 55)
db_count = db55 %>%
  count(diabetes) %>%
  mutate(percent = n / sum(n) * 100)

db_diabetes55 = harmonized_db %>%
  filter(age >= 55, diabetes == 1)

db_diabetes55 = db_diabetes55 %>%
  mutate(
    diabetes_treat = case_when(
      diabetes_med == 1 | insulin == 1 ~ 1,
      is.na(diabetes_med) & is.na(insulin) ~ NA_real_,
      TRUE ~ 0
    )
  )

categorical_vars = c(
  "ever_smoke",
  "current_smoke",
  "diabetes_treat",
  "hypertension_med",
  "pain",
  "vision",
  "medical_insurance",
  "sex",
  "kidney",
  "diabetes_diet")

cat = db_diabetes55 %>%
  dplyr::select(all_of(categorical_vars))
cat1 = cat %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "value") %>%
  count(variable, value) %>%
  group_by(variable) %>%
  mutate(
    percent = round(100 * n / sum(n), 2) ) %>%
  ungroup()

print(cat1)
db_count

### Continuous Variable
pension_mean = mean(mahs_db_diabetes55$pension, na.rm = TRUE)
pension_sd = sd(mahs_db_diabetes55$pension, na.rm = TRUE)

pension_median = median(mahs_db_diabetes55$pension, na.rm = TRUE)
pension_q1 = quantile(mahs_db_diabetes55$pension, 0.25, na.rm = TRUE)
pension_q3 = quantile(mahs_db_diabetes55$pension, 0.75, na.rm = TRUE)

pension_na = sum(is.na(mahs_db_diabetes55$pension))
pension_na_pct = round(pension_na / nrow(mahs_db_diabetes55) * 100, 2)

sw_mean = mean(mahs_db_diabetes55$social_welfare, na.rm = TRUE)
sw_sd = sd(mahs_db_diabetes55$social_welfare, na.rm = TRUE)

sw_median = median(mahs_db_diabetes55$social_welfare, na.rm = TRUE)
sw_q1 = quantile(mahs_db_diabetes55$social_welfare, 0.25, na.rm = TRUE)
sw_q3 = quantile(mahs_db_diabetes55$social_welfare, 0.75, na.rm = TRUE)

sw_na = sum(is.na(mahs_db_diabetes55$social_welfare))
sw_na_pct = round(sw_na / nrow(mahs_db_diabetes55) * 100, 2)

age_mean = mean(mahs_db_diabetes55$age, na.rm = TRUE)
age_sd = sd(mahs_db_diabetes55$age, na.rm = TRUE)

age_median = median(mahs_db_diabetes55$age, na.rm = TRUE)
age_q1 = quantile(mahs_db_diabetes55$age, 0.25, na.rm = TRUE)
age_q3 = quantile(mahs_db_diabetes55$age, 0.75, na.rm = TRUE)

age_na = sum(is.na(mahs_db_diabetes55$age))
age_na_pct = round(age_na / nrow(mahs_db_diabetes55) * 100, 2)

cat(
  "Mean (SD):", round(pension_mean,2), "(", round(pension_sd,2), ")\n",
  "Median (IQR):", round(pension_median,2), "(",
  round(pension_q1,2), "-", round(pension_q3,2), ")\n",
  "NA:", pension_na, "(", pension_na_pct, "%)\n"
)
cat(
  "Mean (SD):", round(sw_mean,2), "(", round(sw_sd,2), ")\n",
  "Median (IQR):", round(sw_median,2), "(",
  round(sw_q1,2), "-", round(sw_q3,2), ")\n",
  "NA:", sw_na, "(", sw_na_pct, "%)\n"
)
cat(
  "Mean (SD):", round(age_mean,2), "(", round(age_sd,2), ")\n",
  "Median (IQR):", round(age_median,2), "(",
  round(age_q1,2), "-", round(age_q3,2), ")\n",
  "NA:", age_na, "(", age_na_pct, "%)\n"
)