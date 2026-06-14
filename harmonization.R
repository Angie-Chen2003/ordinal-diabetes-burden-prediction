### Data Harmonization 
library(dplyr)
data_db = read_xlsx("~/country_datasets.xlsx")
data_db = data_db %>%
  rename(
    current_smoke      = , #current_smoke_col
    ever_smoke         = , #have ever smoked
    diabetes_med       = , #use of oral diabetes medication
    insulin            = , #use of insulin
    diabetes_diet      = , #wether change diets to control diabetes
    hypertension_med   = , #use of high blood control medication 
    pain               = , #self rated: have ever experienced pain in current life
    vision             = , #self rated: wether have vision problems
    medical_insurance  = , #wether have public medical insurance
    age                = , #age column
    sex                =   #sex column
      ) %>%
  mutate(
    diabetes = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  ) %>%
  mutate(
    insulin = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  ) %>%
  mutate(
    diabetes_diet = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  ) %>%
  mutate(
    ever_smoke = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    ) 
  ) %>%
  mutate(
    vision = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    ) 
  ) %>%
  mutate(
    medical_insurance = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  )
data_db = data_db %>%
  mutate(
    hypertension_med = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  ) %>%
  mutate(
    pain = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  ) %>%
  mutate(
    vision = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  )%>%
  mutate(
    current_smoke = case_when(
      code_Yes ~ 1,
      code_Refused ~ NA,
      code_Unknown ~ NA,
      code_NA ~ NA_real_,
      code_No ~ 0
    )
  )
write_xlsx(data_db, "~/harmonized_datasets.xlsx")
