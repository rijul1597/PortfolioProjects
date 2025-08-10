# Cook County Property Valuation (Original Workflow)

Short, simple README for my Big Data Analytics in Finance project.
Random Forest with hand-picked features + linear baselines.
No `codebook.csv` required.

## Quick start
1) Put data files here (or keep them in ~/Downloads):
   - `data/raw/historic_property_data.csv`
   - `data/raw/predict_property_data.csv`
2) Install packages and run:
```r
install.packages(c("tidyverse","DescTools","dplyr","caret","corrr","glmnet","randomForest","corrplot","ggplot2"))
source("src/main.R")
```
3) Results are saved automatically.

## What the script does
- Cleans data (group fill → median/mode → winsorize).
- Drops constant + highly correlated numeric columns (keeps model features).
- Trains:
  - `lm(sale_price ~ .)` baseline
  - Random Forest on selected features
  - Linear model on selected features
- Prints RMSE.
- Saves plots + predictions.

## Outputs
- Plots → `reports/figures/`
  - `density_rf.png`, `density_lm.png`
- Predictions → `outputs/`
  - `assessment.csv`  (pid + RF assessed_value)
  - `assessment2.csv` (pid + all model predictions)

## Repo layout (minimal)
```
projects/cook-county-valuation/
  README.md
  .gitignore
  src/main.R
  data/raw/            # place the CSVs here (not committed)
  reports/figures/     # auto-saved plots
  outputs/             # auto-saved CSVs
```

## Notes
- If `data/raw/` is empty, the script will try `~/Downloads/`.
- Safe to ignore the warning about “rank-deficient” linear model.
- Don’t upload big CSVs to GitHub. They are ignored by `.gitignore`.
