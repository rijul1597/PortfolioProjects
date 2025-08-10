# Cook County Property Valuation (Original Workflow)

One-pager for my Big Data Analytics in Finance project. Random Forest with hand-picked features + linear baselines. No `codebook.csv`.

## Steps

**1) Get the data**

- Put these files in `data/raw/` (you may commit them):
  - `historic_property_data.csv`
  - `predict_property_data.csv`

**2) Install packages** In R:

```r
install.packages(c("tidyverse","DescTools","caret","randomForest","ggplot2"))
```

**3) Run the script**

```r
source("src/main.R")
```

**4) See results**

- Predictions:
  - `outputs/assessment.csv`   → `pid`, `assessed_value` (Random Forest)
  - `outputs/assessment2.csv`  → `pid` + predictions from all models
- The script also prints RMSEs to the console.

**5) (Optional) Edit paths**

- If you move the CSVs, update the two `read.csv("data/raw/...")` lines in `src/main.R`.

## What the script does (super short)

- Group fill by city/zip/municipality → median/mode impute → winsorize (1%/99%).
- Collapse rare categorical levels to `"Other"`.
- Drop constant & highly correlated numeric columns (keeps the model’s key features).
- Train:
  - `lm(sale_price ~ .)` baseline
  - Random Forest on selected features
  - Linear model on the same selected features
- Evaluate (80/20 split, `set.seed(123)`) and save predictions.

## Repo layout (minimal)

```
projects/cook-county-valuation/
  README.md
  .gitignore
  src/main.R
  data/raw/        # the two CSVs
  outputs/         # created by the script
```

## Notes

- “Rank-deficient” warning from the full linear model is expected and OK.

