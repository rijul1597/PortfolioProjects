# --------------------------------------------
# Big Data Finance — Property Valuation (original workflow, hardened)
# --------------------------------------------

# Libraries
library(tidyverse)
library(DescTools)
library(dplyr)
library(caret)
library(corrr)
library(glmnet)
library(randomForest)
library(corrplot)

set.seed(123)

# --------------------------------------------
# Paths (use your absolute paths; fall back to relative if needed)
# --------------------------------------------
# Load data
predict_data  <- read.csv("data/raw/predict_property_data.csv")
historic_data <- read.csv("data/raw/historic_property_data.csv", nrows = 50001)

#codebook_path_abs <- "data/raw/codebook.csv"
#codebook_path_rel <- "codebook.csv"

if (!file.exists(predict_path)) predict_path <- "predict_property_data.csv"
if (!file.exists(historic_path)) historic_path <- "historic_property_data.csv"

# --------------------------------------------
# Load data (you used nrows=50001 on historic; keep it)
# --------------------------------------------
predict_data  <- read.csv(predict_path)
historic_data <- read.csv(historic_path, nrows = 50001)

# Quick EDA
str(predict_data); str(historic_data)
summary(predict_data); summary(historic_data)
hist(historic_data$sale_price, main = "Distribution of Sale Prices", xlab = "Sale Price")
cat("Historic rows:", nrow(historic_data), "\n")
print(colSums(is.na(historic_data)))

# --------------------------------------------
# Drop known high-missing columns (your list)
# --------------------------------------------
drop_missing_heavy <- c(
  "meta_cdu","char_apts","char_tp_plan","char_tp_dsgn",
  "char_gar1_cnst","char_gar1_att","char_gar1_area",
  "char_attic_fnsh","char_renovation","char_porch"
)
historic_data <- historic_data %>% select(-any_of(drop_missing_heavy))
predict_data  <- predict_data  %>% select(-any_of(drop_missing_heavy))

# Drop certified estimate columns
drop_certified <- c("meta_certified_est_bldg", "meta_certified_est_land")
historic_data <- historic_data %>% select(-any_of(drop_certified))
predict_data  <- predict_data  %>% select(-any_of(drop_certified))

# --------------------------------------------
# Optional: codebook drop (only if file exists)
# --------------------------------------------
if (file.exists(codebook_path_abs) || file.exists(codebook_path_rel)) {
  cb_path <- if (file.exists(codebook_path_abs)) codebook_path_abs else codebook_path_rel
  variable_list <- readr::read_csv(cb_path, show_col_types = FALSE)
  if (all(c("var_name_standard","var_is_predictor") %in% names(variable_list))) {
    variables_to_drop <- variable_list$var_name_standard[variable_list$var_is_predictor == FALSE] %>% intersect(names(historic_data))
    historic_data <- historic_data[, !(colnames(historic_data) %in% variables_to_drop), drop = FALSE]
    predict_data  <- predict_data[,  !(colnames(predict_data)  %in% variables_to_drop), drop = FALSE]
  }
} else {
  message("codebook.csv not found — skipping that drop step (totally fine).")
}

# --------------------------------------------
# Group-aware fill, then median/mode imputation
# --------------------------------------------
historic_data_filled <- historic_data %>%
  group_by(geo_property_city, geo_property_zip, geo_municipality) %>%
  tidyr::fill(everything(), .direction = "downup") %>%
  ungroup()

# Numeric/Categorical identification
numeric_cols_hist <- sapply(historic_data_filled, is.numeric)
categorical_cols_hist <- !numeric_cols_hist

# Impute numerics with median
for (col in names(historic_data_filled)[numeric_cols_hist]) {
  historic_data_filled[[col]][is.na(historic_data_filled[[col]])] <- median(historic_data_filled[[col]], na.rm = TRUE)
}
# Impute categoricals with mode
for (col in names(historic_data_filled)[categorical_cols_hist]) {
  mode_val <- names(which.max(table(historic_data_filled[[col]])))
  if (length(mode_val) == 0 || is.na(mode_val)) next
  historic_data_filled[[col]][is.na(historic_data_filled[[col]])] <- mode_val
}

print(colSums(is.na(historic_data_filled)))

# --------------------------------------------
# Winsorize numerics
# --------------------------------------------
numeric_cols_hist <- sapply(historic_data_filled, is.numeric)
historic_data_winsorized <- historic_data_filled %>%
  mutate(across(
    names(historic_data_filled)[numeric_cols_hist],
    ~ Winsorize(., probs = c(0.01, 0.99), na.rm = TRUE)
  ))

# --------------------------------------------
# Collapse rare categorical levels to "Other"
# --------------------------------------------
threshold <- 10
# recompute categorical flags after winsorization
categorical_cols_hist2 <- !sapply(historic_data_winsorized, is.numeric)
cat_names_hist <- names(historic_data_winsorized)[categorical_cols_hist2]

for (col in cat_names_hist) {
  # treat as character for safe replacement
  vec <- as.character(historic_data_winsorized[[col]])
  counts <- table(vec)
  rare <- names(counts)[counts < threshold]
  vec[vec %in% rare] <- "Other"
  historic_data_winsorized[[col]] <- vec
}

# --------------------------------------------
# Remove constant & highly correlated numeric features (by NAME)
# BUT never drop your modeling features (protected)
# --------------------------------------------
protected_features <- c(
  "geo_white_perc","char_age","char_beds","char_heat","char_rooms","char_air",
  "char_bldg_sf","char_type_resd","geo_tract_pop","geo_ohare_noise","geo_fips",
  "econ_tax_rate","econ_midincome"
)

num_names_hist <- names(historic_data_winsorized)[sapply(historic_data_winsorized, is.numeric)]

# Remove constants
const_idx <- sapply(historic_data_winsorized[num_names_hist], function(x) var(x, na.rm = TRUE) == 0)
const_drop <- setdiff(names(const_idx)[which(const_idx)], protected_features)
historic_data_winsorized <- historic_data_winsorized %>% select(-any_of(const_drop))
num_names_hist <- setdiff(num_names_hist, const_drop)

# Correlation on numerics EXCLUDING protected features (we won't drop protected)
num_for_corr <- setdiff(num_names_hist, protected_features)
if (length(num_for_corr) > 1) {
  cor_mat <- cor(historic_data_winsorized[, num_for_corr, drop = FALSE], use = "pairwise.complete.obs")
  hi_idx <- findCorrelation(cor_mat, cutoff = 0.75)
  corr_drop <- if (length(hi_idx)) colnames(cor_mat)[hi_idx] else character(0)
} else {
  corr_drop <- character(0)
}
historic_data_winsorized <- historic_data_winsorized %>% select(-any_of(corr_drop))

# --------------------------------------------
# Train/test split
# --------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(historic_data_winsorized$sale_price, p = .8, list = FALSE, times = 1)
train_data <- historic_data_winsorized[trainIndex, ]
test_data  <- historic_data_winsorized[-trainIndex, ]

# Align factor levels between train and test (for non-numeric columns)
cat_cols_train <- names(train_data)[!sapply(train_data, is.numeric)]
for (col in cat_cols_train) {
  train_data[[col]] <- as.factor(train_data[[col]])
  test_data[[col]]  <- as.factor(test_data[[col]])
  levels(test_data[[col]]) <- levels(train_data[[col]])
}

# --------------------------------------------
# Models (your original choices)
# --------------------------------------------
# Linear model on all remaining features
linear_model <- lm(sale_price ~ ., data = train_data)
summary(linear_model)

# RF with your hand-picked features (kept protected above)
rf_formula <- sale_price ~
  geo_white_perc + char_age + char_beds + char_heat +
  char_rooms + char_air + char_bldg_sf + char_type_resd +
  geo_tract_pop + geo_ohare_noise + geo_fips +
  econ_tax_rate + econ_midincome

rf_model <- randomForest(rf_formula, data = train_data, ntree = 100)
print(rf_model)

# Linear model with important features only (on full historic_data to match your script)
sale_pred_mlm <- lm(
  sale_price ~ geo_white_perc + char_age + char_beds + char_heat +
    char_rooms + char_air + char_bldg_sf + char_type_resd +
    geo_tract_pop + geo_ohare_noise + geo_fips +
    econ_tax_rate + econ_midincome,
  data = historic_data
)
summary(sale_pred_mlm)

# Optionally tuned RF (commented in your first script)
# control  <- trainControl(method="cv", number=5)
# tuneGrid <- expand.grid(.mtry=c(1:10))
# rf_tuned <- train(rf_formula, data=train_data, method="rf",
#                   trControl=control, tuneGrid=tuneGrid)

# --------------------------------------------
# Evaluate
# --------------------------------------------
linear_preds           <- predict(linear_model, newdata = test_data)
rf_preds               <- predict(rf_model,     newdata = test_data)
linearimpfeatures_pred <- predict(sale_pred_mlm, newdata = test_data)

linear_RMSE            <- sqrt(mean((linear_preds - test_data$sale_price)^2))
rf_RMSE                <- sqrt(mean((rf_preds - test_data$sale_price)^2))
linearimpfeatures_RMSE <- sqrt(mean((linearimpfeatures_pred - test_data$sale_price)^2))

cat("Linear Regression RMSE: ", linear_RMSE, "\n")
cat("Random Forest RMSE: ",    rf_RMSE,     "\n")
cat("Linear (important feats) RMSE: ", linearimpfeatures_RMSE, "\n")

# --------------------------------------------
# Plots
# --------------------------------------------
library(ggplot2)

plot_data <- data.frame(
  Sale = c(test_data$sale_price, rf_preds),
  Type = rep(c("Actual", "Predicted"), each = length(test_data$sale_price))
)
sampled_data <- plot_data[sample(nrow(plot_data), max(1, floor(0.1 * nrow(plot_data)))), ]

ggplot(sampled_data, aes(x = Sale, fill = Type)) +
  geom_density(alpha = 0.6) +
  labs(title = "Actual vs Predicted Sale Prices", x = "Sale Price", y = "Density") +
  scale_fill_manual(values = c("red", "blue"))

linear_plot_data <- data.frame(
  Sale = c(test_data$sale_price, linearimpfeatures_pred),
  Type = rep(c("Actual", "Predicted"), each = length(test_data$sale_price))
)
linear_sampled_data <- linear_plot_data[sample(nrow(linear_plot_data), max(1, floor(0.1 * nrow(linear_plot_data)))), ]

ggplot(linear_sampled_data, aes(x = Sale, fill = Type)) +
  geom_density(alpha = 0.6) +
  labs(title = "Actual vs Predicted Sale Prices (Linear Regression)", x = "Sale Price", y = "Density") +
  scale_fill_manual(values = c("red", "blue"))

# --------------------------------------------
# SCORE on predict_data (mirror preprocessing)
# --------------------------------------------

predict_data_filled <- predict_data %>%
  group_by(geo_property_city, geo_property_zip, geo_municipality) %>%
  tidyr::fill(everything(), .direction = "downup") %>%
  ungroup()

numeric_cols_pred <- sapply(predict_data_filled, is.numeric)
categorical_cols_pred <- !numeric_cols_pred

# Median/mode impute
for (col in names(predict_data_filled)[numeric_cols_pred]) {
  predict_data_filled[[col]][is.na(predict_data_filled[[col]])] <- median(predict_data_filled[[col]], na.rm = TRUE)
}
for (col in names(predict_data_filled)[categorical_cols_pred]) {
  mode_val <- names(which.max(table(predict_data_filled[[col]])))
  if (length(mode_val) == 0 || is.na(mode_val)) next
  predict_data_filled[[col]][is.na(predict_data_filled[[col]])] <- mode_val
}

# Winsorize numerics
predict_data_winsorized <- predict_data_filled %>%
  mutate(across(
    names(predict_data_filled)[numeric_cols_pred],
    ~ Winsorize(., probs = c(0.01, 0.99), na.rm = TRUE)
  ))

# Collapse rare levels
categorical_cols_pred2 <- !sapply(predict_data_winsorized, is.numeric)
cat_names_pred <- names(predict_data_winsorized)[categorical_cols_pred2]
for (col in cat_names_pred) {
  vec <- as.character(predict_data_winsorized[[col]])
  counts <- table(vec)
  rare <- names(counts)[counts < threshold]
  vec[vec %in% rare] <- "Other"
  predict_data_winsorized[[col]] <- vec
}

# Remove constants and correlated columns using the SAME logic/names as training
# (constant drop names were in const_drop; corr drop names were in corr_drop)
predict_data_winsorized <- predict_data_winsorized %>%
  select(-any_of(const_drop), -any_of(corr_drop))

# Ensure factors used in models are factors here too, with train levels when possible
# (safe-guard these three if they exist)
for (nm in c("ind_large_home","ind_garage","ind_arms_length")) {
  if (nm %in% names(predict_data_winsorized)) {
    predict_data_winsorized[[nm]] <- as.factor(predict_data_winsorized[[nm]])
  }
}
# Align factor levels for any categorical columns shared with train
for (col in cat_cols_train) {
  if (col %in% names(predict_data_winsorized)) {
    predict_data_winsorized[[col]] <- as.factor(predict_data_winsorized[[col]])
    levels(predict_data_winsorized[[col]]) <- levels(train_data[[col]])
  }
}

# Score — use the preprocessed scoring frame
rf_preds_score   <- predict(rf_model,     newdata = predict_data_winsorized)
# You can score other models too if you want:
linear_score     <- tryCatch(predict(linear_model,    newdata = predict_data_winsorized), error = function(e) rep(NA_real_, nrow(predict_data_winsorized)))
lin_imp_score    <- tryCatch(predict(sale_pred_mlm,   newdata = predict_data_winsorized), error = function(e) rep(NA_real_, nrow(predict_data_winsorized)))

# --------------------------------------------
# Save predictions
# --------------------------------------------
out1 <- "outputs/assessment.csv"
out2 <- "outputs/assessment2.csv"
if (!dir.exists(dirname(out1))) dir.create(dirname(out1), recursive = TRUE, showWarnings = FALSE)

# (keep your write.csv lines the same, they’ll use out1/out2)


write.csv(data.frame(pid = predict_data$pid, assessed_value = rf_preds_score),
          out1, row.names = FALSE)

write.csv(data.frame(pid = predict_data$pid,
                     linear_preds = linear_score,
                     rf_preds = rf_preds_score,
                     linearimpfeatures_pred = lin_imp_score),
          out2, row.names = FALSE)

cat("Saved:\n  ", out1, "\n  ", out2, "\n")
