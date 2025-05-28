# Load necessary R libraries
# Data manipulation
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
library(dplyr)
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)
if (!require(RSQLite)) install.packages("RSQLite", repos = "http://cran.us.r-project.org")
library(RSQLite)

# Modeling and Preprocessing
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret) # For train/test split, preprocessing, PCA
if (!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
library(randomForest) # For feature importance
if (!require(pdp)) install.packages("pdp", repos = "http://cran.us.r-project.org")
library(pdp)


# Plotting
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library(ggplot2)
if (!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
library(ggpubr) # For arranging ggplots
if (!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
library(corrplot) # For correlation heatmaps
if (!require(skimr)) install.packages("skimr", repos = "http://cran.us.r-project.org")
library(skimr) # For descriptive statistics

# Set plotting style (similar to seaborn-v0_8-whitegrid and viridis palette)
theme_set(theme_bw())
options(ggplot2.continuous.colour = "viridis", ggplot2.continuous.fill = "viridis")
options(ggplot2.discrete.colour = RColorBrewer::brewer.pal(8, "Dark2"), ggplot2.discrete.fill = RColorBrewer::brewer.pal(8, "Dark2"))


# --- Phase 2: Data Analysis and Cleansing ---

## --- 2.1 Data Pre-processing and cleaning ---

# --- Configuration ---
current_dir <- getwd()
file_path <- file.path(current_dir, "data", "2019", "2019-01.sqlite")
if (!file.exists(file_path)) {
  stop(paste("Database file not found at:", file_path))
}
target_variable <- "fare_amount"
test_size_prop <- 0.1
random_seed <- 42
task_type <- 'regression' # 'regression' or 'classification'

# --- Load Data ---
load_and_split_data <- function(db_path, target_col, test_prop, seed, task) {
  # Connect to SQLite database
  con <- dbConnect(RSQLite::SQLite(), db_path)
  query <- "SELECT * FROM tripdata"
  df <- dbGetQuery(con, query)
  dbDisconnect(con)

  cat("--- Initial Data Info ---\n")
  print(skim(df))
  cat("\n--- Missing Values ---\n")
  print(colSums(is.na(df)))

  # Split data into features and labels
  y <- df[[target_col]]
  X <- df %>% select(-all_of(target_col))

  if (task == 'classification') {
    # Create fare classes
    y <- case_when(
      y < 10 ~ 1,
      y >= 10 & y < 30 ~ 2,
      y >= 30 & y < 60 ~ 3,
      y >= 60 ~ 4,
      TRUE ~ 0 # default
    )
  }

  # Split data into training and testing sets
  set.seed(seed)
  train_indices <- createDataPartition(y, p = 1 - test_prop, list = FALSE)

  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]

  cat("\nData loaded and split successfully.\n")
  return(list(
    data_train = X_train,
    labels_train = y_train,
    data_test = X_test,
    labels_test = y_test,
    original_df = df
  ))
}

# --- Manipulate Data ---
manipulate_data <- function(data_list) {
  X_train <- data_list$data_train
  X_test <- data_list$data_test

  cat("\n--- Converting Data Types ---\n")
  # Convert datetime columns
  datetime_cols <- c('tpep_pickup_datetime', 'tpep_dropoff_datetime')
  for (col in datetime_cols) {
    X_train[[col]] <- ymd_hms(X_train[[col]], quiet = TRUE, tz = "UTC") # Assuming UTC
    X_test[[col]] <- ymd_hms(X_test[[col]], quiet = TRUE, tz = "UTC")
  }

  # Convert store_and_fwd_flag
  X_train$store_and_fwd_flag <- ifelse(X_train$store_and_fwd_flag == 'Y', 1, 0)
  X_test$store_and_fwd_flag <- ifelse(X_test$store_and_fwd_flag == 'Y', 1, 0)
  
 # Convert other categorical columns to integer
  categorical_cols <- c('vendorid', 'ratecodeid', 'pulocationid', 'dolocationid', 'payment_type', 'store_and_fwd_flag')
  for (col in categorical_cols) {
    if (col %in% names(X_train)) X_train[[col]] <- as.integer(X_train[[col]])
    if (col %in% names(X_test)) X_test[[col]] <- as.integer(X_test[[col]])
  }
  cat("Data types converted.\n")

  cat("\n--- Engineering Base Features ---\n")
  # Create Trip duration feature
  X_train$trip_duration_secs <- as.numeric(difftime(X_train$tpep_dropoff_datetime, X_train$tpep_pickup_datetime, units = "secs"))
  X_test$trip_duration_secs <- as.numeric(difftime(X_test$tpep_dropoff_datetime, X_test$tpep_pickup_datetime, units = "secs"))

  cat("\n--- Handling Data Leakage ---\n")
  leakage_cols <- c('total_amount', 'tip_amount', 'extra', 'mta_tax', 'improvement_surcharge')
  X_train <- X_train %>% select(-any_of(leakage_cols))
  X_test <- X_test %>% select(-any_of(leakage_cols))
  cat("Leakage columns removed.\n")

  data_list$data_train <- X_train
  data_list$data_test <- X_test
  return(data_list)
}


# --- Execute Data Loading and Manipulation ---
loaded_data <- load_and_split_data(file_path, target_variable, test_size_prop, random_seed, task_type)
data_loader_list <- manipulate_data(loaded_data)

# --- Display some initial stats ---
cat("\n--- Post-Manipulation Data Checks (Train Set) ---\n")
numeric_cols_to_check <- c('passenger_count', 'trip_distance', 'trip_duration_secs', 'congestion_surcharge', 'tolls_amount')
categorical_cols_to_check <- c('ratecodeid', 'payment_type', 'store_and_fwd_flag', 'vendorid', 'pulocationid', 'dolocationid')

for(col_name in numeric_cols_to_check){
  if(col_name %in% names(data_loader_list$data_train)){
    cat(paste0("Max value for ", col_name, ": "), max(data_loader_list$data_train[[col_name]], na.rm = TRUE), "\n")
    cat(paste0("Min value for ", col_name, ": "), min(data_loader_list$data_train[[col_name]], na.rm = TRUE), "\n")
  }
}
for(col_name in categorical_cols_to_check){
    if(col_name %in% names(data_loader_list$data_train)){
       cat(paste0("Unique values for ", col_name, ": "), paste(unique(data_loader_list$data_train[[col_name]]), collapse=", "), "\n")
    }
}


# --- Data Cleaning Function ---
clean_data <- function(data_list) {
  X_train <- data_list$data_train
  y_train <- data_list$labels_train
  X_test <- data_list$data_test

  cat("\n--- Removing Duplicates (Train Set) ---\n")
  initial_rows_train <- nrow(X_train)
  # Keep labels aligned with X_train
  train_df_full <- bind_cols(X_train, fare_amount_label = y_train)
  train_df_full_unique <- train_df_full %>% distinct()
  
  X_train <- train_df_full_unique %>% select(-fare_amount_label)
  y_train <- train_df_full_unique$fare_amount_label
  
  if (initial_rows_train != nrow(X_train)) {
    cat("Duplicate rows removed from training data.\n")
  } else {
    cat("No duplicate rows found in the training data.\n")
  }

  cat("\n--- Handling Missing Values ---\n")
  # Remove columns with > 50% missing values (in train set)
  na_percentage_train <- colMeans(is.na(X_train))
  cols_to_remove_train <- names(na_percentage_train[na_percentage_train > 0.5])
  
  if (length(cols_to_remove_train) > 0) {
    X_train <- X_train %>% select(-all_of(cols_to_remove_train))
    X_test <- X_test %>% select(-all_of(cols_to_remove_train)) # Apply same column removal to test
    cat(paste(cols_to_remove_train, collapse=", "), "Columns with > 50% missing values removed.\n")
  }
  
  # Drop rows with any NA
  complete_cases_train <- complete.cases(X_train)
  X_train <- X_train[complete_cases_train, ]
  y_train <- y_train[complete_cases_train]
  
  # For X_test, we can either impute or drop
  y_test_df <- data.frame(fare_amount_label_test = data_list$labels_test)
  X_test_full <- bind_cols(X_test, y_test_df)
  complete_cases_test <- complete.cases(X_test_full) # Check NAs in features and label if it exists
  X_test_full_cleaned <- X_test_full[complete_cases_test, ]
  X_test <- X_test_full_cleaned %>% select(-fare_amount_label_test)
  y_test <- X_test_full_cleaned$fare_amount_label_test
  
  cat("Rows with NA values dropped from train and test sets.\n")

  cat("\n--- Removing Outliers (Train Set) ---\n")
  # Create a temporary data frame for filtering to keep X_train and y_train aligned
  train_df_for_outliers <- X_train
  train_df_for_outliers$fare_amount_label_for_outlier <- y_train

  # Fare amount outliers
  train_df_for_outliers <- train_df_for_outliers %>%
    filter(fare_amount_label_for_outlier > 0 & fare_amount_label_for_outlier <= 300)

  # Passenger count outliers
  if ("passenger_count" %in% names(train_df_for_outliers)) {
    train_df_for_outliers <- train_df_for_outliers %>%
      filter(passenger_count > 0)
  }
  
  # RatecodeID outliers
  if ("ratecodeid" %in% names(train_df_for_outliers)) {
    valid_ratecodes <- c(1, 2, 3, 4, 5, 6)
    train_df_for_outliers <- train_df_for_outliers %>%
      filter(as.integer(ratecodeid) %in% valid_ratecodes)
  }
  
  # VendorID outliers
  if ("vendorid" %in% names(train_df_for_outliers)) {
    valid_vendorid <- c(1, 2)
    train_df_for_outliers <- train_df_for_outliers %>%
      filter(as.integer(vendorid) %in% valid_vendorid)
  }
  
  # Trip distance outliers
  if ("trip_distance" %in% names(train_df_for_outliers)) {
    train_df_for_outliers <- train_df_for_outliers %>%
      filter(trip_distance > 0 & trip_distance <= 60)
  }
  
  # Trip duration outliers
  if ("trip_duration_secs" %in% names(train_df_for_outliers)) {
    train_df_for_outliers <- train_df_for_outliers %>%
      filter(trip_duration_secs > 0 & trip_duration_secs <= 10000.0)
  }
  
  # Tolls amount outliers
  if ("tolls_amount" %in% names(train_df_for_outliers)) {
     train_df_for_outliers <- train_df_for_outliers %>%
      filter(tolls_amount >= 0 & tolls_amount <= 7.42)
  }

  # Update X_train and y_train
  X_train <- train_df_for_outliers %>% select(-fare_amount_label_for_outlier)
  y_train <- train_df_for_outliers$fare_amount_label_for_outlier
  cat("Outliers removed from the training dataset.\n")

  data_list$data_train <- X_train
  data_list$labels_train <- y_train
  data_list$data_test <- X_test
  data_list$labels_test <- y_test # Update y_test after potential row drops
  return(data_list)
}

# --- Execute Data Cleaning ---
cat("\nBefore data cleaning\n")
cat("Training data shape:", dim(data_loader_list$data_train), "\n")
cat("Training labels length:", length(data_loader_list$labels_train), "\n")
cat("Testing data shape:", dim(data_loader_list$data_test), "\n")
cat("Testing labels length:", length(data_loader_list$labels_test), "\n")

data_loader_list <- clean_data(data_loader_list)

cat("\nAfter data cleaning\n")
cat("Training data shape:", dim(data_loader_list$data_train), "\n")
cat("Training labels length:", length(data_loader_list$labels_train), "\n")
cat("Testing data shape:", dim(data_loader_list$data_test), "\n")
cat("Testing labels length:", length(data_loader_list$labels_test), "\n")


# --- 2.2 Exploratory Data Analysis (EDA) ---

# Combine train features and labels for EDA
train_eda_df <- data_loader_list$data_train
train_eda_df$fare_amount <- data_loader_list$labels_train

cat("\n--- Performing EDA ---\n")

# Sample data for plots (to reduce computation)
set.seed(random_seed)
sample_size <- min(5000, nrow(train_eda_df)) # Take at most 5000 rows
sample_train_eda_df <- train_eda_df[sample(nrow(train_eda_df), sample_size), ]

# --- Plot Target Distribution ---
cat("\nPlot Target Distribution\n")
p_hist_fare <- ggplot(sample_train_eda_df, aes(x = fare_amount)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(alpha = .2, fill = "#FF6666") +
  ggtitle("Distribution of Fare Amount (Sampled Data)") +
  xlab("Fare Amount") + ylab("Frequency")
print(p_hist_fare)

p_box_fare <- ggplot(sample_train_eda_df, aes(y = fare_amount)) +
  geom_boxplot(fill = "skyblue", alpha = 0.7) +
  ggtitle("Box Plot of Fare Amount (Sampled Data)") +
  ylab("Fare Amount")
print(p_box_fare)

if (min(sample_train_eda_df$fare_amount, na.rm = TRUE) > 0) {
  p_hist_log_fare <- ggplot(sample_train_eda_df, aes(x = log1p(fare_amount))) +
    geom_histogram(aes(y = ..density..), bins = 50, fill = "skyblue", color = "black", alpha = 0.7) +
    geom_density(alpha = .2, fill = "#FF6666") +
    ggtitle("Distribution of Log(Fare Amount + 1) (Sampled Data)") +
    xlab("Log(Fare Amount + 1)") + ylab("Frequency")
  print(p_hist_log_fare)
} else {
  cat("Cannot log transform Fare Amount as it contains non-positive values after cleaning.\n")
}


# --- Plot Numerical Distributions ---
cat("\nPlot Numerical Distributions\n")
# Define numerical columns 
numerical_cols_eda <- c("trip_distance", "trip_duration_secs", "tolls_amount")
for (col in numerical_cols_eda) {
  if (col %in% names(sample_train_eda_df)) {
    p_hist_num <- ggplot(sample_train_eda_df, aes_string(x = col)) +
      geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", alpha = 0.7) +
      geom_density(alpha = .2, fill = "#FF6666") +
      ggtitle(paste("Distribution of", col, "(Sampled Data)"))
    print(p_hist_num)
    
    p_box_num <- ggplot(sample_train_eda_df, aes_string(y = col)) +
      geom_boxplot(fill = "skyblue", alpha = 0.7) +
      ggtitle(paste("Box Plot of", col, "(Sampled Data)"))
    print(p_box_num)
  } else {
    cat(paste("Numerical column '", col, "' not found for distribution plot.\n"))
  }
}

# --- Plot Categorical Distributions ---
cat("\nPlot Categorical Distributions\n")
# Define categorical columns
categorical_cols_eda <- c("ratecodeid", "payment_type", "vendorid", "store_and_fwd_flag")
for (col in categorical_cols_eda) {
  if (col %in% names(sample_train_eda_df)) {
    p_count_cat <- ggplot(sample_train_eda_df, aes_string(x = factor(sample_train_eda_df[[col]]))) + 
      geom_bar(aes(fill = factor(sample_train_eda_df[[col]])), alpha=0.7) +
      ggtitle(paste("Distribution of", col, "(Sampled Data)")) +
      xlab(col) +
      coord_flip() + # For y-axis countplot style
      theme(legend.position = "none")
    print(p_count_cat)
  } else {
    cat(paste("Categorical column '", col, "' not found for distribution plot.\n"))
  }
}

# --- Plot Numerical vs. Target ---
cat("\nPlot Numerical vs. Target (Fare Amount)\n")
# Further reduce sample size for scatter plots
scatter_sample_size <- min(1000, nrow(sample_train_eda_df))
scatter_sample_df <- sample_train_eda_df[sample(nrow(sample_train_eda_df), scatter_sample_size), ]

for (col in numerical_cols_eda) {
  if (col %in% names(scatter_sample_df)) {
    p_scatter_num_target <- ggplot(scatter_sample_df, aes_string(x = col, y = "fare_amount")) +
      geom_point(alpha = 0.5) +
      ggtitle(paste("Fare Amount vs.", col, "(Sampled Data -", scatter_sample_size, "points)"))
    print(p_scatter_num_target)
  } else {
     cat(paste("Numerical column '", col, "' not found for scatter plot vs target.\n"))
  }
}

# --- Plot Categorical vs. Target ---
cat("\nPlot Categorical vs. Target (Fare Amount)\n")
for (col in categorical_cols_eda) {
  if (col %in% names(sample_train_eda_df)) {
    sample_train_eda_df_cat_factor <- sample_train_eda_df
    sample_train_eda_df_cat_factor[[col]] <- factor(sample_train_eda_df_cat_factor[[col]])
    
    p_box_cat_target <- ggplot(sample_train_eda_df_cat_factor, aes_string(x = col, y = "fare_amount")) +
      geom_boxplot(aes_string(fill = col), alpha = 0.7) +
      ggtitle(paste("Fare Amount vs.", col, "(Sampled Data)")) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
      xlab(col)
    print(p_box_cat_target)
  } else {
    cat(paste("Categorical column '", col, "' not found for box plot vs target.\n"))
  }
}

# --- Plot Correlation Heatmap ---
cat("\nPlot Correlation Heatmap of Numerical Features\n")
numerical_df_for_corr <- sample_train_eda_df %>% select(where(is.numeric))
if (ncol(numerical_df_for_corr) > 1) {
  corr_matrix <- cor(numerical_df_for_corr, use = "pairwise.complete.obs")
  corrplot(corr_matrix, method = "color", type = "upper", order = "hclust",
           addCoef.col = "black", # Add coefficient on heatmap
           tl.col = "black", tl.srt = 45,
           diag = FALSE,
           title = "Correlation Heatmap of Numerical Features (Sampled Data)", mar=c(0,0,1,0))
} else {
  cat("Not enough numerical columns found for correlation heatmap.\n")
}

# --- Plot Feature Importance (using Random Forest) ---
cat("\n--- Feature Importance using Random Forest ---\n")
features_for_rf <- data_loader_list$data_train %>%
                     select(any_of(c(numerical_cols_eda, categorical_cols_eda))) %>%
                     mutate(across(any_of(categorical_cols_eda), as.factor))

rf_data_train <- data_loader_list$data_train
rf_labels_train <- data_loader_list$labels_train

# Convert categorical columns to factors for RF
for(col in categorical_cols_eda) {
  if(col %in% names(rf_data_train)) {
    rf_data_train[[col]] <- as.factor(rf_data_train[[col]])
  }
}
# Ensure all selected columns are numeric or factor
rf_data_train <- rf_data_train %>% select(where(is.numeric), where(is.factor))

cat("\n--- Feature Importance using Random Forest --- 2\n")
temp_rf_df <- cbind(rf_data_train, rf_labels_train)
temp_rf_df_clean <- na.omit(temp_rf_df)
rf_data_train_clean <- temp_rf_df_clean[, -ncol(temp_rf_df_clean)]
rf_labels_train_clean <- temp_rf_df_clean$rf_labels_train

cat("\n--- Feature Importance using Random Forest 3 ---\n")

if(nrow(rf_data_train_clean) > 10 && ncol(rf_data_train_clean) > 0) {
    # Subsample the data to avoid memory issues
    set.seed(random_seed)
    subsample_size <- min(10000, nrow(rf_data_train_clean))
    subsample_indices <- sample(1:nrow(rf_data_train_clean), subsample_size)
    
    rf_data_subsample <- rf_data_train_clean[subsample_indices, ]
    rf_labels_subsample <- rf_labels_train_clean[subsample_indices]
    
    set.seed(random_seed)
    rf_model <- randomForest(y = rf_labels_subsample, 
                            x = rf_data_subsample, 
                            ntree = 20, 
                            maxnodes = 30, 
                            importance = TRUE,
                            na.action = na.roughfix) 

    importance_rf <- importance(rf_model, type = 1) # Type 1 is %IncMSE for regression
    var_importance_df <- data.frame(
    Variable = rownames(importance_rf),
    Importance = importance_rf[, 1]
    )
    var_importance_df <- var_importance_df %>% arrange(Importance)
    var_importance_df$Variable <- factor(var_importance_df$Variable, levels = var_importance_df$Variable)

    cat("\n--- Feature Importance using Random Forest 4 ---\n")
    p_feat_imp <- ggplot(var_importance_df, aes(x = Importance, y = Variable)) +
    geom_bar(stat = "identity", aes(fill=Variable)) +
    ggtitle("Feature Importance (from Random Forest %IncMSE)") +
    theme(legend.position = "none")
    print(p_feat_imp)

    cat("\n--- Feature Importance using Random Forest 5 ---\n")
} else {
    cat("Not enough data or features for Random Forest feature importance plot after NA removal.\n")
}


# --- Perform Dimension Reduction Viz (PCA) ---
cat("\n--- PCA Visualization ---\n")
# Select only numeric features for PCA from the cleaned training set
pca_data <- data_loader_list$data_train %>%
  select(where(is.numeric)) %>%
  na.omit()

if (nrow(pca_data) >= 2 && ncol(pca_data) >= 2) {
  pca_results <- prcomp(pca_data, center = TRUE, scale. = TRUE)
  
  cat(paste("Explained variance by PCA components (first 5):", 
            paste(round(summary(pca_results)$importance[2, 1:min(5, ncol(pca_data))] * 100, 2), collapse="%, "), "%\n"))

  pc_df <- data.frame(
    PC1 = pca_results$x[, 1],
    PC2 = pca_results$x[, 2]
  )
  

  p_pca <- ggplot(pc_df, aes(x = PC1, y = PC2))
  
  p_pca <- p_pca + geom_point(alpha = 0.7, size = 2) +
    ggtitle("2D PCA Visualization") +
    xlab("Principal Component 1") +
    ylab("Principal Component 2") +
    theme_minimal()
  print(p_pca)
  
} else {
  cat("Not enough numeric data for PCA after NA removal or too few features/samples.\n")
}


# --- 2.3 Feature Engineering and Transformation ---

cat("\n--- Final Feature Selection and Normalization\n")
final_numerical_features <- c('trip_distance', 'trip_duration_secs', 'tolls_amount')
final_categorical_features <- c('ratecodeid')

# Select these columns in train and test sets
X_train_final <- data_loader_list$data_train %>% select(any_of(c(final_numerical_features, final_categorical_features)))
X_test_final <- data_loader_list$data_test %>% select(any_of(c(final_numerical_features, final_categorical_features)))

# Convert categoricals to factors if not already
for(col in final_categorical_features){
    if(col %in% names(X_train_final)) X_train_final[[col]] <- as.factor(X_train_final[[col]])
    if(col %in% names(X_test_final)) X_test_final[[col]] <- as.factor(X_test_final[[col]])
}

# Normalizing/Scaling data
num_preproc_pipeline <- preProcess(X_train_final %>% select(any_of(final_numerical_features)), 
                                   method = c("center", "scale", "nzv", "knnImpute")) # nzv for near-zero variance, knnImpute for NAs
X_train_num_processed <- predict(num_preproc_pipeline, X_train_final %>% select(any_of(final_numerical_features)))
X_test_num_processed <- predict(num_preproc_pipeline, X_test_final %>% select(any_of(final_numerical_features)))

# For categorical (if one-hot encoding is desired, else they remain factors)
# Python's ColumnTransformer implicitly handles numeric/categorical separately.
# If one-hot encoding for categorical:
X_train_cat_processed <- X_train_final %>% select(any_of(final_categorical_features))
X_test_cat_processed <- X_test_final %>% select(any_of(final_categorical_features))

# Combine processed numeric and original categorical features
X_train_processed <- bind_cols(X_train_num_processed, X_train_cat_processed)
X_test_processed <- bind_cols(X_test_num_processed, X_test_cat_processed)


# Update the data_loader_list with processed features
data_loader_list$data_train_processed <- X_train_processed
data_loader_list$data_test_processed <- X_test_processed

cat("Data normalization/scaling applied.\n")
cat("\n--- Processed Training Data Head ---\n")
print(head(data_loader_list$data_train_processed))


# --- Save Processed Data (Equivalent to pickling data_loader in python) ---
saveRDS(data_loader_list, file = "../data/processed/final_processed_data_loader_list.rds")
cat("\nProcessed data_loader_list saved to ../data/processed/final_processed_data_loader_list.rds\n")

cat("\n--- R Script Execution Finished ---\n")