# Import necessary libraries
import os
import pandas as pd

# Set up the directory path
main_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/raw"
# Define lists to hold file names
raw_files = []
additional_files = []

# Populate the lists with relevant file names
for filename in os.listdir(main_dir):
    if filename.endswith("_raw.csv"):
        raw_files.append(filename)
    elif "_va3.csv" in filename:
        additional_files.append(filename)

# Sort files for consistency
raw_files.sort()
additional_files.sort()

# Initialize a list to hold all combined dataframes
combined_dataframes = []

# Function to add Subject and Story columns and merge data
def merge_files_add_metadata(raw_file, additional_file):
    # Extract subject and story from the file name
    letter_number = raw_file.split("_")[0]  # Extract "a1", "a2", etc.
    subject = letter_number[0].upper()      # Extract letter and capitalize
    story = letter_number[1]                # Extract number
    
    # Read raw and additional data files
    raw_df = pd.read_csv(os.path.join(main_dir, raw_file))
    additional_df = pd.read_csv(os.path.join(main_dir, additional_file))
    
    # Add Subject and Story columns
    raw_df['Subject'] = subject
    raw_df['Story'] = story
    additional_df['Subject'] = subject
    additional_df['Story'] = story
    
    # Merge additional file data to raw file by columns
    combined_df = pd.concat([raw_df, additional_df.iloc[:len(raw_df)]], axis=1)
    
    return combined_df

# Iterate through raw files to merge with corresponding va3 files
for raw_file in raw_files:
    letter_number = raw_file.split("_")[0]  # Extract "a1", "a2", etc.
    
    # Find corresponding additional file
    additional_file = f"{letter_number}_va3.csv"
    
    if additional_file in additional_files:
        combined_df = merge_files_add_metadata(raw_file, additional_file)
        combined_dataframes.append(combined_df)

# Concatenate all combined dataframes vertically
final_combined_df = pd.concat(combined_dataframes, ignore_index=True)

# Remove duplicate 'Subject' and 'Story' columns, if any exist, by keeping the first occurrence
final_combined_df = final_combined_df.loc[:, ~final_combined_df.columns.duplicated()]

# Save the final dataframe to a new CSV file
final_combined_path = os.path.join(main_dir, "final_combined_data_with_metadata.csv")
final_combined_df.to_csv(final_combined_path, index=False)

# Output confirmation
print(f"Combined data with metadata saved to {final_combined_path}")

# Import necessary additional libraries
import numpy as np
from scipy.stats import skew, kurtosis, zscore

# Set up the directory path
main_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/raw"

# Read the data
df_path = os.path.join(main_dir, "final_combined_data_with_metadata.csv")
df = pd.read_csv(df_path)

# Ensure all columns are numbers where possible
df = df.apply(pd.to_numeric, errors='ignore')

# Lists to store results
numerical_summary = []
categorical_summary = []

# Numeric Features
numeric_features = df.select_dtypes(include=[np.number]).columns.difference(['phase', 'Phase'])
for feature in numeric_features:
    # Calculate basic statistics
    feature_data = df[feature].dropna()
    z_scores = zscore(feature_data)
    
    highest_z = z_scores.max()
    lowest_z = z_scores.min()
    
    # Average Z-scores of top/bottom 300 values
    sorted_indices = np.argsort(z_scores)
    top_300_average = z_scores[sorted_indices[-300:]].mean() if len(z_scores) >= 300 else float('nan')
    bottom_300_average = z_scores[sorted_indices[:300]].mean() if len(z_scores) >= 300 else float('nan')
    
    summary = {
        'Feature': feature,
        'Mean': feature_data.mean(),
        'Median': feature_data.median(),
        'Std': feature_data.std(),
        'Min': feature_data.min(),
        'Max': feature_data.max(),
        'Skewness': skew(feature_data),
        'Kurtosis': kurtosis(feature_data),
        'Highest Z-Score': highest_z,
        'Lowest Z-Score': lowest_z,
        'Avg Z-Score Top 300': top_300_average,
        'Avg Z-Score Bottom 300': bottom_300_average
    }
    numerical_summary.append(summary)

# Create DataFrame for numerical features
numerical_summary_df = pd.DataFrame(numerical_summary)

# Categorical Features
categorical_features = df.select_dtypes(include=[object]).columns.union(['phase', 'Phase'])
for feature in categorical_features:
    feature_data = df[feature].dropna()
    mode = feature_data.mode()[0] if not feature_data.mode().empty else None
    value_counts = feature_data.value_counts(normalize=True) * 100
    
    summary = {
        'Feature': feature,
        'Unique Values': feature_data.nunique(),
        'Mode': mode,
        'Distribution': value_counts.to_dict()
    }
    categorical_summary.append(summary)

# Create DataFrame for categorical features
categorical_summary_df = pd.DataFrame(categorical_summary)

# Print summaries
print("Numerical Feature Summary")
print(numerical_summary_df.to_string(index=False))

print("\nCategorical Feature Summary")
for summary in categorical_summary:
    print(f"\nFeature: {summary['Feature']}")
    print(f"Unique Values: {summary['Unique Values']}")
    print(f"Mode: {summary['Mode']}")
    print("Distribution:")
    for value, percentage in summary['Distribution'].items():
        print(f"  {value}: {percentage:.2f}%")

# Import additional necessary library
import matplotlib.pyplot as plt
from pathlib import Path

# Set up directory paths
main_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/raw"
plots_dir_quantiles = "C:/Users/reicd/Downloads/MY_ML_PROJECT/plots/feature quantile label split"
plots_dir_histograms = "C:/Users/reicd/Downloads/MY_ML_PROJECT/plots/histograms"

# Create directories if they don't exist
Path(plots_dir_quantiles).mkdir(parents=True, exist_ok=True)
Path(plots_dir_histograms).mkdir(parents=True, exist_ok=True)

# Read the data
df_path = os.path.join(main_dir, "final_combined_data_with_metadata.csv")
df = pd.read_csv(df_path)

# Ensure all columns are numbers for calculation
df = df.apply(pd.to_numeric, errors='ignore')

# Remove categorical columns
numeric_features = df.columns.difference(['phase', 'Phase', 'Subject', 'Story'])

def plot_feature_quantile_split(df, numeric_features, plots_dir_quantiles):
    num_features = len(numeric_features)
    grid_size_quantiles = 5  # Number of features per grid for quantile plots
    num_grids_quantiles = int(np.ceil(num_features / grid_size_quantiles))

    for i in range(num_grids_quantiles):
        selected_features = numeric_features[i*grid_size_quantiles:(i+1)*grid_size_quantiles]
        num_selected = len(selected_features)

        # Create figure for the current grid
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))  # Adjust 1x5 grid
        ax = ax.flatten()

        for j, feature in enumerate(selected_features):
            # Compute quartiles
            quantiles = df[feature].quantile([0, 0.25, 0.50, 0.75, 1.0])
            bins = [quantiles[0], quantiles[0.25], quantiles[0.50], quantiles[0.75], quantiles[1.0]]

            # Format bin labels with 4 decimal places
            bin_labels = [f"{x:.4f}" for x in bins]

            # Bin data
            df['quantile_bin'] = pd.cut(df[feature], bins=bins, labels=bin_labels[:-1], include_lowest=True)

            # Calculate label distribution proportions
            label_counts = df.groupby(['quantile_bin', 'phase']).size()
            bin_totals = label_counts.groupby(level=0).sum()
            label_proportions = (label_counts / bin_totals).unstack().fillna(0)

            # Plot stacked bar chart
            label_proportions.plot(
                kind='bar', stacked=True, ax=ax[j], alpha=0.75, width=0.8
            )
            ax[j].set_title(feature)
            ax[j].set_xlabel('Quartile Range')
            ax[j].set_ylabel('Proportion within Quartile')
            ax[j].legend(title='Phase', bbox_to_anchor=(1, 1))

        # Hide any unused subplots
        for j in range(num_selected, len(ax)):
            ax[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plots_dir_quantiles, f"feature_quantile_split_grid_{i+1}.png"), bbox_inches='tight')
        plt.close()

def plot_feature_histograms(df, numeric_features, plots_dir_histograms):
    num_features = len(numeric_features)
    grid_size_histograms = 8  # Number of features per grid for histograms
    num_grids_histograms = int(np.ceil(num_features / grid_size_histograms))

    for i in range(num_grids_histograms):
        selected_features = numeric_features[i*grid_size_histograms:(i+1)*grid_size_histograms]
        num_selected = len(selected_features)

        # Create figure for the current grid
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))  # Adjust 2x4 grid
        ax = ax.flatten()

        for j, feature in enumerate(selected_features):
            # Plot histogram
            ax[j].hist(df[feature].dropna(), bins=30, alpha=0.75)
            ax[j].set_title(feature)
            ax[j].set_xlabel('Value')
            ax[j].set_ylabel('Frequency')

        # Hide any unused subplots
        for j in range(num_selected, len(ax)):
            ax[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plots_dir_histograms, f"feature_histograms_grid_{i+1}.png"), bbox_inches='tight')
        plt.close()

# Execute the plotting functions
plot_feature_quantile_split(df, numeric_features, plots_dir_quantiles)
plot_feature_histograms(df, numeric_features, plots_dir_histograms)

# Import additional necessary libraries
from scipy.stats import zscore
import numpy as np

# Set up directory paths
main_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/raw"
output_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/processed"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the data
df_path = os.path.join(main_dir, "final_combined_data_with_metadata.csv")
df = pd.read_csv(df_path)

# Ensure all columns are treated as numeric where applicable
df = df.apply(pd.to_numeric, errors='ignore')

# Prepare data for outlier analysis
numeric_features = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(zscore(df[numeric_features].fillna(0)))

# Count outlier features per record
outlier_feature_count = (z_scores > 3).sum(axis=1)

# Datasets after dropping based on outlier thresholds
datasets = {
    "3_or_more_outliers_dropped": df[outlier_feature_count < 3],
    "2_or_more_outliers_dropped": df[outlier_feature_count < 2],
    "all_outliers_dropped": df[outlier_feature_count == 0]
}

# Save the datasets to the processed data folder
for key, dataset in datasets.items():
    dataset_path = os.path.join(output_dir, f"{key}.csv")
    dataset.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

# Display sizes of the new datasets
for key, dataset in datasets.items():
    print(f"Dataset '{key}' size: {dataset.shape}")

# Adjust imports as needed
import seaborn as sns

# Set up directory paths
main_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/raw"
processed_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/processed"
output_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/plots/CorrelationMatrices"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the original data
df_path = os.path.join(main_dir, "final_combined_data_with_metadata.csv")
df = pd.read_csv(df_path)

# Exclude non-numerical columns and prepare data for outlier analysis
numerical_columns = df.select_dtypes(include=[np.number]).columns.difference(['phase', 'Phase', 'Subject', 'Story'])
z_scores = np.abs(zscore(df[numerical_columns].fillna(0)))

# Count outlier features per record
outlier_feature_count = (z_scores > 3).sum(axis=1)

# Create datasets based on outlier thresholds
datasets = {
    "3_or_more_outliers_dropped": df[outlier_feature_count < 3],
    "2_or_more_outliers_dropped": df[outlier_feature_count < 2],
    "all_outliers_dropped": df[outlier_feature_count == 0]
}

# Function to plot correlation matrix
def plot_correlation_matrix(dataframe, title, save_path_png, save_path_pdf):
    # Calculating correlation matrix
    corr = dataframe.corr()

    # Plotting correlation matrix
    plt.figure(figsize=(12, 10))
    sns.set(style='white')
    
    ax = sns.heatmap(
        corr, annot=True, fmt=".2f", cmap='coolwarm',
        annot_kws={"size": 5},  # Adjust font size of numbers
        cbar_kws={"shrink": .8}, vmin=-1, vmax=1, center=0
    )
    ax.set_title(title)
    
    # Save the plot as PNG and PDF
    plt.tight_layout()
    plt.savefig(save_path_png, bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()

# Generate correlation matrices for all datasets
for name, dataset in datasets.items():
    processed_data = dataset[numerical_columns]
    plot_title = f'Correlation Matrix ({name.replace("_", " ").title()})'
    plot_save_path_png = os.path.join(output_dir, f'{name}_correlation_matrix.png')
    plot_save_path_pdf = os.path.join(output_dir, f'{name}_correlation_matrix.pdf')
    
    plot_correlation_matrix(processed_data, plot_title, plot_save_path_png, plot_save_path_pdf)

    print(f"Correlation matrix plot saved to {plot_save_path_png} and {plot_save_path_pdf}")
# Import additional necessary library
from sklearn.preprocessing import StandardScaler

# Paths to narrowed datasets and prepared data directory
narrowed_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/narrowed"
prepared_dir = "C:/Users/reicd/Downloads/MY_ML_PROJECT/data/prepared"
os.makedirs(prepared_dir, exist_ok=True)

# Dataset files
files = [
    "narrowed_3_or_more_outliers_dropped.csv",
    "narrowed_2_or_more_outliers_dropped.csv",
    "narrowed_all_outliers_dropped.csv"
]

# Function to process each file
def process_dataset(file_path, filename):
    # Read the data
    df = pd.read_csv(file_path)

    # Drop rows with NA values
    df.dropna(inplace=True)

    # Drop 'phase' column if exists, keep 'Phase'
    if 'phase' in df.columns:
        df.drop(columns=['phase'], inplace=True)
    
    # Identify categorical columns
    categorical_columns = ['Subject', 'Story', 'Phase']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Standardize numerical features
    numerical_columns = df_encoded.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])
    
    # Save the prepared dataset
    save_path = os.path.join(prepared_dir, f"prepared_{filename}")
    df_encoded.to_csv(save_path, index=False)
    print(f"Prepared dataset saved to {save_path}")

# Process each narrowed dataset
for file in files:
    file_path = os.path.join(narrowed_dir, file)
    process_dataset(file_path, file)

print("Data preprocessing and preparation complete.")


# Paths to prepared datasets and columns for feature engineering
import os

# Paths to prepared datasets
file_paths = [
    r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\prepared\prepared_narrowed_all_outliers_dropped.csv",
    r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\prepared\prepared_narrowed_2_or_more_outliers_dropped.csv",
    r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\prepared\prepared_narrowed_3_or_more_outliers_dropped.csv"
]

# Column names in lowercase
columns = ['sy', 'sz', 'timestamp', 'lwz', 'rwz', 'rhx', 'rwy', 'rhy', 'sx', 'hx', 'hz', 'lhx', 'lhy', 'lwx', 'lhz']

# Create a new directory for engineered data
output_directory = r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\engineered data"
os.makedirs(output_directory, exist_ok=True)

# Function to engineer features from dataframe
def engineer_features(df):
    """Function to engineer new features from existing DataFrame."""
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            # Ensure both columns exist in the dataframe
            if col1 in df.columns and col2 in df.columns:
                # Create product
                product_col_name = f'{col1}_{col2}_product'
                df[product_col_name] = df[col1] * df[col2]
                
                # Create sum
                sum_col_name = f'{col1}_{col2}_sum'
                df[sum_col_name] = df[col1] + df[col2]
                
                # Create difference
                diff_col_name = f'{col1}_{col2}_difference'
                df[diff_col_name] = df[col1] - df[col2]
    return df

# Process each file
for file_path in file_paths:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Engineer new features
    df_engineered = engineer_features(df)
    
    # Create output file path
    output_file_path = os.path.join(output_directory, os.path.basename(file_path))
    
    # Save the new DataFrame to CSV
    df_engineered.to_csv(output_file_path, index=False)

print("Feature engineering completed and files saved in 'engineered data' directory.")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier

# Define file paths for the datasets
file_paths = [
    r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\engineered data\prepared_narrowed_all_outliers_dropped.csv",
    r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\engineered data\prepared_narrowed_2_or_more_outliers_dropped.csv",
    r"C:\Users\reicd\Downloads\MY_ML_PROJECT\data\engineered data\prepared_narrowed_3_or_more_outliers_dropped.csv"
]

# Create directories for saving results
plots_dir = r"C:\Users\reicd\Downloads\MY_ML_PROJECT\plots\F1_score_comparison"
os.makedirs(plots_dir, exist_ok=True)

results_dir = r"C:\Users\reicd\Downloads\MY_ML_PROJECT\tests\f1_scores_default_table"
os.makedirs(results_dir, exist_ok=True)

# Define the models to evaluate
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "SVM": SVC(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_jobs=-1, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Perceptron": Perceptron(random_state=42)
}

# Initialize results dictionary for storing F1 scores
f1_scores = {model_name: [] for model_name in models.keys()}

# Define the macro F1 scorer
f1_scorer = make_scorer(f1_score, average='macro')

# Evaluate models on each dataset using cross-validation
for file_path in file_paths:
    df = pd.read_csv(file_path)
    target_column = 'Phase'
    
    # Automatic transformation of the target variable
    X = df.drop(columns=target_column)
    y = df[target_column] - 1

    for model_name, model in models.items():
        # Using cross_val_score for automatic handling
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer, n_jobs=-1)
        f1_scores[model_name].append(cv_scores.mean())

# Create a DataFrame for results
f1_scores_df = pd.DataFrame(f1_scores, index=[
    "all_outliers_removed",
    "2_or_more_outliers_removed",
    "3_or_more_outliers_removed"
])

# Save F1 scores table
table_file_path = os.path.join(results_dir, "f1_scores_macro_cross_val_comparison.csv")
f1_scores_df.to_csv(table_file_path)

print(f"F1 scores with cross-validation saved at: {table_file_path}")

# Descriptive labels for the datasets
dataset_labels = [
    "all_outliers_removed",
    "2_or_more_outliers_removed",
    "3_or_more_outliers_removed"
]

# Plotting the results (side by side)
plt.figure(figsize=(14, 8))
bar_width = 0.2

# Create positions for each dataset's group of bars
positions = np.arange(len(models))
colors = ['#c6dbef', '#6baed6', '#08306b']

for i, (dataset_label, color) in enumerate(zip(dataset_labels, colors)):
    new_positions = positions + i * bar_width
    bars = plt.bar(new_positions, f1_scores_df.iloc[i], width=bar_width, label=dataset_label, color=color, alpha=0.7)
    
    # =======================
    # Adding data labels on
    # each bar in the plot 
    # with percentage format 
    # and two decimal places.
    # You can adjust the font 
    # size by modifying the value
    # in `fontsize` as needed.
    # =======================
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height * 100:.2f}%',  # Convert to percentage
            ha='center', 
            va='bottom',
            fontsize=10  # Adjust fontsize as needed
        )

# =======================
# Utilize descriptive
# labels for each dataset
# to ensure the plot's legend
# accurately reflects the
# source and context of the
# data used for training the
# models. This enhances readability
# and precision in presentation.
# =======================
plt.title('F1 Macro Score Across Models and Datasets')
plt.ylabel('F1 Macro Score')
plt.xlabel('Model')
plt.xticks(positions + bar_width, f1_scores_df.columns, rotation=45)
plt.legend(title='Datasets')  # The legend is setup with descriptive dataset labels
plt.ylim(0, 1)
plt.tight_layout()

# Save the plot
plot_file_path = os.path.join(plots_dir, "f1_score_macro_comparison.png")
plt.savefig(plot_file_path)
plt.show()

import joblib
import os

# Define the directory to save models
models_dir = r"C:\Users\reicd\Downloads\MY_ML_PROJECT\models\default"
os.makedirs(models_dir, exist_ok=True)

# Train and save each model
for file_path in file_paths:
    df = pd.read_csv(file_path)
    target_column = 'Phase'

    # Prepare features and target
    X = df.drop(columns=target_column)
    y = df[target_column] - 1

    for model_name, model in models.items():
        # Fit the model
        model.fit(X, y)
        
        # Save the trained model
        model_file_path = os.path.join(models_dir, f"{model_name.replace(' ', '_')}.joblib")
        joblib.dump(model, model_file_path)
        print(f"Model {model_name} saved at: {model_file_path}")



import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import numpy as np

# Load the dataset
file_path = r"C:\Users\Administrator\Documents\ML PROJECT\DATA\prepared_narrowed_all_outliers_dropped.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=['Phase'])
y = df['Phase'] - 1

# Define parameter grids based on kernel type
param_grid = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf', 'sigmoid'], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}
]

# Set the metric for evaluation
scoring = 'f1_macro'

# To store the results
svm_results = {
    "grid_search_time": None,
    "grid_search_f1": None,
    "random_search_time": None,
    "random_search_f1": None,
    "bayes_search_time": None,
    "bayes_search_f1": None
}

# Grid Search
start_time = time.time()
grid_search = GridSearchCV(SVC(), param_grid, scoring=scoring, cv=5, n_jobs=-1)
grid_search.fit(X, y)
svm_results["grid_search_time"] = time.time() - start_time
svm_results["grid_search_f1"] = grid_search.best_score_

# Random Search
random_iter = int(0.6 * sum(len(v) for d in param_grid for v in d.values()))
start_time = time.time()
random_search = RandomizedSearchCV(SVC(), param_distributions=param_grid, n_iter=random_iter, scoring=scoring, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X, y)
svm_results["random_search_time"] = time.time() - start_time
svm_results["random_search_f1"] = random_search.best_score_

# Bayes Search
start_time = time.time()
bayes_search = BayesSearchCV(SVC(), param_grid, n_iter=random_iter, scoring=scoring, cv=5, n_jobs=-1, random_state=42)
bayes_search.fit(X, y)
svm_results["bayes_search_time"] = time.time() - start_time
svm_results["bayes_search_f1"] = bayes_search.best_score_

# Create and print a table with results
results_df = pd.DataFrame({
    'Search Method': ['Grid Search', 'Random Search', 'Bayes Search'],
    'Time (seconds)': [svm_results["grid_search_time"], svm_results["random_search_time"], svm_results["bayes_search_time"]],
    'F1 Score': [svm_results["grid_search_f1"], svm_results["random_search_f1"], svm_results["bayes_search_f1"]]
})

print(results_df)

# Plotting the results: Time
plt.figure(figsize=(12, 6))
plt.bar(['Grid Search', 'Random Search', 'Bayes Search'], 
        [svm_results["grid_search_time"], svm_results["random_search_time"], svm_results["bayes_search_time"]],
        color=['#c6dbef', '#6baed6', '#08306b'])
plt.xlabel('Search Method')
plt.ylabel('Time (seconds)')
plt.title('SVM Hyperparameter Search: Time')
plt.xticks(rotation=45)
for i, v in enumerate([svm_results["grid_search_time"], svm_results["random_search_time"], svm_results["bayes_search_time"]]):
    plt.text(i, v + max(svm_results["grid_search_time"], svm_results["random_search_time"], svm_results["bayes_search_time"]) * 0.01, f"{v:.2f}s", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Plotting the results: F1 Score
plt.figure(figsize=(12, 6))
plt.bar(['Grid Search', 'Random Search', 'Bayes Search'], 
        [svm_results["grid_search_f1"], svm_results["random_search_f1"], svm_results["bayes_search_f1"]],
        alpha=0.5, color=['#fdbf6f', '#e31a1c', '#ff7f00'])
plt.xlabel('Search Method')
plt.ylabel('F1 Score')
plt.title('SVM Hyperparameter Search: F1 Score')
plt.xticks(rotation=45)
for i, v in enumerate([svm_results["grid_search_f1"], svm_results["random_search_f1"], svm_results["bayes_search_f1"]]):
    plt.text(i, v + max(svm_results["grid_search_f1"], svm_results["random_search_f1"], svm_results["bayes_search_f1"]) * 0.01, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import numpy as np

# Load the dataset
file_path = r"C:\Users\Administrator\Documents\ML PROJECT\DATA\prepared_narrowed_all_outliers_dropped.csv"
df = pd.read_csv(file_path)
X = df.drop(columns=['Phase'])
y = df['Phase'] - 1

# Define an expanded parameter grid
param_grid = [
    {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 
        'max_iter': [500, 1000, 1500, 2000],
        'tol': [1e-4, 1e-3, 1e-2]
    }
]

# Set the metric for evaluation
scoring = 'f1_macro'

# To store the results
perceptron_results = {
    "grid_search_time": None,
    "grid_search_f1": None,
    "random_search_time": None,
    "random_search_f1": None,
    "bayes_search_time": None,
    "bayes_search_f1": None
}

# Grid Search
start_time = time.time()
grid_search = GridSearchCV(Perceptron(), param_grid, scoring=scoring, cv=5, n_jobs=-1)
grid_search.fit(X, y)
perceptron_results["grid_search_time"] = time.time() - start_time
perceptron_results["grid_search_f1"] = grid_search.best_score_

# Random Search
random_iter = 20  # Increase number of iterations for wider coverage
start_time = time.time()
random_search = RandomizedSearchCV(Perceptron(), param_distributions=param_grid, n_iter=random_iter, scoring=scoring, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X, y)
perceptron_results["random_search_time"] = time.time() - start_time
perceptron_results["random_search_f1"] = random_search.best_score_

# Bayes Search
start_time = time.time()
bayes_search = BayesSearchCV(Perceptron(), param_grid, n_iter=random_iter, scoring=scoring, cv=5, n_jobs=-1, random_state=42)
bayes_search.fit(X, y)
perceptron_results["bayes_search_time"] = time.time() - start_time
perceptron_results["bayes_search_f1"] = bayes_search.best_score_

# Plotting the results: Time
plt.figure(figsize=(12, 6))
plt.bar(['Grid Search', 'Random Search', 'Bayes Search'], 
        [perceptron_results["grid_search_time"], perceptron_results["random_search_time"], perceptron_results["bayes_search_time"]],
        color=['#c6dbef', '#6baed6', '#08306b'])
plt.xlabel('Search Method')
plt.ylabel('Time (seconds)')
plt.title('Perceptron Hyperparameter Search: Time')
plt.xticks(rotation=45)
for i, v in enumerate([perceptron_results["grid_search_time"], perceptron_results["random_search_time"], perceptron_results["bayes_search_time"]]):
    plt.text(i, v + 0.5, f"{v:.2f}s", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Plotting the results: F1 Score
plt.figure(figsize=(12, 6))
plt.bar(['Grid Search', 'Random Search', 'Bayes Search'], 
        [perceptron_results["grid_search_f1"], perceptron_results["random_search_f1"], perceptron_results["bayes_search_f1"]],
        alpha=0.5, color=['#fdbf6f', '#e31a1c', '#ff7f00'])
plt.xlabel('Search Method')
plt.ylabel('F1 Score')
plt.title('Perceptron Hyperparameter Search: F1 Score')
plt.xticks(rotation=45)
for i, v in enumerate([perceptron_results["grid_search_f1"], perceptron_results["random_search_f1"], perceptron_results["bayes_search_f1"]]):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()