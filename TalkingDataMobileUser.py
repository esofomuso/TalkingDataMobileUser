# -*- coding: utf-8 -*-
"""
## Methodology
  1. Data collection
  2. Exploratory Data Analysis (EDA) & Class balancing
  3. Supervised Learning
    1. SelectKbest
      1. GridsearchCV with all models (RandomForest, SVM, KNN).
      2. Evaluate using classification report, confusion matrix, cross-validation
    2. PCA
      1. GridsearchCV with all models (RandomForest, SVM, KNN).
      2. Evaluate using classification report, confusion matrix, cross-validation
    3. Write up analysis and comparison
  4. Unsupervised Learning Approach
    1. Using PCA and K-means to do the clustering (visualizations and many clusters, justify using silhouette scores)
    2. For PCA/K-means (K>= 4), draw the bar graphs to answer the following questions:
      1. Which cluster contains the most usage by females between 27 and 32?
      2. Which cluster contains the most device variety by males between 29 and 38?
    3. Using t-SNE and GMM to do the clustering (visualizations and many clusters, justify using silhouette scores)
    4. For t-SNE/GMM (K>= 4), draw the bar graphs to answer the following questions:
      1. Which cluster contains the most usage by females between 27 and 32?
      2. Which cluster contains the most device variety by males between 29 and 38?
    5. Write up the comparison analysis.
  5. Deep Learning
    1. Try different ANN models and train them on the training set with the following:
      1. Number of layers
      2. Activation functions of the layers
      3. Number of neurons in the layers
      4. Different batch sizes during training
    2. CNN & RNN
      1. with unbalanced dataset show classification report
      2. with balanced dataset show classification report
    3. Use the best performing model from supervised learning together with CNN and RNN
    3. Compare the models' training scores and interpret the results.
    4. Evaluate how the models perform on the test set. Compare the results of the models.
  6. Conclusions and recommendations

## 1. Data Collection

### Import necessary Libraries
"""

# Import necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from statsmodels.tools.eval_measures import mse, rmse
from scipy.spatial import distance
import scipy
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

import time
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from google.colab import drive
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow import keras


import warnings

warnings.filterwarnings('ignore')

"""### Set up environment"""

# Set up environment
# 1. Data Collection:
import os

drive.mount('/content/drive')

# Directory where main Capstone files are located
capstone_dir = '/content/drive/My Drive/Data Science/Capstone4'

# Directory where all CSV files are located
csv_dir = f'{capstone_dir}/data'

# Get all CSV files in the directory
all_csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]
print(f"Total number of CSV files containing 'Talkingdata Mobile User Demographics' data: {len(all_csv_files)}")

# Create an empty list to store individual dataframes
# list_of_dataframes = []
datasets = {}

"""##2. Exploratory data analysis (EDA)

###1. Data exploration & Feature Engineering

####1.1 Load the raw CSV files to create datasets
"""

# Step 1: load the datasets
# Loop through all CSV files and read them into individual dataframes
for csv_file in all_csv_files:
    filepath = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(filepath)
    key = csv_file.replace(".csv", "_data")
    datasets[key] = df
    print("---------------")
    print(f"{key} has the following rows and columns: {df.shape} ")
    print(df.head())
    df.info()

"""####1.2 Merge the datasets"""

# Step 2: Merge the datasets
merged_data = pd.merge(datasets['events_data'], datasets['phone_brand_device_model_data'], on='device_id', how='inner')
merged_data = pd.merge(merged_data, datasets['gender_age_train_data'], on='device_id', how='inner')
# merged_data = pd.merge(merged_data, datasets['label_categories_data'], on='label_id', how='inner')
merged_data.info()
merged_data.head()

print(merged_data.isnull().sum())

"""####1.3 Data Cleaning & Processing



"""

# Step 3: Data Cleaning and Preprocessing
# no null entries so proceed to duplicates.
#Handling Duplicate Entries
merged_data.drop_duplicates(inplace=True)

#Convert the 'timestamp' column to a datetime data type and adding more datetime fields:
merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])
merged_data['date'] = merged_data['timestamp'].dt.date
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['time'] = merged_data['timestamp'].dt.time
merged_data['day'] = merged_data['timestamp'].dt.day

#Memory Optimization
merged_data['device_id'] = merged_data['device_id'].astype('int32')
merged_data['age'] = merged_data['age'].astype('int8')
merged_data['day'] = merged_data['day'].astype('int8')
merged_data.info()
merged_data.head()

# Step 4: Creating the age_group column
# Creating age groups based on the age column.
def age_group(age):
    if age < 20:
        return '19-'
    elif 20 <= age < 30:
        return '20+'
    elif 30 <= age < 40:
        return '30+'
    elif 40 <= age < 50:
        return '40+'
    else:
        return '50+'

# Applying the function to create a new age group column
merged_data['age_group'] = merged_data['age'].apply(age_group)

# Step 5: Checking the counts of each new age group
print(gender_age_data['age_group'].value_counts())

# Create 'group' field by concatenating 'gender' and 'age_group'
gender_age_data['group'] = gender_age_data['gender'] + gender_age_data['age_group'].astype(str)

"""####1.4. Class balancing and Store to CSV for subsequent uses"""

# Encode categorical columns
label_encoder = LabelEncoder()
gender_age_data['is_male'] = label_encoder.fit_transform(gender_age_data['gender'])
gender_age_data['phone_brand_id'] = label_encoder.fit_transform(gender_age_data['phone_brand'])
gender_age_data['device_model_id'] = label_encoder.fit_transform(gender_age_data['device_model'])
gender_age_data['age_group_id'] = label_encoder.fit_transform(gender_age_data['age_group'])

#extract features like year, month, day, etc., from the timestamp
gender_age_data['year'] = gender_age_data['date'].dt.year
gender_age_data['month'] = gender_age_data['date'].dt.month
gender_age_data['day'] = gender_age_data['date'].dt.day

# Now drop the original 'date' column
gender_age_data = gender_age_data.drop(columns=['date'])

gender_age_data['time'] = pd.to_datetime(gender_age_data['time']).dt.time
#extract separate features from the time object
gender_age_data['hour'] = gender_age_data['time'].apply(lambda x: x.hour)
gender_age_data['minute'] = gender_age_data['time'].apply(lambda x: x.minute)
gender_age_data['second'] = gender_age_data['time'].apply(lambda x: x.second)

#drop time
gender_age_data = gender_age_data.drop(columns=['time'])

#Memory Optimization
gender_age_data['device_id'] = gender_age_data['device_id']
gender_age_data['age'] = gender_age_data['age']
# gender_age_data['day'] = gender_age_data['day'].astype('int8')

gender_age_data.info()
print(gender_age_data.head())

gender_age_data['group_id'] = label_encoder.fit_transform(gender_age_data['group'])

gender_age_data.info()
gender_age_data.head()

gender_age_data.to_csv(f'{capstone_dir}/full_talking_data_mobile_user.csv', index=False)

"""####1.5 Create a subset and Store to CSV for subsequent uses"""

#Create a subset of the data
subset_gender_age_data = gender_age_data.sample(frac=0.1, random_state=1)

# Save the subset_gender_age_data dataset to a CSV file (optional)
subset_gender_age_data.to_csv(f'{capstone_dir}/talking_data_mobile_user.csv', index=False)

subset_gender_age_data.info()
subset_gender_age_data.head()

"""####1.6 Visualize data of Selected features"""

# Creating pairplot to visualize relationships among the 'day', 'month', 'year', 'hour', 'minute', 'phone_brand_id', 'device_model_id' and group(the gender-age combo)
features = ['group', 'day', 'month', 'year', 'hour', 'minute', 'phone_brand_id', 'device_model_id']
data = subset_gender_age_data[features]

sns.pairplot(data, hue='group')
plt.show()

"""###2. Read cleaned data from CSV in all subsequent uses"""

# From here, since the clean dataset is already created, always start here when the runtime times out.
# Since Data is already created, start from here going forward
#1. Load the dataset and conduct any necessary preprocessing, such as normalizing the data.

drive.mount('/content/drive')

full_talking_data_df = pd.read_csv(f'{capstone_dir}/full_talking_data_mobile_user.csv')
talking_data_df = pd.read_csv(f'{capstone_dir}/talking_data_mobile_user.csv')
print(f"The shape of the full talking data is: {full_talking_data_df.shape}")

full_talking_data_df.info()
print(full_talking_data_df.head())

print(f"The shape of the talking data is: {talking_data_df.shape}")
talking_data_df.info()
talking_data_df.head()

full_talking_data_df['age_group'].value_counts()

talking_data_df['age_group'].value_counts()

talking_data_df['group'].value_counts()

group_counts = talking_data_df['group'].value_counts().reset_index()
group_counts.columns = ['group', 'count']

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='group', data=group_counts, palette='viridis')

# Add title and labels
plt.title('Distribution of Groups', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Group', fontsize=12)

# Show plot
plt.show()

"""I will have to balance the data as there seems to be a class imbalance."""

research_features = ['group_id', 'day', 'month', 'year', 'hour', 'minute', 'phone_brand_id', 'device_model_id']
research_data = talking_data_df[research_features]
selected_features = ['day', 'month', 'year', 'hour', 'minute', 'phone_brand_id', 'device_model_id']

"""##3. Supervised Learning
    1. SelectKbest
      1. GridsearchCV with all models (RandomForest, SVM, KNN).
      2. Evaluate using classification report, confusion matrix, cross-validation
    2. PCA
      1. GridsearchCV with all models (RandomForest, SVM, KNN).
      2. Evaluate using classification report, confusion matrix, cross-validation
    3. Write up analysis and comparison

###3.1 Split data into features (X) and target variable(Y), then divide into train and test data, then visualize the correlation matrix
"""

#Create a new DataFrame X with just selected features
X = talking_data_df[selected_features]

# create target variable
y = talking_data_df['group_id']

#Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Map the correlation matrix of the data
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

correlation_matrix = X_train.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

"""### 3.2 Before using SelectKBest, train different classifiers and evaluate using the test set"""

#train different classifiers on the training set and evaluate them on the test set.
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

"""###3.3 SelectKBest

"""

selector = SelectKBest(score_func=f_classif, k='all')  # select all features to get their scores

"""####3.3.1 Feature Selection Using SelectKBest selector"""

# 1. Feature Selection
X_new = selector.fit_transform(X_train, y_train)

# Print the scores for each feature
for i, score in enumerate(selector.scores_):
    print(f"Feature {X_train.columns[i]}: {score}")

"""The scores represent the importance or relevance of each feature in predicting the target variable (group_id). A higher score indicates a higher relevance:

* day: 1.93 - This indicates that the "day" feature has a relatively low correlation with the target variable, based on its F-score.
* month: 0.60 - Similar to the "day" feature, the "month" feature also has a low correlation with the target variable.
* year: nan - The score is not a number, indicating that this feature probably has constant values, and thus, cannot be used in the classification. You might want to consider removing this feature from your dataset.
* hour: 9.31 - This feature has a somewhat higher correlation compared to "day" and "month", but it is still relatively low.
* minute: 1.60 - This feature again has a low correlation with the target variable, similar to "day" and "month".
* phone_brand_id: 61.63 - This feature has a higher F-score, indicating a higher correlation with the target variable. It seems like an important feature for your model.
* device_model_id: 29.18 - This feature has a moderate correlation with the target variable based on its F-score.
"""

X_train.info()

X_train['year'].unique()

"""year has only 1 value = 2016 so not useful.

Split the data again using only the relevant features
"""

#Recreate the DataFrame X with just selected features
selected_features1 = ['day', 'hour', 'minute', 'phone_brand_id', 'device_model_id']
X = talking_data_df[selected_features1]

# create target variable
y = talking_data_df['group_id']

#Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Feature Selection
selector = SelectKBest(score_func=f_classif, k='all')  # select all features to get their scores
X_new = selector.fit_transform(X_train, y_train)

# Print the scores for each feature
for i, score in enumerate(selector.scores_):
    print(f"Feature {X_train.columns[i]}: {score}")

X.info()

"""####3.3.2 GridSearchCV with different models
(RandomForest, SVM, KNN)

#####3.3.2.1 SelectKBest & RandomForest
"""

try:
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10]
    },
    cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for RandomForest: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for RandomForest:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for RandomForest:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for RandomForest:")
    print(cross_val_score(RandomForestClassifier(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model RandomForest: {e}")

"""**Best Parameters:**

The best parameters obtained from the grid search for the Random Forest model are as follows:

* max_depth: 30 - This suggests that allowing the trees in the forest to have a maximum depth of 30 gave the best performance during the cross-validation in the grid search.
* min_samples_split: 5 - This means that the minimum number of samples required to split an internal node is 5.
* n_estimators: 200 - This means that the best performance was obtained when using 200 trees in the forest.

**Classification Report:**

Classes have considerably varied precision and recall values, indicating that the model is having difficulty in distinguishing between the different classes. The precision and recall for class 0 are particularly low, which suggests that the model is struggling to correctly identify this class.


**Cross-Validation Scores:**

[0.31214161 0.3101471  0.3191622 ] The cross-validation scores are consistent but low, indicating that the model is not overfitting but is not performing well either.

######3.3.2.1.0 Balancing with SMOTE and using 40% with the best parameters
"""

y.value_counts()

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
y_train_balanced.value_counts()

X_train_balanced.shape
X_train_balanced.info()

# Using only 40% of the balanced data
Xtuned_train, X_discard, ytuned_train, y_discard = train_test_split(
    X_train_balanced,
    y_train_balanced,
    test_size=0.6,  # Discarding 60%
    random_state=42,
    stratify=y_train_balanced
)
# Using only 40% of the X_discard data
_, Xtuned_test, _, ytuned_test = train_test_split(
    X_discard,
    y_discard,
    test_size=0.4,
    random_state=42,
    stratify=y_discard
)

#Fit this SelectKBest instance
Xtuned_new = selector.fit_transform(Xtuned_train, ytuned_train)
# Print the scores for each feature
for i, score in enumerate(selector.scores_):
    print(f"Feature {Xtuned_train.columns[i]}: {score}")

"""######3.3.2.1.1 RandomForest redone with SMOTE balanced - split- tuned X and y"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [200],
      'max_depth': [20],
      'min_samples_split': [5]
    },
    cv=5, n_jobs=-1)

    grid.fit(Xtuned_train, ytuned_train)

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test))
    print(f"\nClassification Report for RandomForest:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for RandomForest:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for RandomForest:")
    print(cross_val_score(RandomForestClassifier(), Xtuned_test, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model RandomForest: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""#####3.3.2.2 SelectKBest & K-Nearest Neighbour"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }, cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for KNN: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for KNN:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""The **KNN** performance is similar to the RandomForest in that the recall for is_male = 0 indicates imbalance. Rerun using the resampled data

######3.3.2.2.1 KNN with Best Parameters, balanced - split- tuned X and y
"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [7],
        'weights': ['distance']
    }, cv=5, n_jobs=-1)

    grid.fit(Xtuned_train, ytuned_train)

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test))
    print(f"\nClassification Report for KNN:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), Xtuned_test, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [7],
        'weights': ['distance']
    }, cv=5, n_jobs=-1)

    grid.fit(Xtuned_train, ytuned_train)
    print(f"Best parameters for KNN: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_discard))
    print(f"\nClassification Report for KNN:\n{classification_report(y_discard, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(y_discard, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), X_discard, y_discard, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""#####3.3.2.3 SelectKBest & Support Vector Machines (SVM)"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [0.1, 1, 10],
      'kernel': ['rbf']
    }, cv=5, n_jobs=-1)
    grid.fit(X_new, y_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for SVM:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""The results were mostly 0s for the above SVM & SelectKBest, so try re-sampling
Run with best params and tuned data
"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [10],
      'kernel': ['rbf']
    }, cv=5, n_jobs=-1)
    grid.fit(Xtuned_train, ytuned_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test))
    print(f"\nClassification Report for SVM:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), Xtuned_test, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""#####3.3.2.4
Balancing Data: One reason for the increased accuracy in the tuned Random Forest model could be due to the balancing of the data. Balancing the classes often leads to a better representation of each class, allowing the model to predict the minority class more accurately.

Tuned Parameters: The tuned parameters, especially increasing the n_estimators to 200, could have given the Random Forest more trees to average the results, potentially increasing accuracy. Also, setting the max_depth to 20 could have given enough depth for the trees to capture the intricate patterns in the data without overfitting.

Runtime: Random Forest took 120.9 seconds to run in the tuned scenario, which is expected due to the increased data from balancing and the higher n_estimators.

From this tabulated comparison, we see that both KNN and Random Forest achieve an accuracy of 58% after tuning. The Random Forest model provides feature importance, while KNN does not. On the other hand, KNN might be simpler and faster in certain situations.
"""

# Data
models = ['SVM', 'KNN', 'Random Forest']
accuracy_before = [30, 32, 42]
accuracy_after = [16, 58, 58]
f1_before = [21, 31, 29]
f1_after = [12, 58, 56]

# Setting up the figure and axis
fig, ax = plt.subplots(2, 1, figsize=(10, 15))

# Plotting accuracy
ax[0].bar(models, accuracy_before, width=0.4, align='center', label='Before Tuning', alpha=0.7)
ax[0].bar(models, accuracy_after, width=0.4, align='edge', label='After Tuning', alpha=0.7)
ax[0].set_title("Accuracy Comparison")
ax[0].set_ylabel("Accuracy (%)")
ax[0].legend()
ax[0].set_ylim(0, 70)  # Assuming accuracy is in percentage. Adjust this value if needed

# Plotting F1-score
ax[1].bar(models, f1_before, width=0.4, align='center', label='Before Tuning', alpha=0.7)
ax[1].bar(models, f1_after, width=0.4, align='edge', label='After Tuning', alpha=0.7)
ax[1].set_title("F1-Score Comparison")
ax[1].set_ylabel("F1-Score (%)")
ax[1].legend()
ax[1].set_ylim(0, 70)  # Assuming F1-score is in percentage. Adjust this value if needed

plt.tight_layout()
plt.show()

"""**SVM (Support Vector Machine)**

* **Before tuning:** The accuracy is low at 30%. The f1-scores for the classes are also relatively low, with a weighted average f1-score of 29%.
* **After tuning:** The accuracy increases significantly to 58%, and the weighted average f1-score also improves considerably to 58%.

**KNN (K-Nearest Neighbors)**

* **Before tuning:** The accuracy is slightly higher than SVM's initial performance at 32%. The weighted average f1-score is 31%, which is a little higher than SVM before tuning.
* **After tuning:** The accuracy again sees a significant increase to 58%, and the weighted average f1-score matches the accuracy at 58%.

**Random Forest**

* **Before tuning:** The accuracy for Random Forest before tuning is the highest among the three models at 42%. The f1-scores for the classes vary, with the weighted average f1-score at 39%. However, the macro average f1-score is only 29%, indicating disparities in the f1-scores across classes.
* **After tuning:** The accuracy remains consistent with the tuned SVM and KNN at 58%. The weighted average f1-score is 56%, which is a notable improvement from before tuning but slightly lower than SVM and KNN after tuning.

**Key Observations:**

1. **Model Improvement After Tuning:**
All three models demonstrated substantial improvement after tuning. This highlights the importance of hyperparameter tuning and using balanced data when training machine learning models.

2. **Consistency After Tuning:**
After the tuning process, all three models converged to similar accuracy and f1-scores. This indicates that the three models, when appropriately optimized, can offer comparable performance on this dataset.

3. **Random Forest's Initial Superiority:**
Before any tuning, Random Forest had the best initial performance among the three. This suggests that ensemble methods like Random Forest can naturally handle the complexity of certain datasets better than simpler models. However, with optimization, simpler models like SVM and KNN can catch up.

4. **Class Imbalance:**
The results show a clear disparity in the precision, recall, and f1-scores across different classes. This indicates class imbalance in the dataset, which can affect the performance of models. It's crucial to handle this imbalance, possibly using techniques like over-sampling, under-sampling, or the Synthetic Minority Over-sampling Technique (SMOTE).

5. **SelectKBest:**
The feature selection process seems to have been effective. Using SelectKBest likely streamlined the feature space, eliminating irrelevant or redundant features, and helped in enhancing model performance, especially after tuning.

**Recommendations:**

Model Selection: Given that all models have similar performance after tuning, model selection can be based on other factors such as interpretability, deployment considerations, and computational resources. If interpretability is essential, Random Forest might be a preferred choice due to its ability to rank feature importance.

**Further Exploration:**

As initially observed the data is imbalanced between the genders, balancing the genders and exploring other features could help.

#####3.3.2.4a Since the results were so poor we need to go back to the drawing board. Balance by gender, update the age-gender grouping, then change the feature selection and try again.

######3.3.2.4.1 Update Age-gender (group, and group_id) to create better balance
"""

# Creating age groups based on the age column.
def new_age_group(age):
    if age < 26:
        return '25-'
    elif 26 <= age < 30:
        return '26+'
    elif 30 <= age < 35:
        return '30+'
    elif 35 <= age < 42:
        return '35+'
    else:
        return '42+'

# Applying the function to create a new age group column
talking_data_df['age_group'] = talking_data_df['age'].apply(new_age_group)
label_encoder = LabelEncoder()
talking_data_df['age_group_id'] = label_encoder.fit_transform(talking_data_df['age_group'])
# Create 'group' field by concatenating 'gender' and 'age_group'
talking_data_df['group'] = talking_data_df['gender'] + talking_data_df['age_group'].astype(str)
talking_data_df['group_id'] = label_encoder.fit_transform(talking_data_df['group'])

# Applying the function to create a new age group column
full_talking_data_df['age_group'] = full_talking_data_df['age'].apply(new_age_group)
label_encoder = LabelEncoder()
full_talking_data_df['age_group_id'] = label_encoder.fit_transform(full_talking_data_df['age_group'])
# Create 'group' field by concatenating 'gender' and 'age_group'
full_talking_data_df['group'] = full_talking_data_df['gender'] + full_talking_data_df['age_group'].astype(str)
full_talking_data_df['group_id'] = label_encoder.fit_transform(full_talking_data_df['group'])

"""######3.3.2.4.1a Store the updated gender-groups"""

talking_data_df.to_csv(f'{capstone_dir}/talking_data_mobile_user.csv', index=False)

full_talking_data_df.to_csv(f'{capstone_dir}/full_talking_data_mobile_user.csv', index=False)

"""######3.3.2.4.2 Upsample the female data since it is significantly lower than the male."""

# Separate majority and minority classes
df_majority = talking_data_df[talking_data_df.is_male==1]
df_minority = talking_data_df[talking_data_df.is_male==0]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.is_male.value_counts()

talking_data_df.is_male.value_counts()

unbalanced_talking_data_df = talking_data_df
talking_data_df = df_upsampled

full_talking_data_df.is_male.value_counts()

# Separate majority and minority classes
full_df_majority = full_talking_data_df[full_talking_data_df.is_male==1]
full_df_minority = full_talking_data_df[full_talking_data_df.is_male==0]

# Upsample minority class
full_df_minority_upsampled = resample(full_df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=len(full_df_majority),    # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
full_df_upsampled = pd.concat([full_df_majority, full_df_minority_upsampled])

# Display new class counts
full_df_upsampled.is_male.value_counts()

full_talking_data_df = full_df_upsampled

"""######3.3.2.4.3 Visualize data balance"""

# Visualize Changes before saving to csv
group_counts = talking_data_df['group'].value_counts().reset_index()
group_counts.columns = ['group', 'count']

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='group', data=group_counts, palette='viridis')

# Add title and labels
plt.title('Distribution of Groups', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Group', fontsize=12)

# Show plot
plt.show()

"""######3.3.2.4.4 Store the balanced data for future runs"""

talking_data_df.to_csv(f'{capstone_dir}/talking_data_mobile_user_balanced.csv', index=False)

full_talking_data_df.to_csv(f'{capstone_dir}/full_talking_data_mobile_user_balanced.csv', index=False)

"""#####3.3.2.4.5 Read rebalanced data from file each time I come back for faster continuation"""

# From here, since the clean dataset is already balanced.
# Since Data is already created, start from here going forward
# Load the dataset and conduct any necessary preprocessing

drive.mount('/content/drive')

unbalanced_full_talking_data_df = pd.read_csv(f'{capstone_dir}/full_talking_data_mobile_user.csv')
unbalanced_talking_data_df = pd.read_csv(f'{capstone_dir}/talking_data_mobile_user.csv')
full_talking_data_df = pd.read_csv(f'{capstone_dir}/full_talking_data_mobile_user_balanced.csv')
talking_data_df = pd.read_csv(f'{capstone_dir}/talking_data_mobile_user_balanced.csv')
print(f"The shape of the full talking data is: {full_talking_data_df.shape} and unbalanced data: {unbalanced_talking_data_df.shape}")

print('--------- unbalanced_full_talking_data_df.info ---------')
unbalanced_full_talking_data_df.info()
print('--------- full_talking_data_df.info ---------')
full_talking_data_df.info()
print('--------- unbalanced_talking_data_df.info ---------')
unbalanced_talking_data_df.info()
print('--------- -------- ---------')

print(f"The shape of the talking data is: {talking_data_df.shape}")
talking_data_df.info()
talking_data_df.head()

unbalanced_talking_data_df.head()

"""######3.3.2.4.6 Reselect features then run Supervised learning models again"""

research_features = ['device_id', 'group_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
research_data = talking_data_df[research_features]
selected_features = ['device_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']

"""#####3.3.2.5 Re-split data

"""

#Create a new DataFrame X with just selected features
X = talking_data_df[selected_features]

# create target variable
y = talking_data_df['group_id']

#Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.42, random_state=42)

# Feature Selection
X_new = selector.fit_transform(X_train, y_train)

# Print the scores for each feature
for i, score in enumerate(selector.scores_):
    print(f"Feature {X_train.columns[i]}: {score}")

X_new.shape

"""######3.3.2.5.1 Re-run SelectKBest & Random Forest"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10]
    },
    cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for RandomForest: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for RandomForest:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for RandomForest:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for RandomForest:")
    print(cross_val_score(RandomForestClassifier(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model RandomForest: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""######3.3.2.5.2 Re-run SelectKBest & KNN"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [7],
        'weights': ['distance']
    }, cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for KNN: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for KNN:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""######3.3.2.5.3 Re-run SelectKBest & SVM"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [10],
      'kernel': ['rbf']
    }, cv=5, n_jobs=-1)
    grid.fit(X_new, y_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for SVM:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""######3.3.2.5.4 Scale X then Re-run SelectKBest & SVM"""

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
X_new_scaled = selector.fit_transform(X_train_scaled, y_train)

X_new_scaled.shape

try:
    # Starting a timer
    start_time = time.time()

    grid = GridSearchCV(SVC(),
    {
      'C': [10],
      'kernel': ['rbf'],  # Remove 'linear' to speed up the process
    },
    cv=StratifiedKFold(n_splits=3),  # Reduce number of splits for cross-validation
    n_jobs=-1)

    grid.fit(X_new_scaled, y_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test_scaled))
    print(f"\nClassification Report for SVM on X_new_scaled:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), selector.transform(X_test_scaled), y_test, cv=3))

except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")

finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

# Scale the data (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into a subset (using stratified sampling to maintain class distribution)
X_subset, _, y_subset, _ = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Split the subset into training and testing sets
SVM_X_train, SVM_X_test, SVM_y_train, SVM_y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# Feature selection using SelectKBest
SVM_X_new = selector.fit_transform(SVM_X_train, SVM_y_train)

SVM_X_new.shape

try:
    # Starting a timer
    start_time = time.time()

    grid = GridSearchCV(SVC(),
    {
      'C': [10],
      'kernel': ['rbf'],  # Remove 'linear' to speed up the process
    },
    cv=StratifiedKFold(n_splits=3),  # Reduce number of splits for cross-validation
    n_jobs=-1)

    grid.fit(SVM_X_new, SVM_y_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    SVM_y_pred = grid.predict(selector.transform(SVM_X_test))
    print(f"\nClassification Report for SVM:\n{classification_report(SVM_y_test, SVM_y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(SVM_y_test, SVM_y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), selector.transform(SVM_X_test), SVM_y_test, cv=3))

except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")

finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
y_train_balanced.value_counts()
# Using only 40% of the balanced data
Xtuned_train, X_discard, ytuned_train, y_discard = train_test_split(
    X_train_balanced,
    y_train_balanced,
    test_size=0.6,  # Discarding 60%
    random_state=42,
    stratify=y_train_balanced
)
# Using only 40% of the X_discard data
_, Xtuned_test, _, ytuned_test = train_test_split(
    X_discard,
    y_discard,
    test_size=0.4,
    random_state=42,
    stratify=y_discard
)

Xtuned_train.shape
Xtuned_train.info()
Xtuned_test.shape
Xtuned_test.info()

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [10],
      'kernel': ['rbf']
    }, cv=5, n_jobs=-1)
    grid.fit(Xtuned_train, ytuned_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test))
    print(f"\nClassification Report for SVM:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), Xtuned_test, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

# Define the model names and their scores
models = ['Random Forest', 'KNN', 'SVM']
accuracy = [0.85, 0.97, 0.13]
precision = [0.85, 0.97, 0.06]
recall = [0.85, 0.97, 0.12]
f1_score = [0.85, 0.97, 0.07]
runtime = [2249.88, 10.65, 7810.23]

# Create bar charts
barWidth = 0.25
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(12, 8))

# Create bars
plt.bar(r1, accuracy, width=barWidth, label='Accuracy')
plt.bar(r2, precision, width=barWidth, label='Precision (Macro Avg)')
plt.bar(r3, recall, width=barWidth, label='Recall (Macro Avg)')
plt.bar(r4, f1_score, width=barWidth, label='F1-Score (Macro Avg)')

# Title & subtitle
plt.title('Comparison of Model Performances')
plt.xlabel('Models', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(models))], models)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Now, let's create a bar chart for runtime
plt.figure(figsize=(8, 6))
plt.bar(models, runtime, color=['green', 'blue', 'red'])
plt.title('Comparison of Model Runtimes')
plt.ylabel('Seconds')
plt.show()

"""#####3.3.2.6
#####Random Forest
######Advantages:

Relatively high accuracy of 85%.
Consistent cross-validation scores around 76% indicating the model isn't overfitting significantly.
Given the importance of device_id, day, hour, phone_brand_id, and device_model_id, a Random Forest can capture intricate patterns and non-linearities among these variables.
######Limitations:

The runtime is quite long at approximately 2250 seconds.
Some classes like 5, 6, 7, 8, and 9 have relatively lower precision and recall than the other classes, indicating possible imbalances in class distribution even after upsampling or that the features are not as discriminative for these classes.
#####k-Nearest Neighbors (kNN)
######Advantages:

Exceptionally high accuracy of 97%, which is impressive.
The confusion matrix indicates a very high true positive rate for almost every class, with very few misclassifications.
Significantly faster than Random Forest with a runtime of about 11 seconds.
######Limitations:

kNN can be sensitive to noise in the data or irrelevant features. Given the high performance, it seems that the feature engineering has led to clean, relevant features for the model.
kNN models can be slow for predictions in real-time scenarios due to the need to compute distances to all training samples. However, given the fast runtime, this might not be a significant concern in this case.
#####Support Vector Machine (SVM)
######Advantages:

SVMs are powerful classifiers that can capture high-dimensional patterns.
######Limitations:

Extremely poor performance with an accuracy of just 13%. The confusion matrix and the classification report both show that the SVM model struggles to classify most of the classes correctly.
SVM took the longest time at approximately 7810 seconds, making it both time-consuming and ineffective for this dataset.
The rbf kernel may have led to overfitting, given the many dimensions in my data.
#####Recommendations:
* kNN is the Winner: Based on accuracy, precision, recall, and F1-score, the kNN classifier clearly outperforms the other models. If prediction time isn't a major concern, it's recommended to use the kNN classifier with the best parameters you've found (n_neighbors: 7, weights: 'distance').

* Potential Overfitting with kNN: Despite its performance on the validation set, ensure that kNN isn't overfitting the training data. It's always a good practice to keep a separate test set to evaluate the model's performance before deploying.

* Random Forest Tuning: If you still want to pursue Random Forest (due to its interpretability or any other reason), consider further hyperparameter tuning, exploring feature importance, and possibly engineering more features or removing ones that don't provide much discriminatory power.

* Avoid SVM for This Dataset: Given its poor performance and long runtime, it might not be worth pursuing SVM further for this particular problem and dataset.

* Consider Deep Learning: Given more computational resources and if dealing with a huge amount of data, consider trying deep learning models like neural networks. They can potentially capture even more intricate patterns in the data, especially if there are hidden relationships that simple models can't capture.

* Evaluate on Test Set: Finally, once I finalized a model, I'll evaluate it on a separate test set to get an unbiased estimate of its performance.


"""

from sklearn.metrics import accuracy_score
grid = GridSearchCV(KNeighborsClassifier(), {
    'n_neighbors': [7],
    'weights': ['distance']
}, cv=5, n_jobs=-1)

grid.fit(X_new, y_train)

# Predict on training data
train_predictions = grid.predict(X_train)

train_accuracy = accuracy_score(y_train, train_predictions)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {0.97}")  # Your previous kNN validation accuracy

# If the training accuracy is very close to 1 and there's a significant drop in validation accuracy, be cautious of overfitting.

"""####3.3.3 Stratified y_train (Intentionally named to come after PCA)
After working on PCA I realized that the y_train is imbalanced, it didn't maiantain the balanced data. Could there be overfitting in the above results?
I will try stratified y so as to maintain the same class distribution in both training and test sets.
"""

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, stratify=y)

X_train1.shape

y_train.value_counts()

y_train1.value_counts()

# Feature Selection
X_new1 = selector.fit_transform(X_train1, y_train1)

# Print the scores for each feature
for i, score in enumerate(selector.scores_):
    print(f"Feature {X_train1.columns[i]}: {score}")

"""####3.3.4 Re-split with stratified y_train to shape: (119894, 5)
SelectKBest and all 3 models
"""

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

params = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'KNN': {
        'n_neighbors': [7],
        'weights': ['distance']
    },
    'SVM': {
        'C': [10],
        'kernel': ['rbf']
    }
}

# Use GridSearch on the re-sampled subset data for all 3 models
for name, model in models.items():
    try:
        # Starting a timer
        start_time = time.time()
        grid = GridSearchCV(model, params[name],
        cv=StratifiedKFold(n_splits=3),  # Reduce number of splits for cross-validation
        n_jobs=-1)
        grid.fit(X_new1, y_train1)
        print(f"Best parameters for {name} - SelectKBest: {grid.best_params_}")

        # 3. Evaluation
        y_pred = grid.predict(selector.transform(X_test1))
        print(f"\nClassification Report for {name} - SelectKBest:\n{classification_report(y_test1, y_pred)}")
        print(f"Confusion Matrix for {name}:\n{confusion_matrix(y_test1, y_pred)}\n\n")
        print(f"Cross-validation scores for {name}:")
        print(cross_val_score(model, X_test1, y_test1, cv=3))
    except Exception as e:
        print(f"An error occurred while processing model {name}: {e}")
    finally:
        # Printing the total runtime
        end_time = time.time()
        print(f"Total runtime: {end_time - start_time} seconds")

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [10],
      'kernel': ['rbf']
    }, cv=5, n_jobs=-1)
    grid.fit(X_new1, y_train1)

    # Evaluation
    y_pred1 = grid.predict(selector.transform(X_test1))
    print(f"\nClassification Report for SVM and SelectKBest with stratified y:\n{classification_report(y_test1, y_pred1)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(y_test1, y_pred1)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), X_test1, y_test1, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

"""###3.4 PCA"""

X_train.info()

talking_data_df['group'].value_counts()

y_train.value_counts()

y.value_counts()

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
X_new_scaled = selector.fit_transform(X_train_scaled, y_train)

# Apply PCA to Standardized and scaled data
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_train_pca.shape

# Standardize the data
X_train_scaled1 = scaler.fit_transform(X_train1)
X_test_scaled1 = scaler.transform(X_test1)

# Feature Selection
X_new_scaled1 = selector.fit_transform(X_train_scaled1, y_train1)

# Apply PCA to Standardized and scaled data
X_train_pca1 = pca.fit_transform(X_train_scaled1)
X_test_pca1 = pca.transform(X_test_scaled1)

"""####3.4.1 RandomForest & PCA"""

try:
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10]
    },
    cv=5, n_jobs=-1)

    grid.fit(X_train_pca, y_train)
    print(f"Best parameters for RandomForest: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test_pca))
    print(f"\nClassification Report for PCA & RandomForest:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for RandomForest:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for RandomForest:")
    print(cross_val_score(RandomForestClassifier(), X_test_pca, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model RandomForest: {e}")

"""####3.4.2 SVM & PCA"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [0.1, 1, 10],
      'kernel': ['rbf']
    },
    cv=StratifiedKFold(n_splits=3),  # Reduce number of splits for cross-validation
    n_jobs=-1)
    grid.fit(X_train_pca, y_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test_pca))
    print(f"\nClassification Report for PCA & SVM:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), X_test_pca, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total start_time: {start_time}")
    print(f"Total end_time: {end_time}")
    print(f"Total runtime: {end_time - start_time} seconds")

"""####3.4.3 KNN & PCA"""

try:
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }, cv=5, n_jobs=-1)

    grid.fit(X_train_pca, y_train)
    print(f"Best parameters for KNN: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test_pca))
    print(f"\nClassification Report for KNN:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), X_test_pca, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")

"""####3.4.4 Apply PCA to tuned data"""

# Standardize the data
scaler = StandardScaler()
Xtuned_train_scaled = scaler.fit_transform(Xtuned_train)
Xtuned_test_scaled = scaler.transform(Xtuned_test)

# Apply PCA
pca = PCA(n_components=5)
Xtuned_train_pca = pca.fit_transform(Xtuned_train_scaled)
Xtuned_test_pca = pca.transform(Xtuned_test_scaled)

#Best parameters for RandomForest: {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 200}
try:
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [200],
      'max_depth': [30],
      'min_samples_split': [2]
    },
    cv=5, n_jobs=-1)

    grid.fit(Xtuned_train_pca, ytuned_train)
    print(f"Best parameters for RandomForest: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test_pca))
    print(f"\nClassification Report for RandomForest:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for RandomForest:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for RandomForest:")
    print(cross_val_score(RandomForestClassifier(), Xtuned_test_pca, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model RandomForest: {e}")

"""#####SVM & PCA to Best parameters, tuned data

"""

#Best parameters for SVM: {'C': 10, 'kernel': 'rbf'}
try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(SVC(), {
      'C': [10],
      'kernel': ['rbf']
    },
    cv=StratifiedKFold(n_splits=3),  # Reduce number of splits for cross-validation
    n_jobs=-1)
    grid.fit(Xtuned_train_pca, ytuned_train)
    print(f"Best parameters for SVM: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test_pca))
    print(f"\nClassification Report for PCA & SVM using Best parameters, tuned and balanced data:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for SVM:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for SVM:")
    print(cross_val_score(SVC(), Xtuned_test_pca, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model SVM: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"start_time: {start_time}")
    print(f"end_time: {end_time}")
    print(f"Total runtime: {end_time - start_time} seconds")

"""#####KNN & PCA to Best parameters, tuned data"""

#Best parameters for KNN: {'n_neighbors': 7, 'weights': 'distance'}
try:
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [7],
        'weights': ['distance']
    }, cv=5, n_jobs=-1)

    grid.fit(Xtuned_train_pca, ytuned_train)

    # Evaluation
    y_pred = grid.predict(selector.transform(Xtuned_test_pca))
    print(f"\nClassification Report for KNN:\n{classification_report(ytuned_test, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(ytuned_test, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), Xtuned_test_pca, ytuned_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")

"""####3.4.5 One more attempt of PCA on all 3 models using stratified y_train and shape: (119894, 5)"""

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

params = {
    'RandomForest': {
        'n_estimators': [100],
        'max_depth': [30],
        'min_samples_split': [2]
    },
    'KNN': {
        'n_neighbors': [3],
        'weights': ['distance']
    },
    'SVM': {
        'C': [10],
        'kernel': ['rbf']
    }
}

# Use GridSearch on the full data for all 3 models
for name, model in models.items():
    try:
        # Starting a timer
        start_time = time.time()
        grid = GridSearchCV(model, params[name],
        cv=StratifiedKFold(n_splits=3),  # Reduce number of splits for cross-validation
        n_jobs=-1)
        grid.fit(X_train_pca1, y_train1)
        print(f"Best parameters for {name} - PCA: {grid.best_params_}")

        # 3. Evaluation
        y_pred = grid.predict(selector.transform(X_test_pca1))
        print(f"\nClassification Report for {name} - PCA with stratified split data:\n{classification_report(y_test1, y_pred)}")
        print(f"Confusion Matrix for {name}:\n{confusion_matrix(y_test1, y_pred)}\n\n")
        print(f"Cross-validation scores for {name}:")
        print(cross_val_score(model, X_test_pca1, y_test1, cv=3))
    except Exception as e:
        print(f"An error occurred while processing model {name}: {e}")
    finally:
        # Printing the total runtime
        end_time = time.time()
        print(f"Total runtime: {end_time - start_time} seconds")

"""* **RandomForest (RF):**
* **Accuracy: 0.71**
* **Macro F1-Score: 0.70**

The RandomForest classifier has the best performance among the three classifiers.
The confusion matrix indicates that the model can distinguish between classes reasonably well, but it's struggling more with classes 5 through 9.
Cross-validation scores are around 0.48, which suggests that there may be some overfitting since the performance on the validation set is considerably better than on the cross-validation sets.
* **K-Nearest Neighbors (KNN):**
* **Accuracy: 0.69**
* **Macro F1-Score: 0.68**

The performance of KNN is slightly worse than RF, but the difference isn't too significant.
Similar to RF, KNN struggles with classes 5 through 9.
Cross-validation scores are much lower (around 0.31) compared to the performance on the validation set, suggesting a high degree of overfitting.
* **Support Vector Machine (SVM):**
* **Accuracy: 0.22**
* **Macro F1-Score: 0.20**

SVM's performance is substantially lower than the other two classifiers.
The confusion matrix suggests that the SVM model is having difficulty distinguishing between most classes, often misclassifying one class as another.
Cross-validation scores are consistent with the performance on the validation set, indicating less overfitting than RF and KNN but overall worse performance.
Additionally, SVM took a much longer time to run (over 3300 seconds) compared to the other two classifiers.

**Conclusion**

Comparison with Older PCA Results:
Comparing the results with the older PCA model results on RandomForest:

The accuracy has improved from 70% to 71%.
The precision, recall, and F1-score remain consistent.
Conclusions:

RandomForest with PCA performs the best in terms of accuracy and F1-score among the three models. While it has some challenges with classes 5 through 9, its overall performance is commendable.

KNN with PCA is the fastest model and gives decent results, but its performance is slightly inferior to RandomForest.

SVM with PCA is the slowest and has the worst performance among the three. Given its long runtime and low accuracy, it may not be suitable for this dataset.

###3.5 Gradient Boosting
The Gradient Boosting algorithm is a powerful ensemble technique that constructs a series of decision trees, where each tree tries to correct the mistakes of the previous one. The GradientBoostingClassifier from sklearn is a popular implementation of this algorithm.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Create and fit the model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=42)
clf.fit(X_train1, y_train1)

# Predict on test data
y_pred1 = clf.predict(X_test1)

# Measure the accuracy
accuracy = accuracy_score(y_test1, y_pred1)
print(f"Accuracy of Gradient Boosting attempt: {accuracy:.4f}")

"""## 4. Unsupervised Learning Approach
    1. Using PCA and K-means to do the clustering (visualizations and many clusters, justify using silhouette scores)
    2. For PCA/K-means (K>= 4), draw the bar graphs to answer the following questions:
      1. Which cluster contains the most usage by females between 27 and 32?
      2. Which cluster contains the most device variety by males between 29 and 38?
    3. Using t-SNE and GMM to do the clustering (visualizations and many clusters, justify using silhouette scores)
    4. For t-SNE/GMM (K>= 4), draw the bar graphs to answer the following questions:
      1. Which cluster contains the most usage by females between 27 and 32?
      2. Which cluster contains the most device variety by males between 29 and 38?
    5. Write up the comparison analysis.

### 4.1. Using PCA and K-means to do the clustering (visualizations and many clusters, justify using silhouette scores)
"""

research_features = ['device_id', 'group_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
research_data = talking_data_df[research_features]
selected_features = ['device_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']

research_data.shape

research_data['group_id'].value_counts()

unbalanced_talking_data_df['group'].value_counts()

unbalanced_talking_data_df.shape

print(research_features)

X = unbalanced_talking_data_df[research_features]

Supervised_X_scaled = X_scaled # preserve the old X_scaled in case I need it later.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled.shape

# Range of K values to consider
k_values = range(4, 11)

# Calculate silhouette scores for each K value
print("Silhouette Scores for K-means on scaled data")
silhouette_scores = []
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    inertia_values.append(kmeans.inertia_)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")

# Find the best K value using silhouette score
best_silhouette_score = max(silhouette_scores)
best_k = k_values[np.argmax(silhouette_scores)]
print(f"The best K value based on silhouette score: {best_k} with silhouette score of: {best_silhouette_score}")

# The "elbow method" can be used to visually inspect the best k
# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.show()

# Perform K-means clustering with the best K value
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
best_labels = best_kmeans.fit_predict(X_scaled)

# Plot using best_labels from K-means clustering
plt.figure(figsize=(10, 5))
colours = ["r", "b", "g", "c", "m", "y", "k", "r", "burlywood", "chartreuse"]
for i, label in enumerate(best_labels):
    plt.text(X_scaled[i, 0], X_scaled[i, 1], str(label),
             color=colours[int(label)],
             fontdict={'weight': 'bold', 'size': 12}
        )
plt.title('Kmeans clustering of TalkingData Mobile User Dataset')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

"""#####The plot shows little or no separation, the labels mostly sit on each other. Labels 0 and 2 are separated from the others but the are not clustered together looks like a lot of outliers exist.
####Apply PCA to reduce dimensionality to the scaled data then apply K-means again.
"""

# Apply PCA to reduce dimensionality
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Calculate silhouette scores for each K value on PCA data
print("Silhouette Scores for K-means with PCA on scaled data")
pca_silhouette_scores = []
pca_inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, labels)
    pca_silhouette_scores.append(silhouette_avg)
    pca_inertia_values.append(kmeans.inertia_)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.2f}")

# Find the best K value on PCA data
pca_best_silhouette_score = max(pca_silhouette_scores)
pca_best_k = k_values[np.argmax(pca_silhouette_scores)]
print(f"The best PCA K value based on silhouette score: {pca_best_k} with silhouette score of: {pca_best_silhouette_score}")

"""#####Some improvement in the Silhouette value from 0.171 to 0.361"""

# Perform K-means clustering with the best K value
pca_best_kmeans = KMeans(n_clusters=pca_best_k, random_state=42)
pca_best_labels = pca_best_kmeans.fit_predict(X_pca)

# The "elbow method" can be used to visually inspect the best k on PCA
# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, pca_inertia_values, marker='o')
plt.title('Elbow Method for Optimal K on PCA')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.show()

"""#####The best silouette score improved after PCA was applied to the scaled data. Also the Best K is closer to what the Elbow method revealed. A score of 0.36 suggests that the clusters are somewhat distinct, but there might be room for improvement in terms of separation between clusters.

Now, we proceed with visualization to interpret the characteristics of these clusters and understand their meaning in the context of the dataset and research question.

#### Plot the data for PCA/K-means

"""

# Plot using pca_best_labels from K-means clustering 09/24/23 most recent
plt.figure(figsize=(10, 5))
colours = ["r", "b", "g", "c", "m", "y", "k", "r", "burlywood", "chartreuse"]
for i, label in enumerate(pca_best_labels):
    plt.text(X_pca[i, 0], X_pca[i, 1], str(label),
             color=colours[int(label)],
             fontdict={'weight': 'bold', 'size': 12}
        )
plt.title('PCA (ticks) of TalkingData Mobile User Dataset')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show()

"""The graph shows clusters 0, 2, 3, and 4 nicely clustered together, while 1 is a little scattered but grouped together, though not much gap separating the clusters from each other. Also clusters 3 and 4 have some outliers far off from the rest of the clusters.

###4.2 For PCA/K-means (K>= 4), draw the bar graphs to answer the following questions:
  1. Which cluster contains the most usage by females between 27 and 32?
  2. Which cluster contains the most device variety by males between 29 and 38?
"""

unbalanced_talking_data_df.info()

# Filter data for PCA/K-means (K = 5) results
pca_kmeans_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_kmeans_df['cluster'] = pca_best_labels
pca_kmeans_df['age'] = unbalanced_talking_data_df['age']
pca_kmeans_df['gender'] = unbalanced_talking_data_df['gender']

# Calculate total female Count between 27 and 32
females = pca_kmeans_df[(pca_kmeans_df['gender'] == 'F')]
female_27_32 = females[(females['age'] >= 27) & (females['age'] <= 32)]

female_cluster_counts = female_27_32['cluster'].value_counts()
max_female_cluster = female_cluster_counts.idxmax()
# Plot bar graphs
plt.figure(figsize=(12, 6))
female_cluster_counts.plot(kind='bar', color='blue')
plt.title('PCA/K-means: Total Females between 27 and 32 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Female between 27 & 32 Count')
plt.xticks(rotation=0)

"""####Sub Questions
####4.2.1. Which cluster contains the most usage by females between 27 and 32?
      
"""

# Print the result
print(f"PCA/K-means: Cluster {max_female_cluster} has the most usage by females between 27 and 32 with a total count of {female_cluster_counts[max_female_cluster]}")

"""####4.2.2. Which cluster contains the most device variety by males between 29 and 38?
  
"""

# Calculate total delay minutes for each cluster during rush hour on weekdays (assuming rush hour is from 7 AM to 9 AM)
# Filtering for weekdays (0-4)
pca_kmeans_df['device_model_id'] = unbalanced_talking_data_df['device_model_id']

male_29_38 = pca_kmeans_df[(pca_kmeans_df['gender'] == 'M') & (pca_kmeans_df['age'] >= 29) & (pca_kmeans_df['age'] <= 38)]
device_varieties = male_29_38.groupby('cluster')['device_model_id'].nunique()
max_device_variety_cluster = device_varieties.idxmax()

# Plot bar graphs
plt.figure(figsize=(12, 6))
device_varieties.plot(kind='bar', color='green')
plt.title('PCA/K-means: Total Device Varieties for Men between 29 and 38 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Total Device Variety')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Print the result
print(f"PCA/K-means: Cluster {max_device_variety_cluster} has the most Device Varieties for Men between 29 and 38 with a total of {device_varieties[max_device_variety_cluster]}.")

"""Silhouette scores are useful for assessing the quality of clustering when ground truth labels are not available. In this case, I determined that the best K value for the PCA-transformed data is 5, with a silhouette score of approximately 0.36.

This indicates that, according to the silhouette score, the data points in the PCA space are reasonably well-clustered into 5 distinct clusters. Keeping in mind that the silhouette score is just one metric, and it's also important to visually inspect the resulting clusters to ensure they make sense and align with the domain knowledge.

Next, let's proceed with applying GMM clustering and visualize the clusters using the t-SNE visualization.

###4.3. Using t-SNE and GMM to do the clustering (visualizations and many clusters, justify using silhouette scores)
  
"""

research_features = ['device_id', 'group_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
research_data = talking_data_df[research_features]
selected_features = ['device_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
X = unbalanced_talking_data_df[research_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the best number of components (clusters) for GMM using silhouette score
best_gmm_score = -1
best_gmm_components = -1
for n_components in range(4, 11):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"For n_clusters = {n_components}, the silhouette score is {score:.3f}")
    if score > best_gmm_score:
        best_gmm_score = score
        best_gmm_components = n_components


print(f"TalkingData: Best number of components for GMM: {best_gmm_components} with score: {best_gmm_score}")

# Perform GMM clustering with the best number of components
best_gmm = GaussianMixture(n_components=best_gmm_components, random_state=42)
gmm_labels = best_gmm.fit_predict(X_scaled)

import plotly.express as px

# Create a DataFrame for visualization
vis_df = pd.DataFrame({'PC1': X_scaled[:, 0], 'PC2': X_scaled[:, 1], 'Cluster': gmm_labels})

# Create an interactive scatter plot using Plotly
fig = px.scatter(vis_df, x='PC1', y='PC2', color='Cluster', title='Using Gaussian Mixture Model Visualization of TalkingData Mobile User')
fig.show()

"""####The plot shows little or no separation, the clusters mostly sit on each other.

###Apply t-SNE to reduce dimensionality of the scaled data then apply K-means again.
"""

# Apply t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Determine the best number of components (clusters) for t-SNE/GMM using silhouette score
tsne_best_gmm_score = -1
tsne_best_gmm_components = -1
print("Silhouette Scores for scaled data with t-SNE/GMM")
for n_components in range(4, 11):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X_tsne)
    score = silhouette_score(X_tsne, labels)
    print(f"For n_clusters = {n_components}, the silhouette score is {score:.3f}")
    if score > tsne_best_gmm_score:
        tsne_best_gmm_score = score
        tsne_best_gmm_components = n_components


print(f"TalkingData: Best number of components for t-SNE/GMM: {tsne_best_gmm_components} with silhouette score: {tsne_best_gmm_score}")

"""####The best silhouette score improved significantly from - to - after t-SNE was applied to the scaled data. A score of --- suggests that the clusters are distinct, but there is room for improvement in terms of separation between clusters."""

# Perform t-SNE/GMM clustering with the best number of components
tsne_best_gmm = GaussianMixture(n_components=tsne_best_gmm_components, random_state=42)
tsne_gmm_labels = tsne_best_gmm.fit_predict(X_tsne)

import plotly.express as px

# Create a DataFrame for visualization
vis_df = pd.DataFrame({'PC1': X_tsne[:, 0], 'PC2': X_tsne[:, 1], 'Cluster': tsne_gmm_labels})

# Create an interactive scatter plot using Plotly
fig = px.scatter(vis_df, x='PC1', y='PC2', color='Cluster', title='t-SNE Clustering using Gaussian Mixture Model - TalkingData Mobile User')
fig.show()

"""###The t-SNE/GMM plot looks nicely separated with none of the 6 clusters sitting on the other.

###4.4. For t-SNE/GMM (K>= 4), draw the bar graphs to answer the following questions:
  1. Which cluster contains the most usage by females between 27 and 32?
  2. Which cluster contains the most device variety by males between 29 and 38?

#### Sub Questions
####4.4.1. Which cluster contains the most usage by females between 27 and 32?
"""

# Extract relevant columns from the original dataset and GMM labels
gmm_cluster_df = pd.DataFrame({'cluster': tsne_gmm_labels})

gmm_cluster_df['device_model_id'] = unbalanced_talking_data_df['device_model_id']
gmm_cluster_df['age'] = unbalanced_talking_data_df['age']
gmm_cluster_df['gender'] = unbalanced_talking_data_df['gender']

# Calculate total female Count between 27 and 32
females = gmm_cluster_df[(gmm_cluster_df['gender'] == 'F')]
female_27_32 = females[(females['age'] >= 27) & (females['age'] <= 32)]

female_cluster_counts = female_27_32['cluster'].value_counts()
max_female_cluster = female_cluster_counts.idxmax()
# Plot bar graphs
plt.figure(figsize=(12, 6))
female_cluster_counts.plot(kind='bar', color='blue')
plt.title('t-SNE/GMM: Total Females between 27 and 32 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Female between 27 & 32 Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(f"t-SNE/GMM: Cluster {max_female_cluster} has the most usage by females between 27 and 32 with a total count of {female_cluster_counts[max_female_cluster]}")

"""####4.4.2. Which cluster contains the most device variety by males between 29 and 38?

"""

# Filter data for males between 29 amd 38
male_29_38 = gmm_cluster_df[(gmm_cluster_df['gender'] == 'M') & (gmm_cluster_df['age'] >= 29) & (gmm_cluster_df['age'] <= 38)]
device_varieties = male_29_38.groupby('cluster')['device_model_id'].nunique()
max_device_variety_cluster = device_varieties.idxmax()

# Plot bar graphs
plt.figure(figsize=(12, 6))
device_varieties.plot(kind='bar', color='green')
plt.title('t-SNE/GMM: Total Device Varieties for Men between 29 and 38 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Total Device Variety')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


print(f"t-SNE/GMM: Cluster {max_device_variety_cluster} has the most Device Varieties for Men between 29 and 38 with a total of {device_varieties[max_device_variety_cluster]}.")

"""##5. Deep Learning
    1. Try different ANN models and train them on the training set with the following:
      1. Number of layers
      2. Activation functions of the layers
      3. Number of neurons in the layers
      4. Different batch sizes during training
      5. Comparing Training Scores
    2. CNN & RNN
      1. with unbalanced dataset show classification report
      2. with balanced dataset show classification report
      3. using best SelectKBest Random Forest Model (the best model so far)
    3. Compare the models' training scores and interpret the results.
    4. Evaluate how the models perform on the test set. Compare the results of the models.

### 5.1 Artificial Neural Network (ANN)
"""

# Step 1:  Building and Training different ANN models
selected_features = ['device_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
X = unbalanced_talking_data_df[selected_features]
y = unbalanced_talking_data_df['group_id']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

x_train.shape

x_train.info()

x_test.shape

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Model 1
model1 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model1.fit(x_train, y_train_encoded, epochs=10, batch_size=64, validation_data=(x_test, y_test_encoded))
test_loss1, test_acc1 = model1.evaluate(x_test, y_test_encoded)
print(f'Test accuracy1: {test_acc1}, Test loss1: {test_loss1}')


# Model 2
model2 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(x_train, y_train_encoded, epochs=10, batch_size=128, validation_data=(x_test, y_test_encoded))
test_loss2, test_acc2 = model2.evaluate(x_test, y_test_encoded)
print(f'Test accuracy2: {test_acc2}, Test loss2: {test_loss2}')


# Model 3
model3 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(10, activation='softmax')
])
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history3 = model3.fit(x_train, y_train_encoded, epochs=10, batch_size=64, validation_data=(x_test, y_test_encoded))
test_loss3, test_acc3 = model3.evaluate(x_test, y_test_encoded)
print(f'Test accuracy3: {test_acc3}, Test loss3: {test_loss3}')


# Model 4
model4 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history4 = model4.fit(x_train, y_train_encoded, epochs=10, batch_size=512, validation_data=(x_test, y_test_encoded))
test_loss4, test_acc4 = model4.evaluate(x_test, y_test_encoded)
print(f'Test accuracy4: {test_acc4}, Test loss4: {test_loss4}')


# Model 5
model5 = Sequential([
    layers.Dense(32, activation='relu', input_shape=(5,)),  # Input shape matches x_train shape.
    layers.Dense(10, activation='softmax')  # 10 neurons in the output layer for 10 classes
])

model5.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history5 = model5.fit(x_train, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2)
test_loss5, test_acc5 = model5.evaluate(x_test, y_test_encoded)
print(f'Test accuracy5: {test_acc5}, Test loss5: {test_loss5}')

print("Model 1 Training Accuracy:", max(history1.history['accuracy']))
print("Model 2 Training Accuracy:", max(history2.history['accuracy']))
print("Model 3 Training Accuracy:", max(history3.history['accuracy']))
print("Model 4 Training Accuracy:", max(history4.history['accuracy']))
print("Model 5 Training Accuracy:", max(history5.history['accuracy']))

"""####5.1.5: Comparing Training Scores
Test accuracy1: 0.1004570946097374, Test loss1: 2998315.0

Test accuracy2: 0.1554955393075943, Test loss2: 2.2069268226623535

Test accuracy3: 0.15485143661499023, Test loss3: 2.212794542312622

Test accuracy4: 0.14217743277549744, Test loss4: 3103369.5

Test accuracy5: 0.13106170296669006, Test loss5: 1534474.75



####Analysis:

#####Accuracy:

The highest accuracy is achieved by Model2 (15.54%), closely followed by Model3 (15.48%).
Model4 and Model5 are slightly less accurate, with accuracies of 14.21% and 13.10%, respectively.
Model1 has the lowest accuracy at 10.04%.

#####Loss:

Model2 and Model3 have extremely low losses, near 2.2, which is vastly lower than the other models.
Model1 and Model4 have high losses in the range of 3 million.
Model5's loss is around 1.5 million, which is lower than Model1 and Model4 but significantly higher than Model2 and Model3.

Given the results, Model2 appears to be the best-performing model considering both accuracy and loss. However, the dramatic difference in loss values between Model2, Model3, and the other models suggests a difference in the nature of these models

###Graphical Representation
"""

# Data
models = ['Model1', 'Model2', 'Model3', 'Model4', 'Model5']
accuracy = [0.14566798508167267, 0.1591522991657257, 0.15800955891609192, 0.13862456381320953, 0.10768751055002213]
loss = [4252695.0, 2.203671932220459, 2.207550048828125, 4487747.5, 1486651.25]

# Create a new figure with a specific size
plt.figure(figsize=(12, 5))

# Subplot for Accuracy
plt.subplot(1, 2, 1)
plt.bar(models, accuracy, color=['blue', 'green', 'red', 'cyan', 'yellow'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Model')

# Subplot for Loss
plt.subplot(1, 2, 2)
plt.bar(models, loss, color=['blue', 'green', 'red', 'cyan', 'yellow'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Model')

# Automatically adjust the layout
plt.tight_layout()

# Display the plots
plt.show()

"""#####Plot Model2 data (Best performing)"""

# Model2 Data
epochs = list(range(1, 11))
training_loss = [4853545.5000, 2020499.5000, 1088472.2500, 292186.0938, 48862.3242, 2.2332, 2.2173, 2.2107, 2.2083, 2.2075]
validation_loss = [2266588.7500, 1680652.5000, 399238.1250, 205401.1406, 2.2448, 2.2214, 2.2102, 2.2058, 2.2042, 2.2037]

training_accuracy = [0.1193, 0.1192, 0.1178, 0.1324, 0.1414, 0.1484, 0.1526, 0.1527, 0.1523, 0.1525]
validation_accuracy = [0.0995, 0.1319, 0.1386, 0.1116, 0.1458, 0.1528, 0.1528, 0.1540, 0.1540, 0.1592]

# Plotting Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss', marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

"""####Do same with scaled data"""

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model 1
model1 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model1.fit(x_train_scaled, y_train_encoded, epochs=10, batch_size=64, validation_data=(x_test_scaled, y_test_encoded))
test_loss1, test_acc1 = model1.evaluate(x_test_scaled, y_test_encoded)
print(f'Test accuracy1: {test_acc1}, Test loss1: {test_loss1}')

# Model 2
model2 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(x_train_scaled, y_train_encoded, epochs=10, batch_size=128, validation_data=(x_test_scaled, y_test_encoded))
test_loss2, test_acc2 = model2.evaluate(x_test_scaled, y_test_encoded)
print(f'Test accuracy2: {test_acc2}, Test loss2: {test_loss2}')

# Model 3
model3 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(10, activation='softmax')
])
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history3 = model3.fit(x_train_scaled, y_train_encoded, epochs=10, batch_size=64, validation_data=(x_test_scaled, y_test_encoded))
test_loss3, test_acc3 = model3.evaluate(x_test_scaled, y_test_encoded)
print(f'Test accuracy3: {test_acc3}, Test loss3: {test_loss3}')

# Model 4
model4 = Sequential([
    Flatten(input_shape=(5,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history4 = model4.fit(x_train_scaled, y_train_encoded, epochs=10, batch_size=512, validation_data=(x_test_scaled, y_test_encoded))
test_loss4, test_acc4 = model4.evaluate(x_test_scaled, y_test_encoded)
print(f'Test accuracy4: {test_acc4}, Test loss4: {test_loss4}')

# Model 5
model5 = Sequential([
    layers.Dense(32, activation='relu', input_shape=(5,)),  # Input shape matches x_train shape.
    layers.Dense(10, activation='softmax')  # 10 neurons in the output layer for 10 classes
])

model5.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history5 = model5.fit(x_train_scaled, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2)
test_loss5, test_acc5 = model5.evaluate(x_test_scaled, y_test_encoded)
print(f'Test accuracy5: {test_acc5}, Test loss5: {test_loss5}')

models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
test_loss = [2.1592, 2.1192, 2.1482, 2.1527, 2.1734]
test_accuracy = [0.1950, 0.2251, 0.2012, 0.2054, 0.1860]

barWidth = 0.3
r1 = np.arange(len(test_loss))
r2 = [x + barWidth for x in r1]

plt.bar(r1, test_loss, width=barWidth, color='blue', edgecolor='grey', label='test_loss')
plt.bar(r2, test_accuracy, width=barWidth, color='red', edgecolor='grey', label='test_accuracy')

# Description
plt.title('Comparison among 5 models with scaled data')
plt.xticks([r + barWidth for r in range(len(test_loss))], models)
plt.legend()

plt.show()

"""#### Comparing the Results on Scaled data
1. Test Loss:
The test loss is an indication of how well a model has learned to predict the given data. Lower values are preferable as they indicate smaller errors.

Here's the ranking based on test loss:

Model 2: 2.1192
Model 3: 2.1482
Model 4: 2.1527
Model 1: 2.1592
Model 5: 2.1734
From the test loss perspective, Model 2 has the lowest test loss, making it the best among the five in this aspect.

2. Test Accuracy:
Accuracy gives an overview of the percentage of correct predictions out of total predictions. Higher values are preferable as they indicate a higher proportion of correct predictions.

Here's the ranking based on test accuracy:

Model 2: 0.2251 (22.51%)
Model 4: 0.2054 (20.54%)
Model 3: 0.2012 (20.12%)
Model 1: 0.1950 (19.50%)
Model 5: 0.1860 (18.60%)
From the test accuracy perspective, Model 2 also has the highest accuracy, making it the best among the five in this regard as well.

#####Conclusion:
Comparing the models based on both test loss and test accuracy, Model 2 emerges as the best performing model among the five. It has the lowest test loss and the highest test accuracy.

On the other hand, Model 5 appears to be the weakest model as it ranks last in both test loss and test accuracy.
"""

x_train_reshaped = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_reshaped = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

# Building a simple CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Adjust the output layer based on your task (regression or classification)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Adjust based on your task

# Train the model
model.fit(x_train_reshaped, y_train, epochs=10, batch_size=64, validation_data=(x_test_reshaped, y_test))

# Normalize the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape data to have a channel dimension
x_train = x_train.reshape(x_train.shape[0], 5, 1, 1)
x_test = x_test.reshape(x_test.shape[0], 5, 1, 1)

# Build the CNN model
model = Sequential([
    Conv2D(32, (2,1), activation='relu', input_shape=(5, 1, 1)),
    MaxPooling2D((2,1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

unbalanced_talking_data_df.info()
unbalanced_talking_data_df.shape

"""###5.2.1  Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) with unbalanced data
1. Building the Model:
Use CNN layers to capture local patterns and RNN layers to capture sequences in the data.
2. Training the Model:
Split the dataset into training and testing subsets.
Train the model.
3. Evaluating the Model:
Predict on the test set.
Generate a classification report.
"""

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data to add channel dimension for CNN
X_train = X_train.reshape(X_train.shape[0], 5, 1)
X_test = X_test.reshape(X_test.shape[0], 5, 1)

# Build a CNN-RNN hybrid model
model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(5, 1)),
    MaxPooling1D(2),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(12, activation='softmax')  # Assuming 12 groups (change accordingly)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

"""###5.2.2 CNN and RNN with balanced data"""

selected_features = ['device_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
X2 = talking_data_df[selected_features]
y2 = talking_data_df['group_id']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, stratify=y2, random_state=42)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_balanced1, y_train_balanced1 = smote.fit_resample(X_train2, y_train2)

# Normalize the data
scaler = MinMaxScaler()
X_train_balanced1 = scaler.fit_transform(X_train_balanced1)
X_test2 = scaler.transform(X_test2)

# Reshape data to add channel dimension for CNN
X_train2 = X_train_balanced1.reshape(X_train_balanced1.shape[0], 5, 1)
X_test2 = X_test2.reshape(X_test2.shape[0], 5, 1)

# Train the model
model.fit(X_train2, y_train_balanced1, validation_data=(X_test2, y_test2), epochs=10, batch_size=32)

# Predict on test set
y_pred = model.predict(X_test2)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test2, y_pred_classes))

"""###5.2.3 The best performing model from SelectKBest & RandomForest from [3.3.2.5.1] above together with CNN &RNN to see if results will be better"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10]
    },
    cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for RandomForest: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for RandomForest:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for RandomForest:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for RandomForest:")
    print(cross_val_score(RandomForestClassifier(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model RandomForest: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

try:
    # Starting a timer
    start_time = time.time()

    # GridSearch as already provided
    grid = GridSearchCV(RandomForestClassifier(),
    {
      'n_estimators': [200],
      'max_depth': [None],
      'min_samples_split': [2]
    },
    cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for RandomForest: {grid.best_params_}")

    # Get the best RF model
    best_rf = grid.best_estimator_
    X_train_transformed = best_rf.predict_proba(X_train)
    X_test_transformed = best_rf.predict_proba(X_test)

    # Reshape for CNN
    X_train_transformed = np.expand_dims(X_train_transformed, axis=2)
    X_test_transformed = np.expand_dims(X_test_transformed, axis=2)

    # CNN & RNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train_transformed.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(y.unique()), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

    # Evaluate the deep learning model
    accuracy = model.evaluate(X_test_transformed, y_test)[1]
    print(f"Accuracy of the integrated model: {accuracy * 100:.2f}%")

    # Predict on test set
    y_pred = model.predict(X_test_transformed)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    # Print classification report
    print(classification_report(y_test, y_pred_classes))

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

# Get the best RF model
best_rf = grid.best_estimator_
X_train_transformed = best_rf.predict_proba(X_train)
X_test_transformed = best_rf.predict_proba(X_test)

# Reshape for CNN
X_train_transformed = np.expand_dims(X_train_transformed, axis=2)
X_test_transformed = np.expand_dims(X_test_transformed, axis=2)

# CNN & RNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train_transformed.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y.unique()), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

# Evaluate the deep learning model
accuracy = model.evaluate(X_test_transformed, y_test)[1]
print(f"Accuracy of the integrated model: {accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test_transformed)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

# Evaluate the deep learning model
accuracy = model.evaluate(X_test_transformed, y_test)[1]
print(f"Accuracy of the integrated model: {accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test_transformed)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

# Evaluate the deep learning model
accuracy = model.evaluate(X_test_transformed, y_test)[1]
print(f"Accuracy of the integrated model: {accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test_transformed)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

"""###5.2.4 Get the best performing SelectKBest and KNN and train with CNN and RNN"""

try:
    # Starting a timer
    start_time = time.time()
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [7],
        'weights': ['distance']
    }, cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for KNN: {grid.best_params_}")

    # Evaluation
    y_pred = grid.predict(selector.transform(X_test))
    print(f"\nClassification Report for KNN:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for KNN:\n{confusion_matrix(y_test, y_pred)}\n\n")
    print(f"Cross-validation scores for KNN:")
    print(cross_val_score(KNeighborsClassifier(), X_test, y_test, cv=3))
except Exception as e:
    print(f"An error occurred while processing model KNN: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

try:
    # Starting a timer
    start_time = time.time()

    # GridSearch as already provided
    grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [7],
        'weights': ['distance']
    }, cv=5, n_jobs=-1)

    grid.fit(X_new, y_train)
    print(f"Best parameters for KNN: {grid.best_params_}")

    # Get the best RF model
    best_rf = grid.best_estimator_
    X_train_transformed = best_rf.predict_proba(X_train)
    X_test_transformed = best_rf.predict_proba(X_test)

    # Reshape for CNN
    X_train_transformed = np.expand_dims(X_train_transformed, axis=2)
    X_test_transformed = np.expand_dims(X_test_transformed, axis=2)

    # CNN & RNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train_transformed.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(y.unique()), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

    # Evaluate the deep learning model
    accuracy = model.evaluate(X_test_transformed, y_test)[1]
    print(f"Accuracy of the integrated model KNN, CNN, & RNN: {accuracy * 100:.2f}%")

    # Predict on test set
    y_pred = model.predict(X_test_transformed)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    # Print classification report
    print(classification_report(y_test, y_pred_classes))

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Printing the total runtime
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time} seconds")

# Get the best RF model
best_rf = grid.best_estimator_
X_train_transformed = best_rf.predict_proba(X_train)
X_test_transformed = best_rf.predict_proba(X_test)

# Reshape for CNN
X_train_transformed = np.expand_dims(X_train_transformed, axis=2)
X_test_transformed = np.expand_dims(X_test_transformed, axis=2)

# CNN & RNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train_transformed.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y.unique()), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

# Evaluate the deep learning model
accuracy = model.evaluate(X_test_transformed, y_test)[1]
print(f"Accuracy of the integrated model 2nd run: {accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test_transformed)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

# Evaluate the deep learning model
accuracy = model.evaluate(X_test_transformed, y_test)[1]
print(f"Accuracy of the integrated model 3rd run: {accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test_transformed)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_transformed, y_train, epochs=10, validation_data=(X_test_transformed, y_test), batch_size=32)

# Evaluate the deep learning model
accuracy = model.evaluate(X_test_transformed, y_test)[1]
print(f"Accuracy of the integrated model 4th run: {accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test_transformed)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

from tabulate import tabulate

data = [
    ["KNN", "0.97", "0.97", "97.00%"],
    ["Integrated KNN, CNN, & RNN (1st run)", "0.97", "0.97", "96.80%"],
    ["Integrated KNN, CNN, & RNN (2nd run)", "0.97", "0.97", "96.80%"],
    ["Integrated KNN, CNN, & RNN (3rd run)", "0.97", "0.97", "96.79%"],
    ["Integrated KNN, CNN, & RNN (4th run)", "0.97", "0.97", "96.77%"],
]

headers = ["Model", "Precision (Weighted Avg)", "Recall (Weighted Avg)", "Accuracy"]

table = tabulate(data, headers, tablefmt="grid")

print(table)

# Model names
models = [
    "KNN",
    "Integrated KNN, CNN, & RNN (1st run)",
    "Integrated KNN, CNN, & RNN (2nd run)",
    "Integrated KNN, CNN, & RNN (3rd run)",
    "Integrated KNN, CNN, & RNN (4th run)",
]

# Precision values
precision = [0.97, 0.97, 0.97, 0.97, 0.97]

# Recall values
recall = [0.97, 0.97, 0.97, 0.97, 0.97]

# Accuracy values
accuracy = [97.00, 96.80, 96.80, 96.79, 96.77]

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Set bar width
bar_width = 0.2

# Set positions of bars on X-axis
x = range(len(models))

# Create bars
plt.bar(x, precision, width=bar_width, label='Precision')
plt.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall')
plt.bar([i + bar_width * 2 for i in x], accuracy, width=bar_width, label='Accuracy')

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Precision, Recall, and Accuracy Comparison')
plt.xticks([i + bar_width for i in x], models, rotation=15)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Create a DataFrame with the data
data = {
    "Model Description": [
        "SelectKBest + RandomForest",
        "Integrated RandomForest, CNN, & RNN (1st run)",
        "Integrated RandomForest, CNN, & RNN (2nd run)",
        "Integrated RandomForest, CNN, & RNN (3rd run)",
        "Integrated RandomForest, CNN, & RNN (4th run)",
        "SelectKBest + KNN (Best Model)",
        "Integrated KNN, CNN, & RNN (1st run)",
        "Integrated KNN, CNN, & RNN (2nd run)",
        "Integrated KNN, CNN, & RNN (3rd run)",
        "Integrated KNN, CNN, & RNN (4th run)",
    ],
    "Precision": [0.89, 0.85, 0.84, 0.86, 0.86, 0.97, 0.97, 0.97, 0.97, 0.97],
    "Recall": [0.94, 0.86, 0.85, 0.86, 0.86, 0.97, 0.97, 0.97, 0.97, 0.97],
    "Accuracy": ["86.00%", "80.92%", "85.48%", "85.79%", "85.64%", "97.00%", "96.80%", "96.80%", "96.79%", "96.77%"],
}

df = pd.DataFrame(data)

# Plot the DataFrame as a table
plt.figure(figsize=(10, 4))
plt.title("SelectKBest with (RandomForest and KNN) then each integrated with with CNN & RNN 4 times")
ax = plt.gca()
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*len(df.columns))
plt.show()

# Define the data
models = [
    "SelectKBest + RandomForest",
    "Integrated RandomForest, CNN, & RNN (1st run)",
    "Integrated RandomForest, CNN, & RNN (2nd run)",
    "Integrated RandomForest, CNN, & RNN (3rd run)",
    "Integrated RandomForest, CNN, & RNN (4th run)",
    "SelectKBest + KNN (Best Model)",
    "Integrated KNN, CNN, & RNN (1st run)",
    "Integrated KNN, CNN, & RNN (2nd run)",
    "Integrated KNN, CNN, & RNN (3rd run)",
    "Integrated KNN, CNN, & RNN (4th run)",
]

precision = [0.89, 0.85, 0.84, 0.86, 0.86, 0.97, 0.97, 0.97, 0.97, 0.97]
recall = [0.94, 0.86, 0.85, 0.86, 0.86, 0.97, 0.97, 0.97, 0.97, 0.97]
accuracy = [86.00, 80.92, 85.48, 85.79, 85.64, 97.00, 96.80, 96.80, 96.79, 96.77]

x = np.arange(len(models))
width = 0.3

# Create subplots for precision, recall, and accuracy
fig, ax = plt.subplots(figsize=(14, 8))

rects1 = ax.bar(x - width, precision, width, label='Precision', color='royalblue')
rects2 = ax.bar(x, recall, width, label='Recall', color='forestgreen')
rects3 = ax.bar(x + width, accuracy, width, label='Accuracy', color='tomato')

# Add labels, title, and legend
ax.set_xlabel('Models and Runs')
ax.set_ylabel('Scores')
ax.set_title('SelectKBest with RandomForest and KNN each integrated with with CNN & RNN 4 times - Precision, Recall, and Accuracy by Model and Run')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.legend()

# Display the values on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()

"""###5.4 Evaluation and Comparison of Deep Learning Results

####RandomForest with SelectKBest:

Accuracy: 86%
Precision (weighted): 0.85
Recall (weighted): 0.86
F1-Score (weighted): 0.85
####Deep Learning Model (Fourth Run):

Accuracy: 85.64%
Precision (weighted): 0.86
Recall (weighted): 0.86
F1-Score (weighted): 0.86
Here's a comparison:

#####Accuracy: The RandomForest with SelectKBest achieved a slightly higher accuracy (86%) compared to the deep learning model (85.64%), although the difference is relatively small.

#####Precision, Recall, and F1-Score: The deep learning model achieved slightly better precision, recall, and F1-score, all of which are around 0.86, while the RandomForest with SelectKBest had slightly lower values, around 0.85. Again, the differences are relatively small.

#####Advantages of RandomForest with SelectKBest:

Faster training time: The RandomForest model likely trained faster compared to the deep learning model, as indicated by the total runtime (2511 seconds for RandomForest vs. not specified for deep learning).
Advantages of Deep Learning (Fourth Run):

Flexibility: Deep learning models, once developed, can be fine-tuned for various tasks and may adapt better to more complex data patterns.
Potential for improvement: Deep learning models have the potential for further improvement with more data, more complex architectures, or longer training times.
In summary, while the RandomForest with SelectKBest approach achieved slightly higher accuracy and had a shorter training time, the deep learning model demonstrated competitive performance in terms of precision, recall, and F1-score. The choice between the two approaches depends on various factors, including computational resources, dataset size, and the potential for further optimization.

###5.5 Compare the crossvalidation scores of KNN and Random Forest
"""

import matplotlib.pyplot as plt

# Cross-validation scores
rf_scores = [0.76366821, 0.76074899, 0.75916427]
knn_scores = [0.80145127, 0.80766504, 0.80562159]

# Models' names
models = ['Random Forest', 'KNN']

# Plot the cross-validation scores
plt.figure(figsize=(10, 6))
plt.bar(models, [rf_scores[-1], knn_scores[-1]], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Cross-validation Accuracy')
plt.title('Cross-validation Accuracy for Random Forest vs. KNN')
plt.ylim(0.75, 0.85)  # Adjust the y-axis limits if needed
plt.show()

"""##6. Write up analysis and comparison
Analysis and Comparison of Models:

The project involves the evaluation of multiple machine learning models, including Random Forest, SelectKBest with RandomForest, K-Nearest Neighbors (KNN), and integrated models with Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). These models have been run in multiple iterations to assess their performance.

#####**Random Forest:**

* Precision: 0.89
* Recall: 0.94
* Accuracy: 86.00%

#####**SelectKBest with RandomForest (1st run):**

* Precision: 0.85
* Recall: 0.86
* Accuracy: 80.92%

#####**SelectKBest with RandomForest (2nd run):**

* Precision: 0.84
* Recall: 0.85
* Accuracy: 85.48%

#####**SelectKBest with RandomForest (3rd run):**

* Precision: 0.86
* Recall: 0.86
* Accuracy: 85.79%

#####**SelectKBest with RandomForest (4th run):**

* Precision: 0.86
* Recall: 0.86
* Accuracy: 85.64%

#####**K-Nearest Neighbors (KNN) - (Best Model):**

* Precision: 0.97
* Recall: 0.97
* Accuracy: 97.00%

#####**Integrated KNN, CNN, & RNN (1st run):**

* Precision: 0.97
* Recall: 0.97
* Accuracy: 96.80%

#####**Integrated KNN, CNN, & RNN (2nd run):**

* Precision: 0.97
* Recall: 0.97
* Accuracy: 96.80%

#####**Integrated KNN, CNN, & RNN (3rd run):**

* Precision: 0.97
* Recall: 0.97
* Accuracy: 96.79%

#####**Integrated KNN, CNN, & RNN (4th run):**

* Precision: 0.97
* Recall: 0.97
* Accuracy: 96.77%


##7. Recommendations and Conclusions
Both RandomForest with SelectKBest and Deep Learning models demonstrated competitive performance in predicting user demographics based on mobile device properties. The choice between these models depends on various factors:

Computational Resources: RandomForest with SelectKBest is computationally less intensive and may be preferred when resources are limited.

Complexity of the Task: If the task involves complex patterns in the data, Deep Learning models offer the potential for better performance and can adapt to such complexities.

Training Time: RandomForest with SelectKBest provides faster results, making it suitable for scenarios where quick model deployment is essential.

Accuracy: The differences in accuracy, precision, recall, and F1-score between the models are relatively small, so the choice should be based on other factors like resource availability and the specific problem requirements.

The original Deep Learning CNN and RNN model performed significantly worse than the other models in terms of accuracy, precision, recall, and F1-score. In contrast, both the RandomForest with SelectKBest and Deep Learning (Fourth Run) models demonstrated competitive performance, with only slight differences in accuracy and evaluation metrics.

Therefore, the original Deep Learning CNN and RNN model may not be considered a suitable choice for predicting user demographics based on mobile device properties. On the other hand, both RandomForest with SelectKBest and the improved Deep Learning (Fourth Run) models can provide reliable results, with the choice between them depending on factors like computational resources, training time, and the complexity of the task.

In summary, the deep learning models, particularly the improved Deep Learning (Fourth Run) model, offer promising results and demonstrate the value of deep learning in predictive tasks like this, while the original model highlights the importance of model architecture and hyperparameter tuning in deep learning projects.

Therefore, the decision to use either RandomForest with SelectKBest or Deep Learning should be based on the specific use case, computational resources, and the need for fine-tuning and adaptability. Both approaches have their strengths and can yield effective results in predicting user demographics.

###Recommendations
**Model Integration:** Combining classical machine learning models like Random Forest with deep learning techniques, specifically CNN and RNN, provided results comparable to the standalone Random Forest model. This shows that ensemble and integrated models can be a powerful strategy for complex datasets.

**Continual Learning:** The performance improvement observed across multiple runs suggests that models might benefit from continual or online learning, especially as new data becomes available.


###**Explanation of the Model Choice and Purpose:**

The choice of models for this project reflects a comprehensive approach to solving a classification problem. Here's an explanation of the models and why they were chosen:

**Random Forest:** Random Forest is a versatile ensemble learning method known for its robustness and good performance on various tasks. It was selected as a baseline model for comparison.

**SelectKBest with RandomForest:** This approach combines feature selection using SelectKBest with RandomForest. It helps evaluate the impact of feature selection on model performance.

**K-Nearest Neighbors (KNN):** KNN is a simple yet effective classification algorithm that can capture complex patterns in data. It serves as an alternative approach to tree-based models.

**Integrated KNN, CNN, & RNN:** The integrated models combine traditional machine learning (KNN) with deep learning (CNN and RNN) to leverage the strengths of both approaches.

The problem being addressed likely involves classifying data into one of several categories. The choice of models allows for comparing different techniques to find the most accurate and efficient solution.

####**Running the Model in a Production-like Environment:**

To deploy and run these models in a production-like environment, several steps need to be taken:

**Data Pipeline:** Set up a data pipeline to preprocess incoming data, including feature extraction, transformation, and scaling, to match the format used during training.

**Model Deployment:** Deploy the trained models on a production server or cloud infrastructure, allowing for real-time predictions.

**Monitoring:** Implement monitoring and logging to track model performance and potential issues in a production environment. Set up alerts for model degradation.

**Scalability:** Ensure that the deployed models can handle variable workloads and are scalable to accommodate increased usage.

**API Integration:** Expose the model through an API for easy integration into other applications and systems.

**Data Retraining:** Implement a mechanism for periodic retraining of the models with new data to keep them up to date and accurate.

**Security:** Secure the deployed models and APIs to protect against potential attacks or misuse.

###Conclusions and Marketing Implications:


####**Maintenance Going Forward:**

Maintaining machine learning models in a production environment is an ongoing process. Here are some maintenance tasks:

**Data Quality:** Continuously monitor the quality of incoming data and address issues promptly. Data drift can affect model performance.

**Model Updates:** Periodically retrain models with new data to adapt to changing patterns. Implement automated retraining pipelines.

**Performance Monitoring:** Regularly assess model performance, including accuracy, precision, recall, and F1-score. Investigate and address performance degradation.

**Security:** Stay vigilant for security vulnerabilities and update the security measures as needed.

**Scalability:** Ensure that the infrastructure supporting the models can handle increased usage as the application grows.

**Documentation:** Maintain comprehensive documentation for the models, including model versions, training data, and evaluation metrics.

In summary, the choice of models in this project reflects a well-rounded approach to classification, and deploying them in a production-like environment involves setting up data pipelines, monitoring, scalability measures, and ongoing maintenance to ensure optimal performance.
**Personalized Marketing:** With an accuracy of over 85%, marketing teams can now target ads to specific age and gender groups with high confidence, reducing wasted impressions and improving conversion rates.

**User Segmentation:** The clear segmentation allows for the development of user personas which can guide content creation, product recommendations, and overall marketing strategy.

**Future Direction:** Future work might involve integrating more data sources, considering temporal features in behavioral data, and implementing advanced neural architectures.

In conclusion, leveraging behavioral data with the right combination of data processing techniques and modeling strategies can significantly improve demographic predictions, thus facilitating more effective and personalized marketing strategies.


"""

# The most dorminant feature given the selected model is SelectKBest + KNN
import matplotlib.pyplot as plt

# Features and their SelectKBest score values
features = ['device_id', 'day', 'hour', 'phone_brand_id', 'device_model_id']
values = [79.37798090788442, 3.611097238615927, 13.117683488688748, 91.43347839050287, 33.80696925399008]

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(features, values, color='blue')
plt.xlabel('Features')
plt.ylabel('Dominance Value')
plt.title('Dominance of Features')
plt.ylim(0, 100)  # Adjust the y-axis limits if needed
plt.show()