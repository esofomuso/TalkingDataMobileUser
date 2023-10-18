# TalkingDataMobileUser
The overarching goal is to help clients interact more effectively with their target audiences through data-driven marketing strategies.

## 1. Introduction:
The rise of mobile platforms has brought an immense opportunity to understand user behavior and preferences in an unprecedented way. The TalkingData platform seeks to leverage this opportunity by utilizing a rich dataset derived from 500 million daily active mobile devices in China. The overarching goal is to help clients interact more effectively with their target audiences through data-driven marketing strategies.
In today's digital age, understanding user behavior and preferences is paramount for effective marketing. The study proposes to utilize TalkingData, China’s largest third-party mobile data platform, to predict users' demographic characteristics based on their app usage, geolocation, and mobile device properties. This approach would facilitate more focused and data-driven marketing strategies for developers and brand advertisers worldwide.
TalkingData, China’s largest third-party mobile data platform, understands that everyday choices and behaviors paint a picture of who we are and what we value. Currently, TalkingData is seeking to leverage behavioral data from more than 70% of the 500 million mobile devices active daily in China to help its clients better understand and interact with their audiences.

### Problem Statement:
The project aims to craft predictive models that can accurately infer users' demographic details using the extensive datasets provided by TalkingData. The ultimate goal is to enhance the efficiency and relevance of marketing efforts by offering insights into the preferences and habits of various user demographics.

### Specialization Topics:
The project encapsulates specialization topics such as data analytics, machine learning, and predictive modeling, aligning with the curriculum's focus areas. Advanced data processing and analytics techniques will be employed to derive meaningful insights from the vast datasets.
### Challenges:
The biggest anticipated challenges would be handling the large volume of data efficiently and dealing with potential data inconsistencies and missing values. Additionally, crafting a model that accurately predicts user demographics with a high degree of reliability might be challenging.
### Value Proposition:
The solution holds immense value as it enables advertisers and developers to tailor their marketing strategies based on the predictive analysis of user demographics. It assists in understanding user preferences and behaviors, fostering more targeted and effective marketing campaigns.
### Project Goals:
The primary goal is to develop a robust predictive model to predict users' demographic characteristics accurately. The project seeks to achieve high accuracy and reliability in predictions, helping clients to fine-tune their marketing strategies.

## 2. Data Access:

To ensure a robust analysis, we are leveraging the dataset from '''[Kaggle](https://www.kaggle.com/competitions/talkingdata-mobile-user-demographics/code)''' in 6 CSV files.
###Dataset Highlights:
Total number of CSV files containing 'Talkingdata Mobile User Demographics' data: 6 (Rows/Records, Variables/Columns)
* events.csv the following rows and columns: (3252950, 5)
* app_labels.csv the following rows and columns: (459943, 2)
* label_categories.csv the following rows and columns: (930, 2)
* phone_brand_device_model.csv the following rows and columns: (187245, 3)
* sample_submission.csv the following rows and columns: (112071, 13)

### Key Attributes:
Gender-Age-Group (group_id), Device_ID, Day, Hour, Device_model_id, phone_brand_id

This data, representing a broad spectrum of TalkingData's customer base, serves as the bedrock of our research, enabling us to draw meaningful correlations between various user attributes and their Mobile Usage.

## 3. Research Question:
How can the behavioral data from TalkingData be leveraged to predict users' demographic characteristics more accurately and facilitate effective, personalized marketing strategies?

### Sub-questions:
  1. Which cluster contains the most usage by females between 27 and 32?
  2. Which cluster contains the most device variety by males between 29 and 38?


## 4. Methodology
  1. Data collection
  2. Exploratory Data Analysis (EDA) and Class balancing
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
