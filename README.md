# MachineLearning

## PHW1
- Compare the performance (i.e., accuracy) of the following classification models against the same dataset.
  - Decision tree (using entropy)
  - Decision tree (using Gini index)
  - Logistic regression
  - Support vector machine
- Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ (breast-cancer-winston.data)  
- Colab link: https://colab.research.google.com/drive/1YGrzfTDws2xSI1JQwwcyiToYlszpMw-O?usp=sharing

<details>
    <summary>Detail</summary>
    
    1. Read Data and Set columns
    
    (Check Dataset: printStatistic)
    
    2. Do Preprocessing
      2-1. Drop NaN Value: Preprocess - drop
      2-2. Reset index value: Preprocess - reset
      
    (Check Dataset: printStatistic)
    
    3. Compare the performance for each classificatoin model
      3-1. Inner For Loop: Compare result with various test_ratio (0.1, 0.2, 0.3)
      3-2. Outer For Loop: Compare result with 4 classification models: Model - DecisionTreeEntropy, DecisionTreeGini, LogisticRegression, SVM
          ∙ Use various of the model parameters and hyperparameters for each model: Model - gridSearch
      3-3. Print accuracy and best hyperparameter of each case to compare: Model - printAcc
          ∙ Use various numbers k for k-fold cross validation for set best hyperparameters: Model - test
      
      Repeat above process with 2 cases (scaled/unscaled data): Model - scaling
      
      So, "3 loop (test_ratio) x 4 loop (classification model) = 12 loop" for scaled/unscaled data each

</details>

## PHW2
- Show of the following clustering algorithm and compare and experiment with different things.
  - K-means
  - EM(GMM)
  - CLARANS
  - DBSCAN
 - Dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices (housing.csv)  
 - Colab link: https://colab.research.google.com/drive/17aQGFLuZf8spSgJxFF2nn6JIyTlJyiaG?usp=sharing
 
 <details>
    <summary>Detail</summary>
    
    AUTOML:
      1. Read Data
      
      (Check Dataset: printStatistic)
      
      2. Do Preprocessing
        2-1. Fill NaN with mean value: Preprocess - fill
        2-2. Sampling data: Preprocess - sampling
        2-3. Encoding using various methods (Label, Ordinal): Preprocess - doEncoding
        2-4. Scaling using various methods (Standard, MinMax, Robust): Preprocess - doScaling
      
      3. Clustering for each algorithm (K-means, EM, CLARANS, DBSCAN): Model - KMeans, EM, CLARANS, DBSCAN
        ∙ Use for-loop for each model to experiment algorithm with k value (2, 4, 6, 8, 10, 12)
      
      4. Print plot and score, Compare result
         ∙ Plot the results of clustering to “eyeball” the results: Model - printPlot
         ∙ Use a quality measure tool, such as the Silhouette score, knee method, and purity: Model - printScore, elbow, purity_score
         ∙ Compare the clustering results with N quantiles of the medianHouseValue feature values in the original dataset: model - compare
      
      Repeat above process with 3 cases (Random combination of features)
      
      So, "3 loop (random combination of features) x 2 loop (encoder) x 3 loop (scaler) x 4 loop (cluster) = 72 loop + a (k value loop for each cluster model)"
        
      

</details>
