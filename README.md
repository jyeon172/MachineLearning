# MachineLearning

## PHW1
- Compare the performance (i.e., accuracy) of the following classification models against the same dataset.
  - Decision tree (using entropy)
  - Decision tree (using Gini index)
  - Logistic regression
  - Support vector machine
- Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ (breast-cancer-winston.data)

<details>
    <summary>Detail</summary>
    
    1. Read Data and Set columns
    
    (Check Dataset using "printStatistic")
    
    2. Do Preprocessing
      2-1. Drop NaN Value _(Preprocess)drop_
      2-2. Reset index value ("Preprocess" - "reset")
      
    (Check Dataset using "printStatistic")
    
    3. Double For Loop for test below using scaled data and unscaled data ("Model" - "scaling")
      3-1. Inner For Loop: Compare result with various test_ratio (0.1, 0.2, 0.3)
      3-2. Outer For Loop: Compare result with 4 classification models ("Model" - "DecisionTreeEntropy", "DecisionTreeGini", "LogisticRegression", "SVM")
        -> Use various of the model parameters and hyperparameters for each model
        -> Use various numbers k for k-fold cross validation for set best hyperparameters ("Model" - "gridSearch")
      3-3. Print accuracy and best hyperparameter of each case to compare ("Model" - "printAcc")
       -> Use various numbers k for k-fold cross validation for set best hyperparameters ("Model" - "test")
      So, 3 loop (test_ratio) x 4 loop (classification model) = 12 loop for scaled, unscaled data each

</details>

## PHW2
- Show of the following clustering algorithm and compare and experiment with different things.
  - K-means
  - EM(GMM)
  - CLARANS
  - DBSCAN
 - Dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices (housing.csv)
 
 <details>
    <summary>Detail</summary>
    
    ㅇㅇㅇㅇㅇㅇㅏ아아아아ㅏㅏ

</details>
