# Import Package
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
import sklearn.preprocessing as preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import check_call
import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = None
pd.options.display.width = None


def printStatistic(newdf):
    print(newdf.head())
    print()
    print(newdf.info())
    print()
    for col in newdf:
        print(newdf[col].unique())
    print()


# preprocessing
class Preprocess:
    def __init__(self):
        self.mydf = df.copy()

    def drop(self, col, str):
        idx = self.mydf[self.mydf[col] == str].index
        self.mydf.drop(idx, axis='index', inplace=True)

    def reset(self, col):
        self.mydf = self.mydf.astype('int')
        self.mydf.set_index(self.mydf[col], inplace=True)

    def getDf(self):
        return self.mydf


# Modeling
class Model:
    def __init__(self, ndf, sIdx, eIdx, ratio):
        self.mydf = ndf.copy()
        self.X = self.mydf.iloc[:, sIdx:eIdx]
        self.Y = self.mydf.iloc[:, eIdx]
        self.ratio = ratio
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.X, self.Y, test_size=self.ratio,
                                                                            shuffle=True, random_state=35)
        self.m = None
        self.model = None
        self.params = None
        self.grid = None
        self.em = None
        self.yTrainPred = None
        self.yTestPred = None

    def setModel(self, m):
        self.m = m
        if self.m == 'DecisionTreeEntropy':
            self.DecisionTreeEntropy()
        elif self.m == 'DecisionTreeGini':
            self.DecisionTreeGini()
        elif self.m == 'LogisticRegression':
            self.LogisticRegression()
        elif self.m == 'SVM':
            self.SVM()
        else:
            self.model = None
            self.params = None

    # Decision tree - using entropy
    def DecisionTreeEntropy(self):
        self.model = DecisionTreeClassifier()
        self.params = {
            'criterion': ['entropy'],
            'max_depth': [2, 3],
            'min_samples_split': [2, 3]
        }

    # Decision tree - using Gini index
    def DecisionTreeGini(self):
        self.model = DecisionTreeClassifier()
        self.params = {
            'criterion': ['gini'],
            'max_depth': [2, 3],
            'min_samples_split': [2, 3]
        }

    # Logistic regression
    def LogisticRegression(self):
        self.model = LogisticRegression()
        self.params = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.01, 0.1, 1, 5, 10],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 1000]
        }

    # Support vector machine
    def SVM(self):
        self.model = SVC()
        self.params = {
            'kernel': ['rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        }

    # standard scaling
    def scaling(self):
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.xTrain)
        self.xTrain = scaler.transform(self.xTrain)
        self.xTest = scaler.transform(self.xTest)

    # hyperparameter tunig - gridSearch : We find the best combination of some of the potential parameters we've designated
    def gridSearch(self):
        self.grid = GridSearchCV(self.model, param_grid=self.params, cv=3, refit=True)
        self.grid.fit(self.xTrain, self.yTrain)

        self.em = self.grid.best_estimator_
        self.yTrainPred = self.em.predict(self.xTrain)  # Predict using xTrain
        self.yTestPred = self.em.predict(self.xTest)  # Predict using xTest

    def printAcc(self):
        print("<< Modeling Result >>")

        scores = pd.DataFrame(self.grid.cv_results_)
        scores = scores.sort_values(by=["rank_test_score"])
        scores = scores.set_index(
            scores["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
        ).rename_axis("hyperparameter")
        print(scores[["params", "rank_test_score", "mean_test_score"]])
        print()

        scores = scores.iloc[:3]
        model_scores = scores.filter(regex=r"split\d*_test_score")
        fig, ax = plt.subplots()
        sns.lineplot(
            data=model_scores.transpose().iloc[:3],
            dashes=False,
            palette="Set1",
            marker="o",
            alpha=0.5,
            ax=ax,
        )
        plt.title("For the top 3 mean_test_scores")
        ax.set_xlabel("CV test fold", size=12, labelpad=10)
        ax.set_ylabel("Model AUC", size=12)
        ax.tick_params(bottom=True, labelbottom=False)
        plt.show()

        print('> best parameters:', self.grid.best_params_)
        print('> best score:', self.grid.best_score_)
        print('> best estimator:', self.em)
        print("> Train Accuracy: {}".format(accuracy_score(self.yTrain, self.yTrainPred)))
        print("> Test Accuracy: {}".format(accuracy_score(self.yTest, self.yTestPred)))
        print()

        if self.m == 'DecisionTreeEntropy' or self.m == 'DecisionTreeGini':
            # Make the tree visualization and store
            export_graphviz(self.em, out_file="{} tree(ratio={}).dot".format(self.m, self.ratio),
                            feature_names=self.X.columns, class_names=self.Y.name,
                            filled=True, impurity=False)
            check_call(['dot', '-Tpng', "{} tree(ratio={}).dot".format(self.m, self.ratio), '-o',
                        "{} tree(ratio={}).png".format(self.m, self.ratio)])

    def test(self):
        print("<< Testing Result with best parameter >>")
        k = [3, 5]
        for i in range(len(k)):
            kfold = KFold(n_splits=k[i], shuffle=True, random_state=32)
            score = cross_val_score(self.em, self.X, self.Y, scoring='accuracy', cv=kfold)
            print("> k={} each accuracy : {}".format(k[i], score))
            print("> k={} average accuracy accuracy: {}".format(k[i], np.mean(score)))


# read data file
df = pd.read_csv('../data/breast-cancer-wisconsin.data', header=None)
df.columns = ['id', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'class']
printStatistic(df)

df1 = Preprocess()
df1.drop('7', '?')
df1.reset('id')
df1 = df1.getDf()
printStatistic(df1)

test_ratio = [0.1, 0.2, 0.3]
lists = ['DecisionTreeEntropy', 'DecisionTreeGini', 'LogisticRegression', 'SVM']

for j in range(len(lists)):
    # Training and Testing using different test ratio of dataset
    for i in range(len(test_ratio)):
        print("===== model: {}, test_ratio = {} ==================".format(lists[j], test_ratio[i]))
        print("----- Result for unscaled X -------------")
        ndf = Model(df1, 0, -1, test_ratio[i])
        ndf.setModel(lists[j])
        ndf.gridSearch()
        ndf.printAcc()
        ndf.test()
        print("-----------------------------------------")

        print("----- Result for scaled X ---------------")
        ndf_scaled = Model(df1, 0, -1, test_ratio[i])
        ndf_scaled.setModel(lists[j])
        ndf_scaled.scaling()
        ndf_scaled.gridSearch()
        ndf_scaled.printAcc()
        ndf_scaled.test()
        print("-----------------------------------------")
        print()
