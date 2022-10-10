import numpy as np
import pandas as pd
from sklearn import preprocessing
from pyclustering.cluster.clarans import clarans
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = None
pd.options.display.width = None


def printStatistic(newdf):
    print(newdf.head())
    print()
    print(newdf.info()) # Checking the information in a data frame
    print()
    print(newdf.isna().sum()) # Check the number of NA's in the data frame
    print()
    for col in newdf:
        print(newdf[col].unique())
    print()


class Preprocess:
    def __init__(self, ndf):
        self.mydf = ndf.copy()
        self.X = None
        self.Y = None
        self.newX = None
        self.label = None
        self.scaler = None
        self.encoder = None
        self.cols = None
        self.newCols = None

    def fill(self):
        self.mydf.fillna(self.mydf.mean(), inplace=True)  # Fill NaN with mean value

    def sampling(self):
        self.mydf = self.mydf.sample(frac=0.02, random_state=1004)  # Only 2% is randomly extracted and used
        n = np.random.randint(3, len(self.mydf.columns)-1)  # Extract as many columns as randomInt between 3 and 9
        self.newCols = list(self.mydf.sample(n=n, axis=1).columns)


    def doEncoding(self, encoder, labelIdx): # Encoding
        if encoder == 'Label': # It simply converts the string value into a numeric category value
            self.encoder = preprocessing.LabelEncoder()
        elif encoder == 'Ordinal': # Unlike the label encoder, Returns each element inside an array
            self.encoder = preprocessing.OrdinalEncoder()
        else:
            print("== Invalid Encoder ===") # Incorrect Encoding
            exit(0)
        self.label = pd.DataFrame(self.mydf.iloc[:, labelIdx])
        self.label = self.encoder.fit_transform(self.label)
        self.mydf['ocean_proximity'] = self.label

    def doScaling(self, scaler): # Scaling
        if scaler == 'Standard': # Make all features have a normal distribution with a mean of 0 and a variance of 1
            self.scaler = preprocessing.StandardScaler()
        elif scaler == 'MinMax': # Makes all features have data values between 0 and 1
            self.scaler = preprocessing.MinMaxScaler()
        elif scaler == 'Robust': # Makes all the features have a normal distribution with a median of 0 and a quartile of 1
            self.scaler = preprocessing.RobustScaler()
        else:
            print("== Invalid Scaler ===") # Incorrect Scaling
            exit(0)
        self.cols = list(self.mydf.columns)
        self.mydf[self.cols] = self.scaler.fit_transform(self.mydf[self.cols])
        self.mydf = pd.DataFrame(self.mydf)

    def XY(self, targetIdx):
        self.Y = self.mydf.iloc[:, targetIdx]
        self.X = self.mydf.drop(self.mydf.columns[targetIdx], axis=1)
        while self.Y.name in self.newCols:
            self.newCols.remove(self.Y.name)
        self.newX = self.X.loc[:, self.newCols]
        return self.X, self.Y, self.newX


class Model:
    def __init__(self, x, y, newx):
        self.X = x.copy()
        self.Y = y.copy()
        self.newX = newx.copy()
        self.model = None
        self.ncluster = [2, 4, 6, 8, 10, 12] # Number of Clusters
        self.eps = [0.05, 0.1, 0.5, 1, 3] # Epsilon
        self.grid = None
        self.em = None
        self.data = None

    def clustering(self, cluster):
        if cluster == 'KMeans':
            self.KMeans()
        elif cluster == 'EM':
            self.EM()
        elif cluster == 'CLARANS':
            self.CLARANS()
        elif cluster == 'DBSCAN':
            self.DBSCAN()
        else:
            print("== Invalid Cluster ===")
            exit(0)

    def KMeans(self): # Clustering algorithm that binds data into K clusters
        for n in self.ncluster:
            print("> for n = {}".format(n))
            # Copy to maintain original data.
            self.data = self.newX.copy()

            # Run KMeans Algorithms
            model = KMeans(n_clusters=n, init="k-means++", random_state=42).fit(self.data)
            # Creating a Cluster Label
            self.data["cluster"] = model.labels_

            self.printPlot("[K-means] number of clusters : " + str(n))

            self.compare(n)

            self.printScore()

    def EM(self): # One of several models that applied EM algorithm with Gaussian Mixer Model
        for n in self.ncluster:
            print("> for n = {}".format(n))
            # Copy to maintain original data.
            self.data = self.newX.copy()

            # Run GaussianMixture Algorithms
            model = GaussianMixture(n_components=n).fit(self.data)

            # Creating a Cluster Label
            self.data["cluster"] = model.predict(self.data)

            self.printPlot("[EM(GMM)] number of clusters : " + str(n))

            self.compare(n)

            self.printScore()

    def CLARANS(self):  # Algorithm to select an initial set of representative objects
                        # and explore a new representative object among the neighboring sets
        for n in self.ncluster:
            print("> for n = {}".format(n))
            # Copy to maintain original data.
            self.data = self.newX.copy()

            # Run Clarans Algorithms
            model = clarans(self.data.values.tolist(), n, 3, 5).process()

            # Creating a Cluster Label
            idx_list = [-1 for i in range(0, len(self.data))]
            idx = 0
            for k in model.get_clusters():
                for i in k:
                    idx_list[i] = idx
                idx = idx + 1
            self.data["cluster"] = idx_list

            self.printPlot("[CLARANS] number of clusters : " + str(n))

            self.compare(n)

            self.printScore()

    def DBSCAN(self): # Perform clustering by receiving two parameters: minimum radius and minimum number of points
        self.elbow()  # Identify the optimal Epsilon value
        for eps in self.eps:
            print("> for eps = {}".format(eps))
            # Copy to maintain original data.
            self.data = self.newX.copy()

            # Run DBScan Algorithms
            model = DBSCAN(eps=eps).fit(self.data)

            # Creating a Cluster Label
            self.data["cluster"] = model.labels_

            self.printPlot("[DBSCAN] eps of clusters  : " + str(eps))

            self.compare(self.data["cluster"].max() + 2)

            self.printScore()

    def elbow(self):
        neigh = NearestNeighbors(n_neighbors=5)
        nbrs = neigh.fit(self.newX)
        dist, dices = nbrs.kneighbors(self.newX)
        dist = np.sort(dist, axis=0)
        dist = dist[:, 1]
        plt.title("Find optimal value for epsilon")
        plt.xlabel("5-NN distance")
        plt.ylabel("Points sorted by distance")
        plt.plot(dist)
        plt.show()

    def printPlot(self, title):
        plot = sns.pairplot(self.data, hue='cluster', palette="bright", corner=True)
        plot.fig.suptitle(title, y=1.05)
        plot.fig.set_size_inches(8, 8)
        plt.show()

    def compare(self, n):
        # Reattaching median_house_value data
        self.data['median_house_value'] = self.Y.values

        # Sort by median_house_value data with the original data
        self.data = self.data.sort_values(ascending=False, by='median_house_value')

        # Divide the source data value equally (number of Equivalent = number of Clusters)
        self.data['target'] = pd.cut(self.data['median_house_value'], n, labels=range(0, n))

    def printScore(self):
        compare_score = len(self.data.loc[self.data['target'] == self.data['cluster']]) / len(self.data) * 100

        print("Quantiles clustering result : ", compare_score)

        print("Clustering with purity_score : ", self.purity_score(self.data['target'], self.data['cluster']))

        try:
            print("Clustering with silhouette_score (euclidean) : ", silhouette_score(self.data, self.data["cluster"], metric='euclidean'))

        except:
            print("Clustering with silhouette_score : 0")

    def purity_score(self, yTrue, yPred):
        contingency_matrix = metrics.cluster.contingency_matrix(yTrue, yPred)

        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# A program that can run multiple combinations automatically
def AUTOML(encoder, scaler, cluster):
    df = pd.read_csv('../data/housing.csv')
    printStatistic(df)

    for n in range(3):  # Feature Run for 3 combinations
        df1 = Preprocess(df)
        df1.fill()
        df1.sampling()

        for i in range(len(encoder)):
            for j in range(len(scaler)):
                for m in range(len(cluster)):
                    print("== Cluster: {}, Scaler: {}, Encoder: {} ===========".format(cluster[m], scaler[j], encoder[i]))
                    df1.doEncoding(encoder[i], 9)
                    df1.doScaling(scaler[j])
                    X, Y, newX = df1.XY(8)  # Y = medianHouseValue, X = The rest (It remains scaled and encoded)
                    df2 = Model(X, Y, newX)
                    df2.clustering(cluster[m])
                    print()

# Various combinations
encoder = ['Label', 'Ordinal']
scaler = ['Standard', 'MinMax', 'Robust']
cluster = ['KMeans', 'EM', 'CLARANS', 'DBSCAN']

AUTOML(encoder, scaler, cluster)
