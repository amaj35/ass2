# pandas is necessary for reading csv files.
import pandas as pd
# numpy is used for multidimensional arrays
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# user to compare accuracy between test data and training data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.stats import mode

# read in training data
train = pd.read_csv('MNIST_training.csv', header=None)

#train.shape #(949,785)
#train.info()#<class 'pandas.core.frame.DataFrame'>
            #RangeIndex: 950 entries, 0 to 949
            #Columns: 785 entries, 0 to 784
            #dtypes: object(785)
            #memory usage: 5.7+ MB
#print(train.head(None))

x = np.array(train.iloc[0:,])
#print(train[0])
y = np.array(train[0]).reshape(95, 10)
#print(x)
#print(y)
newy = np.delete(y, 0)
print(newy)

sns.displot(newy)

## d(p,q) = sqrt|(q-p)^2
def euclidean_distance(pt1, pt2):
    distance = np.sqrt(np.sum(pt1-pt2)**2)
    return distance

a = np.array[3, 4]
b = np.array[5, 9]

print(euclidian_distance(a,b))