# Import required libraries
import pandas as pd
from sklearn.neural_network import MLPClassifier

# Import necessary modules
from sklearn.model_selection import train_test_split

# import the dataset and look at the collection
df = pd.read_csv('diabetes.csv')
print(df.shape)
df.describe().transpose()

# declare the desired y-hat value column
target_column = ['Outcome']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()

# split the features and labeled out columns
X = df[predictors].values
y = df[target_column].values

# split train test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

# build a three layer neural network, each with 8 nodes in the layer
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
# fit model to our training set
mlp.fit(X_train,y_train)

# get predictions for the training set and the test set
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))