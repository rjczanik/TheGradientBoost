## X = Head size in cm3, Y = Brain weight in grams


# Earlier we discussed that we will approximate the relationship between X and Y to a line. Let’s say we have a few
# inputs and outputs. And we plot these scatter points in 2D space, we will get an image similar to this. In most cases
# it would be a ordinary least squared method, Today we are going to show an example of an actual case study.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel('medical_data.xls')
print(data.shape)
data.head()

# Collecting X and Y

X = data.iloc[:,0].values
Y = data.iloc[:,1].values

# Mean X and Y
# Using the formula to calculate b1, and b0 ($b1$, $b0$)
mean_x = np.mean(X)
mean_y = np.mean(Y)
# Total number of values
m = len(X)
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)
# Print coefficients
print(b1, b0)

### How do we interpret the regression coefficients for linear relationships?
# Regression coefficients represent the mean change in the response variable for one unit of change in the predictor
# variable while holding other predictors in the model constant. This statistical control, that regression provides,
# is important because it isolates the role of one variable from all of the others in the model.
# Here, we have our coefficients.

##          Total number of claims = 3.41384 + 19.9944 * number of claims



## Visualization:
# Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x
# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

# This model is not bad. But we need to find how good the model is. There are many methods to evaluate models. We will
# use the Root Mean Squared Error and Coefficient of Determination ( R2 Score). Root Mean Squared Error (RMSE) RMSE is
# the square root of sum of all errors divided by number of values, or mathematically,
# Calculating Root Mean Squares Error
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print(rmse)


### Coefficient of Determination ( Score)

# R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the
# coefficient of determination, or the coefficient of multiple determinations for multiple regressions. The definition
# of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by
# a linear model. Or:
#
# R-squared = Explained variation / Total variation
#
# R-squared is always between 0 and 100%:
# 0% indicates that the model explains none of the variability of the response data around its mean. 100% indicates that
# the model explains all the variability of the response data around its mean. In general, the higher the R-squared,
# the better the model fits your data.

ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)



### The easy way: Scikit-learn
# Import libraries and tools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((m, 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)
# Calculating RMSE and  Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)
print(np.sqrt(mse))
print(r2_score)