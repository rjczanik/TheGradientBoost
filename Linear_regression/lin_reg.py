import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
### We will start with the most familiar linear regression, a straight-line fit to data. A straight-line fit is a model of the form
###       y=ax+b
### where a is commonly known as the slope, and b is commonly known as the intercept.
### Consider the following data, which is scattered about a line with a slope of 2 and an intercept of -5

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)

plt.figure()
plt.scatter(x, y)

### We can use Scikit-Learn's LinearRegression estimator to fit this data and construct the best-fit line

model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, yfit);

### The slope and intercept of the data are contained in the model's fit parameters, which in Scikit-Learn are always
### marked by a trailing underscore. Here the relevant parameters are coef_ and intercept_

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)


### The LinearRegression estimator is much more capable than this, however—in addition to simple straight-line fits, it
### can also handle multidimensional linear models of the form
###        y=a0+a1x1+a2x2+...

### where there are multiple x values. Geometrically, this is akin to fitting a plane to points in three dimensions, or
### fitting a hyper-plane to points in higher dimensions.
### The multidimensional nature of such regressions makes them more difficult to visualize, but we can see one of these
### fits in action by building some example data, using NumPy's matrix multiplication operator

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

### Measuring the correlation factor between the dependent and independent variables can be done with numpy:
y = 2 * x - 5 + rng.randn(50)

correlation = np.corrcoef(x, y)
determination = correlation**2

print('Correlation: ' + str(correlation))
print('Determination: ' + str(determination))

x1 = 5 * rng.rand(50)

plt.figure()
plt.scatter(x, x1)

correlation1 = np.corrcoef(x, x1)
determination1 = correlation1**2

print('Correlation1: ' + str(correlation1))
print('Determination1: ' + str(determination1))

plt.show()
