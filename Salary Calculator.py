import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial.csv", sep = ";")

#creating polynomial regression object and assigne degree to 4
polynomial_regression = PolynomialFeatures(degree = 4) 

plt.scatter(df['experience'], df['salary'])
plt.xlabel('Experience (in year)')
plt.ylabel('Salary')
plt.show()

x_polynomial = polynomial_regression.fit_transform(df[['experience']])

#We create our regression model object, our reg object, and call its fit method to fit x_polynomial and y_polynomial
reg = LinearRegression()
reg.fit(x_polynomial, df['salary'])

y_head = reg.predict(x_polynomial)
plt.plot(df['experience'], y_head, color="red", label="polynomial regression")
plt.legend()

plt.scatter(df['experience'], df['salary'])
plt.show()

experience = float(input("Eneter experience year: "))
x_polynomial1 = polynomial_regression.fit_transform([[experience]])
reg.predict(x_polynomial1)



