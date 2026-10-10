import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

#loding the data
data = pd.read_csv("Housing.csv")
print(data.head())
print("Dataset shape:", data.shape)
print(data.isnull().sum())

#simple Linear Regression
x = data[["area"]]
y = data["price"]

#splitting data into training and testing 
X_train, X_test, y_train, y_test= train_test_split(
    x,y,test_size=0.2, random_state=42
) 

#creationg model
simple_model = LinearRegression()

#training the model
simple_model.fit(X_train, y_train)

#pridicting House Price
y_pred = simple_model.predict(X_test)

print("Actual Prices:", y_test.values[:5])
print("pridicted price:", y_pred[:5])

print("Slope:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_test)

print("MAE:", mae)
print("MSE:", mse)
print("R2 score:", r2)

#Ploting the regression line
plot_data = X_test.copy()
plot_data["actual_price"] = y_test
plot_data = plot_data.sort_values("area")

line_pridiction = simple_model.predict(plot_data[["area"]])

#draw actual price
plt.scatter(
    plot_data["area"],
    plot_data["actual_price"],
    label = "Actual Prices"
)

#draw the regression line
plt.plot(
    plot_data["area"],
    line_pridiction,
    color="red",
    label = "Regression Line"
)

plt.xlabel("House Area")
plt.ylabel("House Price")
plt.title("Simple linear regression: House Price Pridiction")
plt.legend()
plt.show()

#Multiple Linear Regression
# Select all features except the target
X = data.drop("price", axis=1)
y = data["price"]

# Convert text categories into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)

# Predict prices
y_pred_multi = multiple_model.predict(X_test)

# Evaluate the model
print("Multiple Regression Results")
print("MAE:", mean_absolute_error(y_test, y_pred_multi))
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("R2 Score:", r2_score(y_test, y_pred_multi))

# Display coefficients
print("\nModel Coefficients:")
for feature, coefficient in zip(X.columns, multiple_model.coef_):
    print(feature, ":", coefficient)

print("Intercept:", multiple_model.intercept_)