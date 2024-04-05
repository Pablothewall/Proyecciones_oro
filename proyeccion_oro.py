import datetime
import pandas_datareader.data as web
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
tickers="REAINTRATREARAT10Y"
start=datetime.datetime(1930,1,1)
real_interest=web.DataReader(tickers,'fred', start) 
cpi=web.DataReader("CPIAUCSL",'fred', start)

control=pd.merge(real_interest, cpi, left_index=True, right_index=True).dropna()

file_path = 'Prices (1).xlsx'

# Name of the sheet to read
sheet_name = 'Daily'

# Read the Excel file for the specific sheet
price = pd.read_excel(file_path, sheet_name=sheet_name)
price.set_index("Date", inplace=True)
modelo=pd.merge(control, price, left_index=True, right_index=True).dropna()

print(modelo)

modelo["log CPIAUCSL"]=modelo['CPIAUCSL'].apply(lambda x:math.log(x,  2.71828))
modelo["log Price"]=modelo['Price'].apply(lambda x:math.log(x,  2.71828))
print(modelo)
# Define the input features (X) and the target variable (y)
X = modelo[['REAINTRATREARAT10Y', "log CPIAUCSL"]]
y = modelo["log Price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=False
                                                    # random_state=42
                                                    )

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)


# Make predictions on the testing data
y_pred_train = model.predict(X_test)
graph=pd.DataFrame(
    {"Real":y_test,
     "Pred":y_pred_train

    }
)
graph.plot()
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)