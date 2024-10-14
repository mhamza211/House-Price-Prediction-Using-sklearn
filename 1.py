# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
train_data = pd.read_csv("E:/Internship/train.csv", index_col="Id")  # Training data
test_data = pd.read_csv("E:/Internship/test.csv", index_col="Id")    # Testing data
submission_template = pd.read_csv("E:/Internship/submission.csv", index_col="Id")  # Submission template

# Merge test dataset with submission dataset
merged_data = test_data.merge(submission_template, right_index=True, left_index=True)

# Combine training and merged testing datasets for comprehensive analysis
combined_data = pd.concat([train_data, merged_data], axis=1)

# Select numeric features from the combined dataset
numeric_data = combined_data.select_dtypes(include=[np.number])

# Fill missing values with the mean of each numeric column
cleaned_data = numeric_data.fillna(numeric_data.mean())

# Filter relevant features for model training
features_data = train_data[['GrLivArea', 'BedroomAbvGr', 'BsmtFullBath', 'FullBath', 'SalePrice']]

# Create a new feature 'TotalBaths' that sums full baths and basement baths
features_data['TotalBaths'] = features_data['BsmtFullBath'] + features_data['FullBath'].copy()

# Define features (X) and target variable (y)
x = features_data[['GrLivArea', 'BedroomAbvGr', 'TotalBaths']]  
y = features_data['SalePrice']

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
linear_regressor = LinearRegression()

# Train the model on the training data
linear_regressor.fit(x_train, y_train)

# Make predictions on the testing data
y_predictions = linear_regressor.predict(x_test)

# Calculate and print performance metrics
mse_value = mean_squared_error(y_test, y_predictions)  # Mean Squared Error
print("The r^2 score is :", r2_score(y_test, y_predictions))  # R² score
print("The mean squared error is :", mse_value)  # MSE

# Prepare new data for prediction
new_house_data = pd.DataFrame({
    'GrLivArea': [8000],    # Example living area
    'BedroomAbvGr': [4],    # Example number of bedrooms
    'TotalBaths': [2]       # Example total bathrooms
})

# Predict the price of the new house
predicted_house_price = linear_regressor.predict(new_house_data)
print("The predicted price of the house is ", predicted_house_price[0])  # Display the predicted price
