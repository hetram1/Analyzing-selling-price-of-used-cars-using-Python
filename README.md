# Analyzing Selling Price of Used Cars Using Python

## Overview
In this project, we analyze the selling price of used cars using Python. The goal is to explore various factors affecting car prices and apply data analysis techniques to extract insights. This project is structured to clean and preprocess data, perform exploratory data analysis (EDA), and visualize the relationships between different features and price.

## Dataset
The dataset used in this analysis contains information about used cars, including attributes like brand, fuel type, engine size, horsepower, and more. The data is available in `.csv` format and can be downloaded from [here](https://media.geeksforgeeks.org/wp-content/uploads/20240909124157/archive.zip). The dataset includes the following features:

- **symboling**: Risk factor of a vehicle (assigned by insurance companies)
- **normalized-losses**: Normalized loss values
- **make**: Car brand
- **fuel-type**: Type of fuel (gas, diesel)
- **aspiration**: Engine aspiration type
- **num-of-doors**: Number of doors
- **body-style**: Type of car body (sedan, hatchback, etc.)
- **drive-wheels**: Type of drivetrain
- **engine-location**: Location of the engine (front, rear)
- **wheel-base**: Distance between front and rear wheels
- **dimensions (length, width, height)**: Car dimensions
- **curb-weight**: Weight of the car
- **engine-type**: Type of engine
- **num-of-cylinders**: Number of cylinders
- **engine-size**: Size of the engine
- **fuel-system**: Fuel system type
- **horsepower**: Engine power
- **peak-rpm**: Maximum RPM
- **city-mpg & highway-mpg**: Fuel efficiency
- **price**: Selling price of the car

## Installation
To run this project, install the required dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Data Preprocessing & Analysis
### Step 1: Importing Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
```

### Step 2: Loading the Dataset
```python
df = pd.read_csv('used_cars.csv')
df.head()
```

### Step 3: Handling Missing Values
```python
# Check for missing values
df.isnull().sum()

# Drop or fill missing values
df = df.dropna()  # Alternatively, use df.fillna(value)
```

### Step 4: Data Cleaning & Transformation
```python
# Convert price to numeric
df = df[df['price'] != '?']
df['price'] = df['price'].astype(float)
```

### Step 5: Feature Scaling
```python
df['city-mpg'] = 235 / df['city-mpg']
df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)
```

### Step 6: Binning the Price Column
```python
bins = np.linspace(df['price'].min(), df['price'].max(), 4)
labels = ['Low', 'Medium', 'High']
df['price-binned'] = pd.cut(df['price'], bins, labels=labels, include_lowest=True)
```

### Step 7: Exploratory Data Analysis (EDA)
#### Correlation Heatmap
```python
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```
#### Box Plot for Price Distribution
```python
sns.boxplot(x='drive-wheels', y='price', data=df)
plt.show()
```
#### Scatter Plot of Engine Size vs. Price
```python
plt.scatter(df['engine-size'], df['price'])
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.title('Engine Size vs Price')
plt.show()
```

### Step 8: ANOVA Test for Feature Importance
```python
grouped_data = df[['make', 'price']].groupby(['make'])
f_val, p_val = stats.f_oneway(grouped_data.get_group('honda')['price'],
                              grouped_data.get_group('toyota')['price'])
print(f"ANOVA Results: F={f_val}, P={p_val}")
```

## Conclusion
- The analysis helps in understanding which car features influence the selling price the most.
- Engine size, fuel type, and brand have a significant impact on price.
- The heatmap and scatter plots provide strong visual insights.

## Future Enhancements
- Build a Machine Learning model to predict car prices.
- Implement more advanced feature engineering.
- Extend the dataset with additional real-world features.

## Author
This project is developed and maintained by **Hetram**, an AI/ML engineer and data science enthusiast.

---
*Feel free to contribute and improve this project!* 🚀

