# Youtube Trending Videos Analysis with Python
# YouTube Trending Videos Dataset Analysis 📊

## Description 🎥
This project focuses on analyzing a dataset containing data for trending YouTube videos in Mediterranean countries. The goal is to explore various factors influencing video popularity such as category, comments, descriptions, tags, and more, while also applying machine learning techniques to predict video success.

## Contents 📂
- **Data Exploration**: Understanding the data and performing initial analysis.
- **Data Preprocessing**: Preparing the data for model training.
- **Machine Learning Models**:
  - **Regression**: Predicting the number of views.
  - **Classification**: Categorizing videos into "low", "medium", "high" popularity.
  - **Clustering**: Grouping similar videos based on attributes.
- **Statistical Analysis**: Identifying correlations and providing insights.
- **Data Visualization**: Visualizing data to understand trends.

Libraries Used 📚
pandas: Data manipulation and analysis

matplotlib and seaborn: Data visualization

sklearn: Machine learning models

wordcloud: Generate word clouds

tensorflow/keras (optional): Deep learning models for advanced techniques

Requirements 💻
To run this project locally, you will need to install the following libraries:
pip install pandas matplotlib seaborn scikit-learn wordcloud tensorflow

- Conclusions 🧠
This project provides valuable insights into trending YouTube videos in Mediterranean countries. Through data analysis and machine learning, we can better understand the factors contributing to video popularity and predict their success in real-time!

## How to Run 🚀

Follow these steps to execute the project:

### 1. Load Data

```python
1. Load Data

```python
import pandas as pd

# Load the dataset
ds = "path_to_dataset/youtube_trending_mediterranean.csv"
df = pd.read_csv(ds)

# Display the first 5 rows
df.head()

2. Data Exploration
# Dataset info
df.info()

# Correlation between views and comments
import seaborn as sb
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sb.scatterplot(x='comment_count', y='view_count', data=df)
plt.title('Correlation between Views and Comments')
plt.xlabel('Comments')
plt.ylabel('Views')
plt.show()

3. Machine Learning
Regression: Predicting Views
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare data for regression
X = df[['categoryId', 'comment_count']]  # Select features
y = df['view_count']

# Train the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)

4.Statistical Analysis
Correlation between Views and Comments
correlation = df['view_count'].corr(df['comment_count'])
print(f" Correlation coefficient: {correlation:.2f}")

5.Data Visualization
Barplot for Top Countries
df['country'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title(" Top Countries with Trending Videos")
plt.ylabel("Number of Videos")
plt.show()







