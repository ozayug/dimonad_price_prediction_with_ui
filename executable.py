# %% [markdown]
# # <font color='Red'>Project Title : ***Diamonds Price Prediction*** </font>

# %% [markdown]
# # <font color='Blue'> Description</font>

# %% [markdown]
# 
# The "Diamonds Price Prediction" project leverages the power of machine learning to accurately forecast the prices of diamonds. With the diamond market being highly dynamic and influenced by various factors, this project aims to provide reliable predictions to assist buyers, sellers, and investors in making informed decisions.
# 
# The project involves several key steps. Firstly, the dataset is preprocessed to handle missing values, outliers, and categorical variables, ensuring the data is in a suitable format for training the machine learning model. Feature engineering techniques may also be employed to extract additional relevant information and enhance the predictive capabilities of the model.
# ![image-2.png](attachment:image-2.png)
# 

# %% [markdown]
# ## <font color ='Green'>1) Loading Libraries</font>

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as pp


# %%
df = pd.read_csv('diamonds.csv')


# %%
df.head()

# %%
df.shape  # dataset has 53940 rows and 10 columns

# %% [markdown]
# **Observation:**
# <font color='Red'>1) We have Total 53940 number of Rows and 10 Columns in the Dataset.</font>

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# <font color='orange'>**Data Preprocessing have some steps involved:**</font>
# 
# 1)Data cleaning
# 
# 2)Identifying and removing outliers
# 
# 3)Encoding categorical variables **

# %%
df.info()
# Display information about the DataFrame, including column names, data types, and memory usage

# %% [markdown]
# ### <font color='purple'> Checking Null Values</font>

# %%
#Checking for the null values
df.isnull().sum()

# %% [markdown]
# 
# **Observation:*** 
# we don't have any null values present in our Dataset.

# %% [markdown]
# ## Missingno

# %% [markdown]
# The missingno library is used to visualize missing (null or NaN) values in a DataFrame. It provides a clear and concise way to understand the distribution and patterns of missing data in your dataset.
# 
# 

# %%
pip install missingno

# %%
import missingno as msno 
msno.bar(df)


# %% [markdown]
# **observation:**
# Not Any missing values are present in any Features..

# %%
df['carat'].unique()

# %%
#Checking for unique values in cut column
df['cut'].unique()

# %%
df['clarity'].unique()

# %%
df['color'].unique()

# %%
df.describe()

# %%
pp.ProfileReport(df)

# %%
# Select numeric columns
numeric_columns = df.select_dtypes(include=[np.number])

sns.set(style="whitegrid")

# Set the figure size before creating the heatmap
plt.figure(figsize=(12, 12))

# Create the heatmap for numeric columns
heatmap = sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')

# Display the plot
plt.show()


# %% [markdown]
# <font color='brown'>**Observation:**</font>
# 1) "x", "y" and "z" show a high correlation to the target column.
# 
# 2) "depth" and "table" show low correlation.

# %% [markdown]
# # <font color='Dark Blue'> Comparing 'cut' and 'price' features by using Bar Graph</font>

# %%


sns.set(style="whitegrid")

# Trim whitespaces from 'cut' column
df['cut'] = df['cut'].str.strip()

# Create a bar plot for 'cut' and 'price'
plt.figure(figsize=(10, 6))
sns.barplot(x='cut', y='price', data=df, order=df['cut'].unique())

# Add labels and title
plt.xlabel('Cut')
plt.ylabel('Price')
plt.title('Bar Graph: Price vs Cut')

# Show the plot
plt.show()



# %% [markdown]
# ### <font color='Gray'> Observation:</font>

# %% [markdown]
# <font color='Chocolate'>1)Premium Cut have High Prices than Other cuts</font>  
#  <font color='Dark Blue'>2) Ideal have low price than other cuts</font> 

# %% [markdown]
# # <font color='Dark orange'> Comparing 'Carat' and 'price' features by using  scatterplot</font>

# %%


# Creating a scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(x='carat', y='price', data=df, s=100, color='blue', alpha=0.7)

# Adding labels and title
plt.title('Carat vs Price')
plt.xlabel('Carat')
plt.ylabel('Price')

# Show the plot
plt.show()

# %% [markdown]
# ### <font color='Red'>**Observation:**</font>
# <font color ='Blue'>1) Prices is increasing for higher carats..

# %% [markdown]
# ## <font color= 'Red' > Price VS Clarity by using pie chart</font>

# %%
# Calculate the total charges for each clarity
prices_by_clarity = df.groupby('clarity')['price'].sum()

explode = [0, 0, 0.1, 0,0,0,0,0]

#Set the colors
colors = ['skyblue', 'lightpink', 'lightgreen', 'yellow','beige','violet','indigo','orange']

# Create the pie chart
plt.pie(prices_by_clarity, labels=prices_by_clarity.index, autopct='%2.1f%%',colors=colors,explode=explode, shadow=True, startangle=90)

# Set aspect ratio to be equal to make the pie circular
plt.axis('equal')

#title
plt.title('Clarity vs Prices', fontsize=16, fontweight='bold')

# %% [markdown]
# ## **Observation:**
# 
# 1) Higher Proportion of Prices are contributed by the Sl1 clarity Level.  
# 2) Lower Proportion of Prices are contributed by the l1 clarity Level. 

# %% [markdown]
# # <font color='Red'>scatter graph</font>

# %%
## Grouping the data by depth and calculating the mean of Prices
grouped_data = df.groupby('depth')['price'].mean().reset_index()

# Creating the scatter plot
plt.scatter(grouped_data['depth'], grouped_data['price'])
plt.title('Depth vs Price')
plt.xlabel('Depth')
plt.ylabel('Price')

# %% [markdown]
# ## **Observation:**
# 
# 1) Highest Price is Noted when the Depth is Between 65 to 70.  
# 2) more than 50% of Price lies between 3000 to 6000.  
# 3) 50% of the Diamond's Depth is in Between 60 to 70.
# 

# %% [markdown]
# 
# # <font color='Brown'>Converting Categorical variable into numeric values</font>

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# %%
Le = LabelEncoder()
Ohe = OneHotEncoder(sparse_output=False, drop='first')

# %% [markdown]
# # LabelEncoder

# %%
df['color'] = Le.fit_transform(df['color'])

# %%
df['clarity']=Le.fit_transform(df['clarity'])

# %%
df

# %% [markdown]
# # <font color='Blue'> OneHotEncoder<font>

# %%
encoded_features = Ohe.fit_transform(df[['cut']])

# %%
new_columns = Ohe.get_feature_names_out(['cut'])
print(new_columns)

# %%
# Create a new DataFrame with the encoded features
df_encoded = pd.DataFrame(encoded_features, columns=new_columns) 

# %%
# Concatenate the original DataFrame and the encoded DataFrame
df = pd.concat([df,df_encoded],axis=1)

# %%
df.head()

# %%
#Checking the name of the total columns present in df
df.columns

# %% [markdown]
# # Heatmap

# %%
df.drop(columns='cut',axis=1,inplace=True)
# Compute the correlation matrix .
import pandas as pd



numeric_df = df.select_dtypes(include=['float64', 'int64'])


corr_matrix = numeric_df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='rainbow')
plt.title('Correlation Heatmap')
plt.show()


# %% [markdown]
# # Creating the independent variable(X) and dependent variable(y)

# %%
# Extract features (X) and target variable (y)
x = df[['carat', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z',
       'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good']]
y = df[['price']]

# %%
# Convert X and y to numpy arrays
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

# %%
df.head()

# %%
x

# %%
y

# %% [markdown]
# # Splitting the data set into train and test using train_test_split from sklearn.model_selection

# %%
from sklearn.model_selection import train_test_split

# %%
x_train,x_test,y_train,y_test = train_test_split(df.drop('carat',axis=1),df['carat'],test_size=0.2)

# %%
df.head()

# %%
print('x_train_shape: ',x_train.shape)
print('y_train_shape: ',y_train.shape)
print('x_test_shape: ',x_test.shape)
print('y_test_shape: ',y_test.shape)

# %%
y_train = y_train.values.reshape(-1, 1)
y_test= y_test.values.reshape(-1, 1)

# %% [markdown]
# # RobustScaler 
# 1)To handle the outliers

# %%
from sklearn.preprocessing import RobustScaler

# %%
#Creating the object of Robust
Rb = RobustScaler()

# %%
x_train = Rb.fit_transform(x_train)


# %%
x_test = Rb.transform(x_test)

# %%
y_train = Rb.fit_transform(y_train)

# %%
y_test = Rb.transform(y_test)

# %% [markdown]
# ## <font color= 'Brown'> Creating LinearRegression Model</font>

# %%
x_mean = x_train.mean(axis=0)

# %%
y_mean = y_train.mean(axis=0)

# %% [markdown]
# <font color ='sky Blue'> These lines calculate the mean values along axis 0 for the x_train and y_train arrays. This is done to center the data around the mean, which is a common step in linear regression.**</font>

# %%
num = 0 #Initialize variables.
dim = 0
epsilon = 1e-8  
for i in range(len(x_train)):
    num += (x_train[i] - x_mean) * (y_train[i] - y_mean)
    dim += (x_train[i] - x_mean) ** 2

coff = num/(dim + epsilon)#coff is the coefficient (slope) of the linear regression model.
inter = y_mean - (coff * x_mean)#inter is the intercept of the linear regression model.
print('Coff:', coff)
print('Intercept:', inter)

# %% [markdown]
# **1)num is used to accumulate the numerator of the coefficient calculation.**  
# **2)dim is used to accumulate the denominator of the coefficient calculation.**    
# **3)epsilon is a small constant added to the denominator to avoid division by zero.**  
# 
# <font color='red'>This loop iterates through each data point in the training set and updates the values of num and dim for the coefficient calculation.</font>
# 
# **num is the sum of the product of the differences between each x and y and their respective means.**  
# **dim is the sum of the squared differences between each x and its mean.**  

# %%
coff.shape

# %%
m = coff
c = inter
y = m * 11 + c # y=Mx + c. x=11 input value here. This line calculates the predicted value of y
z = m * 95644.50 + c   #This line calculates the predicted value of z.
print('y:', y)
print('z:', z)

# %%
from sklearn.metrics import mean_squared_error,mean_absolute_error

# %%
# Evaluate the model on the test set
y_pred = x_test.dot(coff)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)

print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):',mae)
print('Root Mean Squared Error (RMSE): ',rmse)

# %%
# Visualize the results
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression - Actual vs. Predicted Price')
plt.show()

# %% [markdown]
# #   LinearRegression Through Function

# %%
from sklearn.linear_model import LinearRegression

# %%
Lr = LinearRegression()

# %%
Lr.fit(x_train,y_train)

# %%
y_hat = Lr.predict(x_test) # y_hat is predicted values.

# %%
from sklearn.metrics import r2_score
r2_score(y_test,y_hat)

# %% [markdown]
# ## **Observation:**  
# <font color='Red'>1)In this model, an R-squared score of approximately 0.96 means  model explains about 96.20% of the variance in the test data. </font>  
# <font color='Red'>2)the model's predictions are  capturing a significant portion of the variability present in the actual data.</font>  
# 
# **A higher R-squared value (closer to 1) would indicate better explanatory power.**
#     
#     
# 

# %% [markdown]
# ## 1) Finding
# MSE(mean_squared_error)  
# MAE(mean_absolute_error)  
# RMSE(root_mean_squared_error)

# %%
print(mean_squared_error(y_test,y_hat))
print(mean_absolute_error(y_test,y_hat))
print(np.sqrt(mean_squared_error(y_test,y_hat)))

# %% [markdown]
# **The Lower value of MSE,MAE,RMSE is indicating Better Model's Performances.** 

# %% [markdown]
# <font color =' Green'>In conclusion, the "Diamonds Price Prediction" project utilizes machine learning to accurately forecast diamond prices. By analyzing a comprehensive dataset and employing advanced algorithms, the project empowers stakeholders with data-driven decision-making, enhancing transparency and efficiency in the diamond market. Real-time predictions enable informed transactions, benefiting buyers, sellers, and investors.</font>

# %%
import joblib

# Linear Regression model save
joblib.dump(Lr, 'model.pkl')

# RobustScaler save
joblib.dump(Rb, 'scaler.pkl')


# %% [markdown]
# <font color='Brown'>***Thank you...***</font>

# %%


# %%



