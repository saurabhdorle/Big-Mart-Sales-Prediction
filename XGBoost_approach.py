# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')
y = dataset.iloc[:, -1].values # dependent values
dataset.drop(['Item_Outlet_Sales'], axis =1, inplace = True )
#check the data
dataset.describe()

# =============================================================================
# # check the categorical data
# =============================================================================
dataset.Item_Fat_Content.value_counts()

dataset.Item_Type.value_counts()

dataset.Outlet_Size.value_counts()

dataset.Outlet_Location_Type.value_counts()

dataset.Outlet_Type.value_counts()

'''
# =============================================================================
# # Missing values
# ============================================================================='''

data_missing = dataset.isnull().sum()
print(data_missing)

# Numpy array for imputing missing values
X = dataset.iloc[:, :-1].values

# =============================================================================
## Missing Categorical Values
# =============================================================================
from sklearn_pandas import CategoricalImputer

data = np.array(X[:,8], dtype=object)
imputer = CategoricalImputer()
X[:,8] = imputer.fit_transform(data)
dataset['Outlet_Size'] = X[:,8]

# =============================================================================
# # Imputer for numeric values
# =============================================================================

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, 1:2] = imputer.fit_transform(X[:, 1:2])
dataset['Item_Weight'] = X[:,1:2] 

# Check Values in Item Visibilty
dataset.Item_Visibility.value_counts()
# Replace 0 with NaN
dataset['Item_Visibility'].replace(0.000000, np.nan, inplace=True)
# replace NaN with Mean value
X[:, 3] = dataset['Item_Visibility']
X[:, 3:4] = imputer.fit_transform(X[:, 3:4])
dataset['Item_Visibility'] = X[:, 3:4]

dataset.Item_Visibility.isnull().sum()




'''
# =============================================================================
# Categorical values to Numeric
# ============================================================================='''

#Fat_Content has some redudancy
dataset.Item_Fat_Content.value_counts()
# Removing redundancy
dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace(
                                {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'})

#X[:, 2] = dataset['Item_Fat_Content']


dataset['Outlet_Years'] = 2018 - dataset['Outlet_Establishment_Year']

Mean_Visibility=dataset['Item_Visibility'].mean()

dataset['Item_Visibility_MeanRatio']=dataset.apply(lambda x:x['Item_Visibility']/Mean_Visibility,axis=1)



#Convert categorical into numerical 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']
labelencoder = LabelEncoder()

dataset['Item_Identifier'] = labelencoder.fit_transform(dataset['Item_Identifier'])
dataset['Outlet_Identifier'] = labelencoder.fit_transform(dataset['Outlet_Identifier'])

for i in var_mod:
    dataset[i] = labelencoder.fit_transform(dataset[i])




# =============================================================================
# Model Training and testing
# =============================================================================
X = dataset.iloc[:,:].values

# Splitting Train Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test) # predicting test set

from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(y_test, y_pred)) # Calculating root mear square error
print(rms)

'''
Save Model
from sklearn.externals import joblib
# save the model to disk
filename = 'model_xgbreg.sav'
joblib.dump(regressor, filename)'''
