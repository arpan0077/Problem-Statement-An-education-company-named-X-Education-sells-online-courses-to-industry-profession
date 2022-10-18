#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import time, warnings
import datetime as dt


# # STEP FOR MY MODEL
# 1. Read and understand the data
# 2. Clean the data
# 3. Prepare the data for Model Building
# 4. Model Building
# 5. Model Evaluation
# 6. Making Predictions on the Test Set

# # Reading and Understanding data

# In[21]:


lead_score = pd.read_csv("Leads.csv")


# In[22]:


lead_score.head()


# In[23]:


lead_score.tail()


# In[24]:


print ("shape of data", lead_score.shape, "37 columns")


# In[25]:


lead_score.info()


# # Looks like there are quite a few categorical variables present in this dataset for which we 
# # will need to create dummy variables. 
# # Also, there are a lot of null values present as well, so we will need to treat them accordingly.
we found 3 float and int is 3 
# In[26]:


lead_score.describe()


# In[27]:


# Inspect the different columsn in the dataset

lead_score.columns

for i in lead_score:
    print (i)


# # data clealing and prepartion on model building

# In[28]:


#check the missing value
lead_score.isnull().sum()


# In[29]:





# In[30]:


# Check the summary of the dataset

lead_score.describe(include='all')


# # missing value

# In[32]:


# Drop all the columns in which greater than 3000 missing values are present

for col in lead_score.columns:
    if lead_score[col].isnull().sum() > 3000:
        lead_score.drop(col, 1, inplace=True)
        
print (3000/9240*100 , "we will drop the columns which having more than 32% missing value " ,"taken a whole number " )# we will drop the columns which having more than 32% missing value
print (lead_score.shape, "shape ")


# ### Check the number of null values again

# In[36]:


lead_score.isnull().sum()


# ### I want to drop city and country column as it will no be used in this analysis

# In[ ]:


lead_score.drop(['City'], axis = 1, inplace = True)


# In[41]:




lead_score.drop(['Country'], axis = 1, inplace = True)


# In[43]:


#checking null value percentage
round(100*(lead_score.isnull().sum()/len(lead_score.index)), 2)


# ### will remove the "select " as it,s same as missing value

# In[47]:


for column in lead_score:
    print(lead_score[column].astype('category').value_counts())
    print('############################################################################################')
    print('______________________________________________________________________________________________')
    


# In[48]:


#specilixation , "How did you hear about X Education", "Lead Profile"


# # we have found three column having "select" statement
# 1."specilization"
# 2."How did you hear about X Education"
# 3."Lead Profile"

# In[50]:


print (lead_score['Lead Profile'].astype('category').value_counts())
print("_____________________________________________________________")
print(lead_score['How did you hear about X Education'].value_counts())
print("_____________________________________________________________")
print(lead_score['Specialization'].value_counts())
print("_____________________________________________________________")


# In[52]:


lead_score.drop(['Lead Profile', 'How did you hear about X Education'], axis = 1, inplace = True)

print ("we set criteria about 30% missing value to drop so we are  2 out of 3")


# # Also notice that when you got the value counts of all the columns, there were a few columns in whih only one value was majorly present for all the data points. These include 
# "Do Not Call", `Search`, `Magazine`, `Newspaper Article`, `X Education Forums`, `Newspaper`, `Digital Advertisement`, `Through Recommendations`, `Receive More Updates About Our Courses`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque`. Since practically all of the values for these variables are `No`, it's best that we drop these columns as they won't help with our analysis.

# In[53]:


lead_score.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[56]:


#Better Career Prospects 6528 , rest 1 and 2
print(lead_score['What matters most to you in choosing a course'].value_counts())
print("_____________________________________________________________")

print ("i will drop the other 2 catagory")


# In[60]:


lead_score.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[67]:


#checking column
print(lead_score.columns)
print ("_________________________________ ", "only 14 column")
print (lead_score.shape)
print ("_______________________________________________________________________")
print (lead_score.isnull().sum())


# # as we have set criteria of 30+% for null value to drop now its the time to clean the row null value 

# In[68]:


lead_score = lead_score[~pd.isnull(lead_score['What is your current occupation'])]


# In[69]:


lead_score = lead_score[~pd.isnull(lead_score['TotalVisits'])]


# In[70]:


lead_score = lead_score[~pd.isnull(lead_score['Lead Source'])]


# In[71]:


lead_score = lead_score[~pd.isnull(lead_score['Specialization'])]


# In[72]:



print(lead_score.isnull().sum())
print ("_____________________________________________________________________")

print (" 1.removed all the missing row and data is free of null values")


# In[75]:


#checking the percentage of row 
print(len(lead_score.index), "current row")
print ("_____________________________________________________________________")

print(len(lead_score.index)/9240, "comparing percentage with total 9240.  ")
print ("we still have 68%+ of data")


# In[77]:


#head 6
lead_score.head(6)


# In[80]:


print (lead_score.shape ," <-data shape")
print ("_____________________________________________________________________")
print ("_____________________________________________________________________")
print (lead_score.info())
print ("_____________________________________________________________________")
print ("_____________________________________________________________________")
print (lead_score.describe())
print ("_____________________________________________________________________")
print ("_____________________________________________________________________")


# In[81]:


#Prospect ID , 'Lead Number' need to remove as it is not use for this analysis
lead_score.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[82]:


lead_score.head()


# In[84]:


print (lead_score.shape ," --- only 12 columns")


# # 3.PREPARE OF DATA MODELING

# In[85]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[87]:


sns.pairplot(lead_score,diag_kind='kde',hue='Converted')
plt.show()


# In[92]:


ttpc= lead_score[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]
sns.pairplot(ttpc,diag_kind='kde',hue='Converted')
plt.show()


# In[94]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
transform_ttpc = pd.DataFrame(pt.fit_transform(ttpc))
transform_ttpc.columns = ttpc.columns
transform_ttpc.head()


# In[97]:


sns.pairplot(transform_ttpc,diag_kind='auto',hue='Converted')
plt.show()


# # Dummy variable creation || checking categorical variables
# The next step is to deal with the categorical variables present in the dataset. 
# So  taking  a look at which variables are actually categorical variables.

# In[102]:


temp_object = lead_score.loc[:, lead_score.dtypes == 'object']
temp_object.columns


# # Create dummy variables using the 'get_dummies' command
# 

# In[103]:


# Create dummy variables using the 'get_dummies' command fsb
dummy_data = pd.get_dummies(lead_score[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True)

# Add the results to the master dataframe
lead_score = pd.concat([lead_score, dummy_data], axis=1)


# In[104]:


lead_score.head()


# # separate dummy variable for 'Specialization' due to level "select" || drop

# In[105]:


# Creating dummy variable separately for the variable 'Specialization' since it
#has the level 'Select' which is useless so we
# drop that level 

dummy_Specialization = pd.get_dummies(lead_score['Specialization'], prefix = 'Specialization')
dummy_Specialization = dummy_Specialization.drop(['Specialization_Select'], 1)
lead_score = pd.concat([lead_score, dummy_Specialization], axis = 1)


# In[108]:


lead_score.head(1)


# In[109]:


# Droping the variables for which the dummy variables created

lead_score = lead_score.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[110]:


lead_score.head(1)


# In[111]:


print ( "from 136 >>128 columns")


# # train_test_split

# In[112]:




from sklearn.model_selection import train_test_split


# In[113]:


# Put all the feature variables in X

X = lead_score.drop(['Converted'], 1)
X.head()


# # target variable || "converted"

# In[115]:


# Put the target variable in y

y = lead_score['Converted']

y.head()


# In[116]:


# Split the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Trying to scale the numeric variable

# In[117]:


# Import MinMax scaler

from sklearn.preprocessing import MinMaxScaler


# In[118]:


# Scale the three numeric features present in the dataset

scaler_3 = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler_3.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# # checking corelation through heat map

# In[120]:


# Looking at the correlation table
plt.figure(figsize = (40,40))
sns.heatmap(lead_score.corr())
plt.show()


# # model building using RFE

# In[121]:


# Importing 'LogisticRegression' and create a LogisticRegression object

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[122]:


logreg


# # Import RFE and select 15 variables

# In[123]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)# running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[125]:


# Let's take a look at which features have been selected by RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[128]:


# Puting all the columns selected by RFE in the variable 'column_1'

column_1= X_train.columns[rfe.support_]


# # using p value and and vif trying to build a logistic regression

# In[129]:


# Select only the columns selected by RFE

X_train = X_train[column_1]
X_train


# In[132]:


X_train
for i in X_train:
    print (i)


# In[133]:


# Import statsmodels

import statsmodels.api as sm


# In[134]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary
#using binomial
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[136]:


# Import 'variance_inflation_factor' "vif"

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[137]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif


# In[157]:


vif['Features'] = X_train.columns
vif.head(35)


# In[ ]:





# In[153]:


vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif.head()


# In[151]:


vif['VIF'] = round(vif['VIF'], 2)
vif.head()


# In[148]:


vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[159]:


X_train.head()


# In[160]:


X_train.drop('Lead Source_Olark Chat', axis = 1, inplace = True)


# In[162]:


X_train.head()


# In[163]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[165]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[166]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[167]:



vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[169]:


X_train.columns


# In[170]:


X_train.drop('What is your current occupation_Working Professional', axis = 1, inplace = True)
X_train.drop('Last Notable Activity_Unreachable', axis = 1, inplace = True)


# In[171]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[172]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[173]:


X_train.drop('What is your current occupation_Student', axis = 1, inplace = True)


# In[174]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[175]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[179]:


y_train


# In[188]:


# Import metrics from sklearn for evaluation

from sklearn import metrics


# In[196]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[197]:


X_train.columns


# In[198]:


X_train.drop('Do Not Email_Yes', axis = 1, inplace = True)


# In[199]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[200]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[202]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[203]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[204]:



X_train.drop('What is your current occupation_Unemployed', axis = 1, inplace = True)


# In[205]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[206]:


# Import 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[209]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]


vif


# In[212]:



X_train.shape


# In[213]:


X_test_sm = sm.add_constant(X_test)


# In[216]:


X_test_sm


# In[217]:


y_test_pred = res.predict(X_test_sm)


# In[ ]:




