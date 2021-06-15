########################### problem  1 ##########################
import pandas as pd
#loading the affairsset
affairs = pd.read_csv("C:/Users/usach/Desktop/Logistic regression/Affairs.csv")

#2.	Work on each feature of the affairsset to create a affairs dictionary as displayed in the below image
#######feature of the affairsset to create a affairs dictionary
d_type=["count","count","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary","Binary"]

affairs_details =pd.DataFrame({"column name":affairs.columns,
                            "affairs type(in Python)": affairs.dtypes,
                            "Data type":d_type})

#3.	affairs Pre-affairscessing
#3.1 affairs Cleaning, Feature Engineering, etc
#details of affairs 
affairs.info()
affairs.describe()          
#droping index colunms 
affairs.drop(['Unnamed: 0'], axis = 1, inplace = True)
import numpy as np
#creating a new column 
affairs['affairsyn'] =np.where(affairs.naffairs > 0, 1, 0)

#deleting naffairs columns
affairs.drop(["naffairs"] ,axis= 1 ,inplace = True)
#affairs types        
affairs.dtypes
#checking for na value
affairs.isna().sum()
affairs.isnull().sum()
#checking unique value for each columns
affairs.nunique()
"""	Exploratory affairs Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
# covariance for affairs set 
covariance = affairs.cov()
covariance

# Correlation matrix 
co = affairs.corr()
co
affairs.columns
import seaborn as sns
####### graphiaffairs repersentation 
# Boxplot of independent variable distribution for each category of affairsyn 
sns.boxplot(x = "affairsyn", y = "kids", data = affairs)
sns.boxplot(x = "affairsyn", y = "vryunhap", data = affairs)
sns.boxplot(x = "affairsyn", y = "avgmarr", data = affairs)
sns.boxplot(x = "affairsyn", y = "vryrel", data = affairs)


# Scatter plot for each categorical affairsyn of car
sns.stripplot(x = "affairsyn", y = "kids", jitter = True, data = affairs)
sns.stripplot(x = "affairsyn", y = "vryunhap", jitter = True, data = affairs)
sns.stripplot(x = "affairsyn", y = "avgmarr", jitter = True, data = affairs)
sns.stripplot(x = "affairsyn", y = "vryrel", jitter = True, data = affairs)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 

sns.pairplot(affairs, hue = "affairsyn") # With showing the category of each car affairsyn in the scatter plot

# Rearrange the order of the variables
affairs = affairs.iloc[:, [17,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
"""
5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Logistic Regression model.
5.3	Train and Test the data and compare accuracies by Confusion Matrix, plot ROC AUC curve. 
5.4	Briefly explain the model output in the documentation. 
		
 """
from sklearn.model_selection import train_test_split # train and test 
### Splitting the data into train and test data 
# from sklearn.affairsl_selection import train_test_split

train_data, test_data = train_test_split(affairs, test_size = 0.2, random_state=1) # 20% test data

import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.metrics import classification_report
# affairsl building 
# import statsaffairsls.formula.api as sm
logit_affairsl = sm.logit('affairsyn ~ kids + vryunhap +  unhap + avgmarr +  hapavg+  vryhap+  antirel+ notrel+  slghtrel+ smerel+ vryrel+ yrsmarr1+ yrsmarr2+ yrsmarr3+  yrsmarr4+ yrsmarr5+ yrsmarr6 ', data = train_data).fit()

#summary
logit_affairsl.summary2() # for AIC
logit_affairsl.summary()
#prediction
train_pred = logit_affairsl.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(480)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train_data.loc[train_pred > 0.5, "train_pred"] = 1

# classification report
classification = classification_report(train_data["train_pred"], train_data["affairsyn"])
classification

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['affairsyn'])
confusion_matrx

accuracy_train = (347 + 31)/(480)
print(accuracy_train)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train_data["affairsyn"], train_pred)

import matplotlib.pyplot as plt
#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  #AUC = 0.76
roc_auc
# Prediction on Test data set
test_pred = logit_affairsl.predict(test_data)
# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(121)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test_data.loc[test_pred > 0.5, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['affairsyn'])
confusion_matrix
accuracy_test = (80 + 5)/(121) 
accuracy_test
# Based on ROC curv we can say that cut-off value should be 0.70, We can select it
############################problem 2 #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ad_data = pd.read_csv("C:/Users/usach/Desktop/Logistic regression/advertising.csv")
ad_data.describe()
ad_data["Age"].plot.hist(bins=30)
plt.show()

sns.jointplot(data=ad_data, x="Age", y="Area_Income")
plt.show()

sns.jointplot(data=ad_data, x="Age", y="Daily_Time_ Spent _on_Site", kind="kde", color="red")
plt.show()

sns.jointplot(data=ad_data, x="Daily_Time_ Spent _on_Site", y="Daily Internet Usage", color="green")
plt.show()

sns.pairplot(ad_data, hue="Clicked_on_Ad")
plt.show()

X = ad_data[["Daily_Time_ Spent _on_Site", "Area_Income", "Daily Internet Usage", "Male"]]
y = ad_data["Clicked_on_Ad"]
#model bullding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

lm = LogisticRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

print(classification_report(y_test, predictions))

######################problem 3 #################
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

election=pd.read_csv('C:/Users/usach/Desktop/Logistic regression/election_data.csv')
election.columns="ID","Results","Years","Amount","Rank"
election.head()
election.info()
election.loc[election.Results>0,'Results']=1
#Removing first line as the record is empty
election=election.drop(election.index[[0]],axis=0)
election.describe()
# Model building 
#Election ID is not significant for election result hence, preparing model without Election ID
election=election.drop('ID',axis=1)
import statsmodels.formula.api as sm

Model2=sm.logit('Results~Years+Amount', data=election).fit()
Model2.summary()
Model2.summary2()  #AIC=15.80
Pred=Model2.predict(pd.DataFrame(election[['Years','Amount']]))
from sklearn.metrics import confusion_matrix
Confusion_Table=confusion_matrix(election['Results'],Pred>0.45)
Confusion_Table
Accuracy=9/10
Accuracy

########################### problem 4######################
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
bank=pd.read_csv("C:/Users/usach/Desktop/Logistic regression/bank_data.csv")
bank=bank.drop('pdays',axis=1)
bank=bank.drop('previous',axis=1)
#exploratory data analysis
bank.describe()
bank.columns
#model building
#model 1
var=bank.iloc[:,:29]
Model1=sm.logit('y~var',data= bank).fit()
Model1.summary()
Model1.summary2()      #AIC-22695 

#probabilities of variables are greater than 0.05 so we will remove some variables and construct the model
#model2
var2=bank.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,14,15,17,18,19,20,21,22,23,24,25,26,27]]
Model2=sm.logit('y~var2',data =bank).fit()
Model2.summary()
Model2.summary2() # AIC = 11295

#as AIC value for model2 is low we are going to consider model 2
print(np.exp(Model2.params)) #as logistic regression is function of odds
#prediction
pred=Model2.predict(bank.iloc[:,1:]) 
bank["bank_pred"]=pred# creating new column

#confusion matrix
C_Table=confusion_matrix(bank.y,bank.bank_pred>0.5)
C_Table
accuracy = (39012+1703)/(39012+910+3586+1703) 
accuracy  # 90 %

#ROC CURVE AND AUC
from sklearn import metrics
fpr,tpr,threshold = metrics.roc_curve(bank.y, bank.bank_pred)
#PLOT OF ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr) #ocvalue of 0.5
roc_auc

