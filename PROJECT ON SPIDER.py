# solving chance of admit problem using Multiple Linear Regression

####################### Importing the libraries ###############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm
import seaborn as sns
###############################################################################


########################## Importing the dataset###############################
dataset = pd.read_csv('ad.csv')
x = dataset.iloc[:,0:7].values        # x consists all the independent variable
y = dataset.iloc[:, -1].values              # y consists the dependent variable
###############################################################################


########################### Encoding categorical data #########################
labelencoder = LabelEncoder()            #create object 1 of LabelEncoder class
labelencoder1 = LabelEncoder()           #create object 2 of LabelEncoder class
x[:, 6] = labelencoder1.fit_transform(x[:, 6]) #fitting the labelencoder object 1 and transform the research column
x[:, 2] = labelencoder.fit_transform(x[:, 2])  #fitting the labelencoder object 2 and transform the university ranking column

onehotencoder = OneHotEncoder(categorical_features = [2]) #making dummy variable for university ranking column
x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]    # Avoiding the Dummy Variable Trap, now x includes GRE_SCORE, TOEFL_SCORE, LOR, CGPA, Research columns
###############################################################################


###################3finding optimal variables##################################
x=np.append(arr = np.ones((400,1)).astype(int) , values = x , axis = 1) 
              #appending the column of ones in the variable x used in OLS class

def backwardElimination(x, sl):               #function of backward elimination
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05               #setting up the threshold value for backward eliminaion
x_opt = x[:,:]                               #taking all columns of x in  x_opt
x_Modeled = backwardElimination(x_opt, SL)#calling backward elimination function
x=x_Modeled                                  #remove the columns of one 
regres = sm.OLS(y, x).fit()

###############################################################################


########## Splitting the dataset into the Training set and Test set############
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
###############################################################################


############################ Feature Scaling###################################
sc_x = StandardScaler()               #create object 1 of Standard Scaler class
sc_y = StandardScaler()               #create object 2 of Standard Scaler class
x_train = sc_x.fit_transform(x_train)                #transform the x_train set
x_test = sc_x.transform(x_test)                       #transform the x_test set
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)) #transform the y_train set
###############################################################################


######### Fitting Multiple Linear Regression to the Training set###############
regressor = LinearRegression()      #creating object of linear Regression class
regressor.fit(x_train, y_train)  #fitting the regression model on x and y train 
###############################################################################


################### Predicting the Test set results############################
y_pred = regressor.predict(x_test)               #predicting the x_test results
y_pred = sc_y.inverse_transform(y_pred)
###############################################################################


############################user interface and graph###########################
def view():
    print("\nThe data set used have 8 columns(features) and 400 rows(observations)\n")
    print("The columns are named as GRE_score(Graduate Record Examinations) , TOEFl_score( Test of English  as a Foreign Language), University Ranking ,SOP(Satement of proof), LOR(Letter of Recommendation) ,CGPA(Cumulative Grade Point Average) ,Research(is student done a  research in any field or not) ,chance_of_Admit(Dependent variable) ")
    print("\npress 1 to view the dataset\n")
    a=int(input())
    if a==1:
        print(dataset)
        
def correlation():
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(dataset.corr(method='pearson'),annot = True,cmap = 'Blues')
    plt.show()
def graphs():
    i1='y'
    while i1=='y':
        print("\nplease enter which graph you want to see ")
        print("press 1 for view the GRE_Score vs Chane_of_Admit garph")
        print("press 2 for view the TOEFL_Score vs Chane_of_Admit garph")
        print("press 3 for view the LOR vs Chane_of_Admit garph")
        print("press 4 for view the CGPA vs Chane_of_Admit garph")
        print("press 5 for view the how many applicant done any research or not")
        n1=int(input())
        if n1==1:
            sns.regplot(dataset['GRE_Score'],dataset['Chance_of_Admit'])
            plt.title("GRE Score vs Chance of Admit")
            plt.show()
        elif n1==2:
            sns.regplot(dataset['TOEFL_Score'],dataset['Chance_of_Admit'])
            plt.title("TOEFL Score vs Chance of Admit")
            plt.show()
        elif n1==3:
            l=dataset.iloc[:,4].values
            sns.regplot(l,dataset['Chance_of_Admit'])
            plt.title("LOR vs Chance of Admit")
            plt.xlabel("LOR")
            plt.show()
        elif n1==4:
            sns.regplot(dataset['CGPA'],dataset['Chance_of_Admit'])
            plt.title("CGPA Vs Chance of Admit ")
            plt.show()
            
        elif n1==5:
            plt.figure(figsize=(8,6))
            R = ['No Research Experience','With Research Experience']
            ypos = np.arange(len(R))
            sns.countplot(dataset['Research'])
            plt.title('Research Experience')
            plt.ylabel('No. of Applicant')
            plt.xticks(ypos,R)
            plt.show()
            
        print("do you want to view other graph then press y otherwise press N/n ")
        i1=input()
    
def results():
    print("\nThe test set result is\n")
    print(y_pred)
    sns.regplot(y_pred,y_test)
    plt.title("prediction vs test result")
    plt.xlabel("prediction of test set")
    plt.ylabel("test set values")
    plt.show()
    
def predic():
    p=[1]
    print("\nFor predicting a new value you have to enter some information \n like GRE_Score , TOEFL_Score ,LOR , CGPA and Research of the student\n ")
    p.append(int(input("Enter the GRE_Score of the student\n")))
    p.append(int(input("Enter the TOEFL_Score of the student\n")))
    p.append(float(input("Enter the LOR information of the student\n")))
    p.append(float(input("Enter the CGPA of the student\n")))
    p.append(int(input("Press 1 if student done a research and 0 if student did not done a research\n")))
    p = np.array(p).reshape(1,6)
    p[:, 5] = labelencoder1.transform(p[:, 5])  
    p = sc_x.transform(p)
    p_pred = regressor.predict(p)
    p_pred = sc_y.inverse_transform(p_pred)
    print("the result of prediction is ",p_pred)

def summ():
    print("\nThe dataset include 8 variables")
    print("Based on the backward Eloimination method of the independent variables ")
    print("We identify some features which have p value less than 0.05 ")
    print("These features are GRE_Score ,TOEFL_Score,LOR ,CGPA ,Research")
    print("The summary of fitting these feature with dependent variable is as follows\n")
    print(regres.summary())
    
print("\n This programm is for predicting that the student will get the admission in the college or not based on pevious data\n")
i='y'
while i=='y':
    print("enter your choice ")
    print("press 1 for view the dataset(train + test)")
    print("press 2 for view the correlation table between the features of dataset")
    print("press 3 for view the graphs between independent and dependent variables")
    print("press 4 for view the test set result")
    print("press 5 for predicting a new value")
    print("press 6 for view some essential information")
    n=int(input())
    if n==1:
        view()
    elif n==2:
        correlation()
    elif n==3:
        graphs()
    elif n==4:
        results()
    elif n==5:
        predic()
    else:
        summ()
    print("\ndo you want to view another things in the model then press y otherwise press N/n ")
    i=input()
    





