# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:21:09 2020

@author: VRUTIKA

"""

# importing required libraries
import warnings
warnings.filterwarnings('ignore') # to filter warings
import pandas as pd # python library for data manipulation
import matplotlib.pyplot as plt # python library for plottng graphs and charts
plt.style.use('ggplot') 
import numpy as np # Library for manipulating large arrays
import seaborn as sns # data visualization library based on matplotlib
#import utils 
from scipy.stats import ttest_1samp # module contain number of probability ditribution


def clear(): # function to clear screen
    pl = input("Press Y to clear the screen(Press any other key to continue)?:").lower()
    if pl == 'y':
        # "\033{2J" == \Esc2J
        print("\033[2J")

# 1.Structure of Dataset.
#    iris has 6 fields including Id, it has one class variable species
# 2.Datatype of each column
# Using info() finding the above values    
def data_structure(xls):
    #print("\n-----Columns-----")
    #for col in xls.columns:
    #    print(col)
    
    print("\n-----Struture-----")
    xls.info()

    print("\n-----Sample Data-----")
    print(xls.head()) # display top five records

def cleaning(xls):    
    # alpha numeric length
    for col in xls.columns:
        if xls[col].dtype == 'object':
            print("Length of AlphaNumeric Column:",col,":",xls[col].map(lambda x: len(str(x))).max())
    
    # checking alpha numeric columns for obvious errors
    print("\n-----checking alpha numeric columns for obvious errors-----")
    
    #values of aplha numeric columns after solving above errors
    for col in xls.columns:
        if xls[col].dtype == 'object':
            print(xls.groupby(col).size())
     
    return xls

#converting alpha-numeric columns to numeric using map()
def transform(xls):
    for col in xls.columns:
        d = {}
        if xls[col].dtype == 'object':
            xs = pd.DataFrame(xls.groupby(col,as_index = False))
            #print(xs[0])
            xlist = xs[0].tolist()
            for val in xlist:
                if val.lower() == 'unknown':
                    d[val] = -1
                    continue
                d[val] = xlist.index(val)
            xls[col] = xls[col].map(d)
    return xls


def imputation(xls):
    
    # 5.	What are the significant columns?
    #       Significant columns are columns that add meaningful data to dataset.
    #       (After removing Id, Name, descriptive columns whatever remains are significant columns)
    # 6.	Identify significant columns of the dataset.
    
    sgft = []
    # identify significant columns by dropping other columns
    for i in xls.columns:
        if xls[i].dtype == 'object' or i == 'Id':
            continue
        sgft.append(i)
    print(sgft)
    
    # 7.	Find out for each significant column
    # 	Number of Null values
    #   Number of zeros
    print("\n-----Null values and Zeros Before Imputation-----")
    for i in xls.columns:
        if xls[i].dtype == 'object' or i.lower() == 'id':
            continue
        print("Null values in ",i," : ",xls[i].isnull().sum()) # counting null(using isnull()) 
        print("Zeros in ",i," : ",xls[i].isin([0]).sum(),"\n") # and zero values in each column
    
    # 9.	For each numeric column
    #   Replace null values & zeros with median value of the column.
    for i in xls.columns:
        if xls[i].dtype == 'object' or i.lower() == 'id':
            continue
        d_mean = xls[i].median()
        xls[i] = np.where(xls[i].isnull(), d_mean, xls[i]) # using where() from numpy
        xls[i] = np.where(xls[i] == 0, d_mean, xls[i]) # to replace mean with null and zero
    
    print("\n-----Null values and Zeros After Imputation-----")
    for i in xls.columns:
        if xls[i].dtype == 'object' or i.lower() == 'id':
            continue
        print("Null values in ",i," : ",xls[i].isnull().sum()) # counting zero and null after
        print("Zeros in ",i," : ",xls[i].isin([0]).sum(),"\n") # replacement
        
    # 8.	For each significant column
    # 	Provide the obvious errors
    print("\n-----Obvious Errors-----")
    pd.options.display.float_format = '{:,.2f}'.format
    for colName in xls.columns:
        if colName.lower() == 'id' or xls[colName].dtype == 'object':
            continue
        print("Unique values in",colName,"\n",xls[colName].unique())
        
    # 4.    Precision & scale of numeric columns
    print("\n-----Precision and Scale before-----")
    for col in xls.columns:
        if xls[col].dtype == 'object' or col.lower() == 'id':
            continue
        print(col,"Precision:",xls[col].map(lambda x: len(str(float(x)))-1).max(), #calculating precision and 
          "Scale:",xls[col].map(lambda x: len(str(float(x)).split('.')[1])).max()) # scale using lamda function
    
    print("\n-----resolving Obvious Errors-----")
    
    for col in xls.columns:
        if xls[col].dtype == 'object' or col.lower() == 'id':
            continue
        xls[col]=round(xls[col],2) # making precision and scale equal for all values
    
    for col in xls.columns:
        if col == 'Id' or xls[col].dtype == 'object':
            continue
        print("Unique values in",col,"\n",xls[col].unique())   
    
    print("\n-----Precision and Scale After-----")
    for col in xls.columns:
        if xls[col].dtype == 'object' or col.lower() == 'id':
            continue
        print(col,"Precision:",xls[col].map(lambda x: len(str(float(x)))-1).max(),
          "Scale:",xls[col].map(lambda x: len(str(float(x)).split('.')[1])).max()) 
    
    return xls
        

# 10.	For each significant column
# Provide the quartile summary along with the count, mean & sum
# 11.	For each significant column
# Provide the range, variance and standard deviation
def summary_data(xls):
    print("\n-----Measures of Central Tendency and Statistical Dispersion-----")
    des_df = xls.drop(['id'],axis = 1)
    des_r = des_df.describe() # describe() gives us mean,min,max,median,1Q,3Q,std
    var_r = des_df.var() # calulating variance seperately
    varlist = []
    for col in des_r.columns: # making variance scale = 5
        varlist.append(round(var_r[col],5))
        
    sumlist = []
    for i in xls.columns.tolist():
        if i.lower() == 'id' or xls[i].dtype== 'object':
            continue
        sumlist.append(xls[i].sum())
    
    sum_r = pd.DataFrame([sumlist],columns=des_r.columns, index=['sum'])
    des_r = des_r.append(sum_r)
    var_r = pd.DataFrame([varlist],columns=des_r.columns, index=['var']) # putting results of variance into dataframe
    mct = des_r.append(var_r) # adding variance result to describe data frame
    print(mct) # printing all results together


# 12. Provide the count of outliers and their value
## Below given 4 functions are copied from utils.py file
## Taking scale is changed to 1.5 from 3.0 

# returns: count of outliers in the colName
# usage: colOutCount(colValues)
def colOutCount(colValues):
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(ndOutData)
    return ndOutData.size

# returns: actual outliers values in the colName
# usage: colOutValues(colValues)
def colOutValues(colValues):
    quartile_1, quartile_3 = np.percentile(colValues, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 3.0)
    upper_bound = quartile_3 + (iqr * 3.0)
    ndOutData = np.where((colValues > upper_bound) | (colValues < lower_bound))
    ndOutData = np.array(colValues[ndOutData])
    return ndOutData

# returns: actual of outliers in each column of dataframe
# usage: OutlierValues(df):
def OutlierValues(df): 
    colNames = df.columns
    strRetValue = ""
    for colName in colNames:
        if df[colName].dtype == "object": 
            continue
        colValues = df[colName].values
        #print('Column: ', colName)
        strRetValue = strRetValue + colName + " " + "\n"
        strRetValue = strRetValue + str(colOutValues(colValues)) + " \n"
        #print(colOutValues(colValues))
        #print(" ")
    return(strRetValue)


# returns: count of outliers in each column of dataframe
# usage: OutlierCount(df): 
def OutlierCount(df): 
    colNames = df.columns
    strRetValue = ""
    for colName in colNames:
        #print(colName)
        colValues = df[colName].values
        #print(colValues)
        outCount = colOutCount(colValues)
        #print(outCount)
        strRetValue = strRetValue + colName.ljust(15, ' ') + "   " + str(outCount) + "\n"
    return(strRetValue)


# 13.	Are there any class variables? If yes,
# 	provide frequency distribution table & chart for the same            
def fre_dist(xls):
    for colName in xls.columns:
        if xls[colName].dtype == 'object':
            print("\n*"+colName+"*")
            #print(df.groupby(colName).size()) # display frequency table
            #print("")
            plt.figure() # initialize graph
            sns.countplot(xls[colName],label="Count") # plot the distribution
            plt.title(colName) # provide title
            plt.show() # show plot
            

# 14.	For all numeric columns
# 	Provide histogram
def histograms(xls):
    colNames = xls.columns.tolist() # make a list of columns present in dataframes
    for colName in colNames:
        if xls[colName].dtype == "object" or colName.lower() == 'id': # skip Id and string columns
            continue
        colValues = xls[colName].values # get array of values
        plt.figure() # initialize graph
        sns.distplot(colValues, bins=5,kde=False, color='g') # plot the histogram
        plt.title(colName) # provide title
        plt.ylabel(colName) # name Y axis
        plt.xlabel('Bins') # name X axis
        plt.show() # show the plot


# 15.	For all numeric variables
# 	Provide box & whisker plot        
def boxplots(xls):
    colNames = xls.columns.tolist() # make a list of columns present in dataframes
    for colName in colNames:
        if xls[colName].dtype == "object" or colName.lower() == 'id': # skip Id and string columns
            continue
        plt.figure()
        sns.boxplot(y=xls[colName], color='g') # plot the boxplot
        plt.title(colName) # provide title
        plt.ylabel(colName) # name Y axis
        plt.xlabel('Bins') # name X axis
        plt.show() # show the plot


# 16.	For all numeric variables
# 	Provide correlation table & graph        
def corelations(xls):
    #Co-relation for numeric columns
    print("\n-----Co-relation Table-----")
    pd.options.display.float_format = '{:,.2f}'.format # limit diplay to 2 decimal places
    cor_xls = xls.drop('id', axis = 1)
    dfc = print(cor_xls.corr()) # get co-relation table
    print(dfc) # print co-relation table
    # check relation with corelation - heatmap
    print("\n-----Co-relation Graph-----")
    plt.figure(figsize=(9,9)) # initialize graph
    ax = sns.heatmap(cor_xls.corr(), annot=True, cmap="PiYG") # plot heatmap
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+0.5, top-0.5)
    plt.show() # show plot


# 17.	Based on the correlation table
# 	Provide scatter plot relevant columns    
def scatter_plot(xls):
    cor_xls = xls.drop('id', axis = 1)
    dfc = pd.DataFrame(cor_xls.corr()) # get co-relation table into Dataframe
    c = 0   # counter for traversal
    tr = int(dfc.size/2) # range for traversal
    for i in dfc.index:
        if c > tr:      # stop after half traversal
            break
        for j in dfc.columns:
            c = c + 1
            if i == j:
                continue
            if dfc[i][j] > 0.7 and dfc[i][j] < 1:
                plt.figure()
                sns.regplot(data=xls, x=i, y=j, color= 'b', scatter_kws={"s": 5}) # plot scatter
                plt.title(i + ' v/s ' + j) #provide title
                plt.show() # show plot    


# Driver code
exe = True
while(exe == True):
    print("\033[2J")
    print("-----EDA/VDA on iris.xlsx-----")
    
    try:
        #Extract data from file using ExcelFile() and 
        #load data in data frame using parse()
        xls = pd.read_csv("./data/diamonds.csv")
        f_xls = pd.read_csv("./data/diamonds.csv")
    except FileNotFoundError:
        print("Please make sure path given for data file is correct")
        
        
    data_structure(xls)
    
    clear()
    
    print("\n-----Alpha Numeric Columns-----")
    clean_xls = cleaning(xls)
    
    clear()
    
    print("\n-----Significant Columns-----")
    mod_xls = imputation(transform(clean_xls))
    
    clear()
    
    summary_data(mod_xls)
    
    #     12.	For each significant column
    # 	Provide the count of outliers and their value
    print("\n-----Outliers-----")
    adf = xls.drop('id', axis = 1)
    print(OutlierCount(adf))    # using ready made function from utils file with
    print(OutlierValues(adf))   # taking scale of 1.5 
    
    clear()
    
    print("\n-----VDA-----")
    
    print("\n-----Frequency Distribution for Class Variable-----")
    fre_dist(f_xls)
    
    clear()
    
    print("\n-----Histograms-----")
    histograms(mod_xls)
    
    clear()
    
    print("\n-----Box Plots-----")
    boxplots(mod_xls)
    
    clear()
    
    print("\n-----Co-relation Table and Graph-----")
    corelations(mod_xls)
    
    clear()    
    
    print("\n-----Scatter Plots-----")
    scatter_plot(mod_xls)
    
    clear()
    
    # Only for diamonds.csv
    
    #print("\n-----Adding difference column to dataframe-----")
    #print("\n-----Actual - Ideal Depth-----")
    #18. Difference between actual depth and ideal depth
    #xls['actual-ideal depth']=xls['z']-xls['depth']
    
    
    while(True):
        pl = input("Do you want to Perform EDA/VDA again?(Y/N):").lower()
        if pl == 'n':
            exe = False
            break
        elif pl == 'y':
            break
        else:
            print("Please enter Y or N....")
        