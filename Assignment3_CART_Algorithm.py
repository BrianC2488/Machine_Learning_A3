#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
from random import seed
from random import randrange
from sklearn.model_selection import train_test_split


# This code below imports my data set from a CSV file and spilts it up into training and testing data and saves it seperately in arrays.

# In[2]:


def create_Test_and_Train(csv_file):
    
    #This function takes in the Owls.CSV file and splits 
    #it up into Testing and Training Datasets
    #Our dataset is read into using the Pandas function in Python. 
    owls_data = pd.read_csv(csv_file)

    train, test = train_test_split(owls_data, test_size=0.3)


    #each row in the train and test datasets 
    #is then added to an array and returned
    Test_Array = []

    for row in test.values:
        Test_Array.append((list(row)))

    Train_Array = []

    for row in train.values:
        Train_Array.append((list(row)))

    return Test_Array, Train_Array


# In[3]:


#The label count method returns 
#how many types of each owl there
def label_count(rows):
    #This holds our 
    count = {}  
    for row in rows:
        #this points to the last column in  
        #the dataset which is our label 
        data_label = row[-1] 
        
        #This iterates through each row individually
        #if the label in the row is not already in counts 
        #then it will be initialised with a value of 0 and then incremented in counts
        if data_label not in count:
            count[data_label] = 0
        count[data_label] = count[data_label]+1
    return count


# In[4]:


class Question:
    
    #stores an instance of a question which is a column and 
    #its corrsponding value. For example column = wingspan, wingspan = 3.5
    def __init__(self, column, value):
        self.column = column
        self.value = value
        
    #then this object is compared against a row of data
    def comparison(self, eg):
        
        #the sample row is compared against the object to see 
        #if both match and if they do match then it returns true. 
        
        sample_row = eg[self.column]
        
        if sample_row >= self.value:
            return True 


# In[5]:


#This function takes in the training data array and splits 
#it up depending on the Question it is being compared to 
def partition(rows, question):
  
    #at a node, rows are split up into 2 groups 
    #True Rows and False Rows. 
    
    true_rows, false_rows = [], []
    
    #the array is iterated through and if a row matches 
    #a question it is added to the array of true_rows
    for i in rows:
        if question.comparison(i):
            true_rows.append(i)
        else:
    #otherwise it is added to the array of false rows
            false_rows.append(i)
    return true_rows, false_rows


# In[6]:


def gini_index(rows):
    #This funciton calculates the impurity of a given set of data 
    #it uses the label_count function to calculate how many of each type of label there is 
    ammount_labels = label_count(rows)
    
    impurity = 1
    for i in ammount_labels:
        #In our case if we had (3 Snowy Owls, 2 LongEaredOwl, 5 BarnOwl)
        prob_of_label = ammount_labels[i] / float(len(rows))
        impurity = impurity-prob_of_label**2
        #Impurity = 1 - (3/10)^2 + (2/10)^2 + (5/10)^2 = 0.62
    return impurity


# In[7]:


def info_gain(Child_left, Child_right, Parent):
    
    #Information Gain is calculated by taking the Gini Impurity at a given node 
    #and subtracting the Gini Impurity at both child nodes after a question has 
    #seperated the group into 2 parts 
    
    Total = (len(Child_left) + len(Child_right))
    x = float(len(Child_left)) / Total
    
    y = float(len(Child_right)) / Total
    
    return Parent - x * gini_index(Child_left) -  y * gini_index(Child_right)


# In[8]:


def find_best_split(rows):
   
    #this method iterates through our Training Data and finds 
    #the question which yields the highest Information Gain
    
    highest_gain = 0  
    best_question = None 
    parent = gini_index(rows)
    no_of_features = len(rows[0]) - 1  #ammount of features in a row - (label)

    for column in range(no_of_features):  

        
        row_n = set([row[column] for row in rows]) 

        
        for value in row_n: 
            
            #Each column of the dataset is iterated through and each value 
            #in each column is also itterated through to find the question that 
            #will best split the data up to give the best info_gain
            question = Question(column, value)
            true_rows, false_rows = partition(rows, question)

            
     
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

                
            gain = info_gain(true_rows, false_rows, parent)
            
            #The gain for each question is recorded and if a new higher gain is  
            #calculated it will be set as the highest gain and the question recorded
            
            if gain >= highest_gain:
                highest_gain, best_question = gain, question

    return highest_gain, best_question


# In[9]:


class Decision_Made:
    #This records the number of times a type of owl appears at a node 
    def __init__(self, rows):
        self.predictions = label_count(rows)
        


# In[10]:


class Decision_Node:
    #This class holds a reference to the question and 
    #true and false branches nodes after split
    def __init__(self,question,true_node,false_node):
        self.question = question
        self.true_node = true_node
        self.false_node = false_node


# In[11]:


#the build tree function takes in the Training Array
def build_tree(rows):
    #from this it calculates the question which yields the best Gain
    gain, question = find_best_split(rows)
    
    #if the info gain = 0 then there are no more questions to be asked
    if gain == 0:
        return Decision_Made(rows)

    #Using this question it will split the dataset into true and false rows
    true_rows, false_rows = partition(rows, question)
    
    #Then it uses the True and False Rows to continue on 
    #building more branches until a decision has been made 
    
    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    #Then the Decision_Node object is returned 
    return Decision_Node(question, true_branch, false_branch)


# In[12]:


def classification(row, node):
    

    if isinstance(node, Decision_Made):
        return node.predictions
    #if the test row matches a question then it is passed along the true
    #branch and classfied again until a deicision is reached.
    if node.question.comparison(row):
        return classification(row, node.true_node)
        
    else:
        #otherwise it is passed along the false 
        return classification(row, node.false_node)


# In[13]:


def TF_Rate(array, model):
    
    #True_Array holds an array full of predictions that were true
    #False_Array holds an array full of predictions that were false
    #Outputs the percentage of True and False Predictions
    True_Array = []
    False_Array = []
    
    for row in array:
        if row[-1] in (classification(row, model)):
            True_Array.append("True")
        else:
            False_Array.append("False")
    
    total = len(True_Array)+len(False_Array)
    T = len(True_Array)
    F = len(False_Array)
    return ((T/total), (F/total))
    


# In[15]:


def Automated_Testing():
    #This function automates the training, testing and averaging of all results
    #Each time this method is called it creates a new model using random training data
    #and then classifys a random testing data set.
    #Each individual test classification results printed and there is also a average of 
    #all of the testing data printed at the end. 
    Total_T = []
    Total_F = []
    
    for i in range(10):
       
        Test_Array, Train_Array = create_Test_and_Train("owls.csv")
        tree = build_tree(Train_Array)
        T, F = TF_Rate(Test_Array, tree)
        print("Test #%s: True:%.2f , False:%.2f" %(str(i+1),(T*100),(F*100)))
        Total_T.append(T) 
        Total_F.append(F)
        
    print("\nAverage True Classifications: %.2f percent \nAverage False Classifications: %.2f percent" % ((sum(Total_T)/len(Total_T)*100),(sum(Total_F)/len(Total_F)*100)))


# In[24]:


Automated_Testing()


# In[ ]:




