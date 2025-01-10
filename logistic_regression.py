# logistic regression

import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

import numpy as np
import pandas as pd

from statmethods import LogRegModel

bank_data = pd.read_csv("BankChurners.csv")

# cleaning

bank_data = bank_data.drop({"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"}, axis=1)

# print(bank_data.head())

logreg = LogRegModel(bank_data=bank_data, col='Attrition_Flag', test_size=0.3) 

existing_customer_data, logreg_output = logreg.model_training(display_coefs=True)

existing_customer_data_in_sample, logreg_output_in_sample = logreg.model_training(in_sample=True)

# evaluate the model and return its estimated parameters

logreg.model_evaluation(existing_customer_data['Actual_Customer_Decision'],existing_customer_data['Predicted_Customer_Decision'])

logreg.model_evaluation(existing_customer_data_in_sample['Actual_Customer_Decision'],existing_customer_data_in_sample['Predicted_Customer_Decision'])

# bootstrapping

# scores = logreg.bootstrapping("logistic") 

# print("Bootstrap Accuracy Scores - Logistic Regression")

# scores_df = pd.DataFrame()

# scores_df["Sample Number"] = list(range(1,len(scores)+1))

# scores_df["Accuracy Scores"] = scores

# print(scores_df)

# print("The average accuracy is ", np.mean(scores))

# # cross-validation

# scores_cv = logreg.cross_validation("logistic")

# print("Accuracy(CV): ", scores_cv.mean())

# visualization plots

logreg.model_visualization(existing_customer_data['Actual_Customer_Decision'],
                           existing_customer_data['Predicted_Customer_Decision'], logreg_output, title="Confusion Matrix(Log) - Out-of-sample")


logreg.model_visualization(existing_customer_data_in_sample['Actual_Customer_Decision'],
                           existing_customer_data_in_sample['Predicted_Customer_Decision'], logreg_output_in_sample,title="Confusion Matrix(Log) - In-sample")

# decision boundary (fix that):

# logreg.decision_boundary(x_test, existing_customer_data['Actual_Customer_Decision'], logreg_output)



