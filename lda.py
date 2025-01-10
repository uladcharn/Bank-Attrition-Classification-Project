# Linear Discriminant Analysis (LDA)

import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

import pandas as pd
import numpy as np

from statmethods import LDAModel

bank_data = pd.read_csv("BankChurners.csv")

bank_data = bank_data.drop({"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"}, axis=1)

# print(bank_data.head())

lda = LDAModel(bank_data=bank_data, col='Attrition_Flag', test_size=0.3)

existing_customer_data, lda_model = lda.model_training(n_components=1)

existing_customer_data_in_sample, lda_model_in_sample = lda.model_training(n_components=1,in_sample=True)

# print(existing_customer_data.head(10))

lda.model_evaluation(existing_customer_data['Actual_Customer_Decision'],existing_customer_data['Predicted_Customer_Decision'])
lda.model_evaluation(existing_customer_data_in_sample['Actual_Customer_Decision'],existing_customer_data_in_sample['Predicted_Customer_Decision'])

# bootstrapping

scores = lda.bootstrapping("lda") 

print("Bootstrap Accuracy Scores - Linear Discriminant Analysis")

scores_df = pd.DataFrame()

scores_df["Sample Number"] = list(range(1,len(scores)+1))

scores_df["Accuracy Scores"] = scores

print(scores_df)

print("The average accuracy is ", np.mean(scores))

# cross-validation

scores_cv = lda.cross_validation("lda")

print("Accuracy(CV): ", scores_cv.mean())

# model visualization

lda.model_visualization(existing_customer_data['Actual_Customer_Decision'],
                        existing_customer_data['Predicted_Customer_Decision'], lda_model, title="Confusion Matrix(LDA) - Out-of-sample")
lda.model_visualization(existing_customer_data_in_sample['Actual_Customer_Decision'],
                        existing_customer_data_in_sample['Predicted_Customer_Decision'], lda_model_in_sample, title="Confusion Matrix(LDA) - In-sample")

