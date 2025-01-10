#K-Nearest Neighbors (KNN) Classification 

import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

import pandas as pd
import numpy as np

from statmethods import KNNModel

bank_data = pd.read_csv("BankChurners.csv")

bank_data = bank_data.drop({"Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"}, axis=1)

# print(bank_data.head())

knn = KNNModel(bank_data=bank_data, col='Attrition_Flag', test_size=0.3)

existing_customer_data, knn_model = knn.model_training(n_neighbors=5)
existing_customer_data_3, knn_model_3 = knn.model_training(n_neighbors=3)
existing_customer_data_10, knn_model_10 = knn.model_training(n_neighbors=10)

existing_customer_data_in_sample, knn_model_in_sample = knn.model_training(n_neighbors=5, in_sample=True)
existing_customer_data_3_in_sample, knn_model_3_in_sample = knn.model_training(n_neighbors=3, in_sample=True)
existing_customer_data_10_in_sample, knn_model_10_in_sample = knn.model_training(n_neighbors=10, in_sample=True)

# print(existing_customer_data.head(10))

knn.model_evaluation(existing_customer_data['Actual_Customer_Decision'],existing_customer_data['Predicted_Customer_Decision'])
knn.model_evaluation(existing_customer_data_3['Actual_Customer_Decision'],existing_customer_data_3['Predicted_Customer_Decision'])
knn.model_evaluation(existing_customer_data_10['Actual_Customer_Decision'],existing_customer_data_10['Predicted_Customer_Decision'])

knn.model_evaluation(existing_customer_data_in_sample['Actual_Customer_Decision'],existing_customer_data_in_sample['Predicted_Customer_Decision'])
knn.model_evaluation(existing_customer_data_3_in_sample['Actual_Customer_Decision'],existing_customer_data_3_in_sample['Predicted_Customer_Decision'])
knn.model_evaluation(existing_customer_data_10_in_sample['Actual_Customer_Decision'],existing_customer_data_10_in_sample['Predicted_Customer_Decision'])

# bootstrapping

scores = knn.bootstrapping("knn", n_neighbors=5) 
scores_3 = knn.bootstrapping("knn", n_neighbors=3) 
scores_10 = knn.bootstrapping("knn", n_neighbors=10) 

print("Bootstrap Accuracy Scores - KNN")

scores_df = pd.DataFrame()

scores_df["Sample Number"] = list(range(1,len(scores)+1))

scores_df["Accuracy Scores(k=5)"] = scores
scores_df["Accuracy Scores(k=3)"] = scores_3
scores_df["Accuracy Scores(k=10)"] = scores_10

print(scores_df)

print("The average accuracy (k=5) is ", np.mean(scores))
print("The average accuracy (k=3) is ", np.mean(scores_3))
print("The average accuracy (k=10) is ", np.mean(scores_10))

# cross-validation

scores_cv = knn.cross_validation("knn", n_neighbors=5)
scores_cv_3 = knn.cross_validation("knn", n_neighbors=3)
scores_cv_10 = knn.cross_validation("knn", n_neighbors=10)

print("Accuracy(CV) - k=5: ", scores_cv.mean())
print("Accuracy(CV) - k=3: ", scores_cv_3.mean())
print("Accuracy(CV) - k=10: ", scores_cv_10.mean())

# model visualization

knn.model_visualization(existing_customer_data['Actual_Customer_Decision'],existing_customer_data['Predicted_Customer_Decision'], knn_model, title="Confusion Matrix(KNN,K=5) - Out-of-sample")
knn.model_visualization(existing_customer_data_3['Actual_Customer_Decision'],existing_customer_data_3['Predicted_Customer_Decision'], knn_model, title="Confusion Matrix(KNN,K=3) - Out-of-sample")
knn.model_visualization(existing_customer_data_10['Actual_Customer_Decision'],existing_customer_data_10['Predicted_Customer_Decision'], knn_model, title="Confusion Matrix(KNN,K=10) - Out-of-sample")

knn.model_visualization(existing_customer_data_in_sample['Actual_Customer_Decision'],existing_customer_data_in_sample['Predicted_Customer_Decision'], knn_model, title="Confusion Matrix(KNN,K=5) - In-sample")
knn.model_visualization(existing_customer_data_3_in_sample['Actual_Customer_Decision'],existing_customer_data_3_in_sample['Predicted_Customer_Decision'], knn_model, title="Confusion Matrix(KNN,K=3) - In-sample")
knn.model_visualization(existing_customer_data_10_in_sample['Actual_Customer_Decision'],existing_customer_data_10_in_sample['Predicted_Customer_Decision'], knn_model, title="Confusion Matrix(KNN,K=10) - In-sample")




