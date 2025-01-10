import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

class Statmethod(ABC):
    def __init__(self, bank_data, col, test_size): 
        self.bank_data = bank_data
        self.col = col
        self.test_size = test_size
    
    @abstractmethod

    def model_training(self):
        pass

    def model_evaluation(self, y_true, y_pred):

        accuracy = accuracy_score(y_true, y_pred)

        print('Model accuracy: ', accuracy)
        
        conf_matrix = confusion_matrix(y_true, y_pred)

        print('Confusion Matrix: \n', conf_matrix)

        class_rep = classification_report(y_true, y_pred)

        print('Classification Report: \n', class_rep)
    
    def data_preprocessing(self, in_sample = False):
        y_pred_set = self.bank_data[self.col]
        x_set = self.bank_data.drop([self.col, 'CLIENTNUM'], axis=1) # no need in client IDs

        y_pred_set = y_pred_set.apply(lambda x: 1 if x == 'Existing Customer' else 0)

        x_set = pd.get_dummies(x_set, drop_first=True) # One-hot encode categorical columns

        size = self.test_size

        if (in_sample): size = None

        x_train, x_test, y_train, y_test = train_test_split(x_set, y_pred_set, test_size=size, random_state=42)

        # print(x_test.columns)

        return x_train, x_test, y_train, y_test
    
    def bootstrapping(self, model_name, n_neighbors=5, n_iterations = 10):
        data = self.bank_data.drop('CLIENTNUM', axis=1)
        data = pd.get_dummies(data, drop_first=True)

        n_size = int(len(data)*0.7)
        stats = list()
        values = data.values # ValueError: could not convert string to float: 'Existing Customer'

        model_dict = {
            "name": ["logistic", "lda", "qda", "knn"],
            "model": [LogisticRegression(max_iter=1000, random_state=42), LinearDiscriminantAnalysis(n_components=1), QuadraticDiscriminantAnalysis(), KNeighborsClassifier(n_neighbors=n_neighbors)]
        }
        model_index = model_dict['name'].index(model_name)
        model = model_dict["model"][model_index]

        for i in range(n_iterations):
            train = resample(values, n_samples=n_size)
            test = np.array([x for x in values if x.tolist() not in train.tolist()])

            model.fit(train[:,:-1],train[:,-1]) # our y_train is the last value of our bootstrapped training set

            predictions = model.predict(test[:,:-1])

            score = accuracy_score(test[:,-1], predictions)

            stats.append(score)

        return stats

    def cross_validation(self, model_name, n_splits=10, n_neighbors=5, random_state=42):
        X = self.bank_data.drop([self.col, 'CLIENTNUM'], axis=1)
        y = self.bank_data[self.col]
       
        X = pd.get_dummies(X, drop_first=True)
        y = y.apply(lambda x: 1 if x == 'Existing Customer' else 0)

        model_dict = {
            "name": ["logistic", "lda", "qda", "knn"],
            "model": [LogisticRegression(max_iter=1000, random_state=42), LinearDiscriminantAnalysis(n_components=1), QuadraticDiscriminantAnalysis(), KNeighborsClassifier(n_neighbors=n_neighbors)]
        }
        model_index = model_dict['name'].index(model_name)
        model = model_dict["model"][model_index]

        kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        results = cross_val_score(model,X,y,cv = kfold)

        return results

    def create_result_data(self, y_test, y_pred):
        result_data = pd.DataFrame()

        result_data['Actual_Customer_Decision'] = y_test

        result_data['Predicted_Customer_Decision'] = y_pred

        return result_data
    
    def model_visualization(self, y_true, y_pred, model,title="Confusion Matrix"): 
        conf_matrix = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)

        disp.plot(cmap='Blues')
        plt.title(title)
        plt.show()

    def decision_boundary(self, x_test, y_true, model): # find a way how to fix this function

        #sns.scatterplot(x=x_test['Total_Trans_Amt'], y=x_test['Months_on_book'], hue=self.col, data=self.bank_data)
        # now create a split

        # x_test = [['Total_Trans_Amt','Months_on_book']]

        # print(x_test.columns)

        min1, max1 = x_test['Total_Trans_Amt'].min() - 1, x_test['Total_Trans_Amt'].max() + 1
        min2, max2 = x_test['Months_on_book'].min() - 1, x_test['Months_on_book'].max() + 1

        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))    

        grid = np.hstack((r1,r2))

        yhat = model.predict(grid) # issue is here: ValueError: X has 2 features, but LogisticRegression is expecting 32 features as input.

        zz = yhat.reshape(xx.shape)

        plt.contourf(xx, yy, zz, cmap='Paired')

        for class_value in ['Existing Customer','Attrited Customer']:
            row_ix = np.where(y_true == class_value)
            plt.scatter(x_test[row_ix, 0], x_test[row_ix, 1], cmap='Paired')

        plt.show()

class LogRegModel(Statmethod):

    def model_training(self,random_state=42,in_sample = False, display_coefs = False):

        x_train, x_test, y_train, y_test = self.data_preprocessing(in_sample=in_sample)

        # x_train = sm.add_constant(x_train)

        logr = LogisticRegression(max_iter=1000, random_state=random_state).fit(x_train, y_train)

        # logr = sm.Logit(y_train, x_train).fit()

        y_pred = logr.predict(x_test)

        if(display_coefs): self.display_coefs(x_train, logr)

        result_data = self.create_result_data(y_test, y_pred)

        return result_data, logr
    
    def display_coefs(self, x_train, logr):

        coefs = logr.coef_[0]
        intercept = logr.intercept_[0]

        std_errors = self.std_errors(x_train, logr)

        z_values, p_values = self.z_p_values(coefs, std_errors)

        coef_df = pd.DataFrame({
                "Feature": ["Intercept"] + list(x_train.columns), 
                "Coefficient": [intercept] + list(coefs),
                "Std.Error": std_errors,
                "Z-value": [intercept / std_errors[0]] + list(z_values),
                "P-value": [2 * (1 - norm.cdf(np.abs(intercept / std_errors[0])))] + list(p_values)
            })
        

        print(coef_df)
    
    def std_errors(self, x_train, model):
        probs = model.predict_proba(x_train)

        W = np.diagflat(probs[:, 1] * (1 - probs[:, 1]))

        X_design = np.hstack([np.ones((x_train.shape[0], 1)), x_train]) 
        Hessian = X_design.T @ W @ X_design # Hessian covariance matrix approximation

        cov_matrix = np.linalg.inv(Hessian)

        standard_errors = np.sqrt(np.diag(cov_matrix))

        return standard_errors
    
    def z_p_values(self, coefficients, std_errors):

        z_values = coefficients/std_errors[1:]

        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))  # Two-tailed test

        return z_values, p_values

class LDAModel(Statmethod):

    def model_training(self, n_components=1, in_sample=False):

        x_train, x_test, y_train, y_test = self.data_preprocessing(in_sample=in_sample)

        lda = LinearDiscriminantAnalysis(n_components=n_components)

        y_pred = lda.fit(x_train, y_train).predict(x_test)

        result_data = self.create_result_data(y_test, y_pred)

        return result_data, lda

class QDAModel(Statmethod):

    def model_training(self, in_sample=False):

        x_train, x_test, y_train, y_test = self.data_preprocessing(in_sample=in_sample)

        qda = QuadraticDiscriminantAnalysis()

        y_pred = qda.fit(x_train, y_train).predict(x_test)

        result_data = self.create_result_data(y_test, y_pred)

        return result_data, qda
    
class KNNModel(Statmethod):

    def model_training(self, n_neighbors, in_sample=False):
        
        x_train, x_test, y_train, y_test = self.data_preprocessing(in_sample=in_sample)

        knn = KNeighborsClassifier(n_neighbors).fit(x_train,y_train)

        y_pred = knn.predict(x_test)

        result_data = self.create_result_data(y_test, y_pred)

        return result_data, knn