# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:59:33 2020

@author: 86156
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter,KaplanMeierFitter 
from lifelines.statistics import logrank_test,multivariate_logrank_test
from matplotlib import font_manager
from lifelines.utils import concordance_index
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import calibration_curve 
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.metrics import confusion_matrix
import statsmodels.stats.proportion as sm
from scipy.stats import ks_2samp
from sklearn.utils import shuffle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind
from scipy.stats import levene
import matplotlib.pylab as pylab


def Cox_Model(train,test):
    '''
    train: train_data
    test: test_data
    vars_list: variables list
    '''
    cph = CoxPHFitter(penalizer = 15)
    cph.fit(train,duration_col='生存时间(天)',event_col='是否死亡',show_progress=True,step_size = 1)
    Cox_train_Cindex = concordance_index(train['生存时间(天)'],-cph.predict_partial_hazard(train),train['是否死亡'])
    Cox_test_Cindex = concordance_index(test['生存时间(天)'],-cph.predict_partial_hazard(test),test['是否死亡'])
    return Cox_train_Cindex,Cox_test_Cindex,cph

data_n = pd.read_excel('不同seed划分的数据集//data_0.xlsx')
n = len(data_n)
need_vars = ['专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','病理分期-N分期','病理类型-鳞癌：分化程度',
             '生存时间(天)','是否死亡']
need_vars_2 = ['病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度',
             '生存时间(天)','是否死亡']
data_1 = data_n[need_vars]
data_2 = data_n[need_vars_2]
n = len(data_n)
train_1 = data_1[:int(0.7*n)]
train_2 = data_2[:int(0.7*n)]

test_1 = data_1[int(0.7*n):]
test_2 = data_2[int(0.7*n):]


scaler = MinMaxScaler()
scaler_vars_1 = ['专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','病理分期-N分期','病理类型-鳞癌：分化程度']
train_1[scaler_vars_1] = scaler.fit_transform(train_1[scaler_vars_1])
test_1[scaler_vars_1] = scaler.transform(test_1[scaler_vars_1])


scaler_vars_2 = ['病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度']
train_2[scaler_vars_2] = scaler.fit_transform(train_2[scaler_vars_2])
test_2[scaler_vars_2] = scaler.transform(test_2[scaler_vars_2])

Cox_train_Cindex_1,Cox_test_Cindex_1,cph_1 = Cox_Model(train_1,test_1)
Cox_train_Cindex_2,Cox_test_Cindex_2,cph_2 = Cox_Model(train_2,test_2)



year_1 = 357
year_3 = 1098
year_5 = 1828

def generate_label(data,time):
    label_list = []
    for i in range(len(data)):
        if data['是否死亡'].iloc[i] == 0:
            label = 0
        else:
            if data['生存时间(天)'].iloc[i]>time:
                label = 0
            else:
                label = 1
        label_list.append(label)
    return label_list


def main(cph_1,test_1,type_name):
    predict_1_df = 1-cph_1.predict_survival_function(test_1).T

    predict_1_year = list(predict_1_df[year_1])
    label_1_year = generate_label(test_1,year_1)
    predict_3_year = list(predict_1_df[year_3])
    label_3_year = generate_label(test_1,year_3)
    predict_5_year = list(predict_1_df[year_5])
    label_5_year = generate_label(test_1,year_5)
    
    
    fpr_1, tpr_1, thresholds_1 = roc_curve(label_1_year, predict_1_year, pos_label=1)
    fpr_3, tpr_3, thresholds_3 = roc_curve(label_3_year, predict_3_year, pos_label=1)
    fpr_5, tpr_5, thresholds_5 = roc_curve(label_5_year, predict_5_year, pos_label=1)
    
    model_auc_1 = auc(fpr_1, tpr_1)
    model_auc_3 = auc(fpr_3, tpr_3)
    model_auc_5 = auc(fpr_5, tpr_5)
    with PdfPages('结果更新V1.2//ROC曲线//'+type_name+'.pdf') as pdf:
        params = {
        'font.sans-serif': 'Times New Roman', #显示中文
        'axes.unicode_minus': False} #显示负号
        pylab.rcParams.update(params)
        
        csfont = {'fontname':'Times New Roman'}
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.figure()
        plt.xlabel('1-Specificity',**csfont)
        plt.ylabel('Sensitivity',**csfont)
        plt.plot([0, 1], [0, 1], color='silver', linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Classifier ROC',**csfont)
        plt.plot(fpr_1, tpr_1, color='blue', lw=1, label='AUC at 1 year: %0.3f' % model_auc_1)
        plt.plot(fpr_3, tpr_3, color='red', lw=1, label='AUC at 3 years: %0.3f' % model_auc_3)
        plt.plot(fpr_5, tpr_5, color='green', lw=1, label='AUC at 5 years: %0.3f' % model_auc_5)
        plt.legend(loc="lower right")
        pdf.savefig()
        plt.close()
    
main(cph_1,train_1,'其他因子ROC_训练集')
main(cph_2,train_2,'TN因子ROC_训练集')
main(cph_1,test_1,'其他因子ROC_测试集')
main(cph_2,test_2,'TN因子ROC_测试集')










































