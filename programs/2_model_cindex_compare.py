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
train_1_all = data_1[:int(0.7*n)]
train_2_all = data_2[:int(0.7*n)]

test_one = data_1[int(0.7*n):]
test_two = data_2[int(0.7*n):]

train_index_all = train_1_all.index.to_list()
test_index_all = test_one.index.to_list()


train_cindex_1 = []
test_cindex_1 = []
train_cindex_2 = []
test_cindex_2 = []
epochs = 1000
j = 1


while j <= epochs:
    train_index = np.random.choice(train_index_all,len(train_index_all))
    test_index = np.random.choice(test_index_all,len(test_index_all))
    train_1 = train_1_all.loc[train_index]
    test_1 = test_one.loc[test_index]
    train_2 = train_2_all.loc[train_index]
    test_2 = test_two.loc[test_index]
    
    scaler = MinMaxScaler()
    scaler_vars_1 = ['专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','病理分期-N分期','病理类型-鳞癌：分化程度']
    train_1[scaler_vars_1] = scaler.fit_transform(train_1[scaler_vars_1])
    test_1[scaler_vars_1] = scaler.transform(test_1[scaler_vars_1])
    
    train_2 = data_2.loc[train_index]
    test_2 = data_2.loc[test_index]
    scaler_vars_2 = ['病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度']
    train_2[scaler_vars_2] = scaler.fit_transform(train_2[scaler_vars_2])
    test_2[scaler_vars_2] = scaler.transform(test_2[scaler_vars_2])
    
    Cox_train_Cindex_1,Cox_test_Cindex_1,cph_1 = Cox_Model(train_1,test_1)
    Cox_train_Cindex_2,Cox_test_Cindex_2,cph_2 = Cox_Model(train_2,test_2)
    train_cindex_1.append(Cox_train_Cindex_1)
    test_cindex_1.append(Cox_test_Cindex_1)
    train_cindex_2.append(Cox_train_Cindex_2)
    test_cindex_2.append(Cox_test_Cindex_2)
    j+=1

cindex_df = pd.DataFrame({'其他train':train_cindex_1,'其他test':test_cindex_1,'TNtrain':train_cindex_2,'TNtest':test_cindex_2})
#cindex_df.to_excel('cindex置信区间结果//cindex汇总表2.xlsx',index = None)


