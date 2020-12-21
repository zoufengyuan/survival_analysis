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
def summary(cph):
    """Summary statistics describing the fit.
    Set alpha property in the object before calling.

    Returns
    -------
    df : DataFrame
        Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
    ci = 1 - cph.alpha
    with np.errstate(invalid="ignore", divide="ignore"):
        df = pd.DataFrame(index=cph.hazards_.index)
        df["coef"] = cph.hazards_
        df["exp(coef)"] = np.exp(cph.hazards_)
        df["se(coef)"] = cph.standard_errors_
        df["z"] = cph._compute_z_values()
        df["p"] = cph._compute_p_values()
        df["-log2(p)"] = -np.log2(df["p"])
        df["lower %g" % ci] = cph.confidence_intervals_["lower-bound"]
        df["upper %g" % ci] = cph.confidence_intervals_["upper-bound"]
    return df

data_n = pd.read_excel('不同seed划分的数据集//data_0.xlsx')
n = len(data_n)

source_need_vars = ['性别','年龄','既往史-颌面部放射线接触史',
             '个人史-吸烟史','专科检查-淋巴结肿大','CT_MRI-淋巴结转移','专科检查-原发肿瘤主体部位',
             '专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','病理分期-N分期','病理分期-T分期',
             '病理类型-鳞癌：分化程度']
source_need_vars_2 = ['性别','年龄','既往史-颌面部放射线接触史','专科检查-原发肿瘤主体部位',
             '个人史-吸烟史','病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度']

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

scaler = MinMaxScaler()
scaler_vars_1 = ['专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','病理分期-N分期','病理类型-鳞癌：分化程度']
train_1_all[scaler_vars_1] = scaler.fit_transform(train_1_all[scaler_vars_1])
test_one[scaler_vars_1] = scaler.transform(test_one[scaler_vars_1])


scaler_vars_2 = ['病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度']
train_2_all[scaler_vars_2] = scaler.fit_transform(train_2_all[scaler_vars_2])
test_two[scaler_vars_2] = scaler.transform(test_two[scaler_vars_2])

Cox_train_Cindex_1,Cox_test_Cindex_1,cph_1 = Cox_Model(train_1_all,test_one)
Cox_train_Cindex_2,Cox_test_Cindex_2,cph_2 = Cox_Model(train_2_all,test_two)

df_other = summary(cph_1)
df_tn = summary(cph_2)

#df_other.to_excel('回归系数表及列线图数据集//seed0其他因子系数表(逐步回归).xlsx')
#df_tn.to_excel('回归系数表及列线图数据集//seed0TN因子系数表(逐步回归).xlsx')
def split_data(data):
    n = len(data)
    train = data[:int(n*0.7)]
    test = data[int(0.7*n):]
    return train,test
    
    
def single_summary(vars_list,data_n):
    single_summary_df = pd.DataFrame()
    for var in vars_list:
        data = data_n[[var]+['生存时间(天)','是否死亡']]
        if var == '专科检查-原发肿瘤主体部位':
            tmp_df = pd.get_dummies(data[var],prefix = var.split('-')[0],drop_first = True)
            data = pd.concat([data,tmp_df],axis = 1)
            del data[var]
        train,test = split_data(data)
        _,_,cph = Cox_Model(train,test)
        summary_df = summary(cph)
        single_summary_df = single_summary_df.append(summary_df)
    return single_summary_df
single_summary_df_other = single_summary(source_need_vars,data_n)
single_summary_df_tn = single_summary(source_need_vars_2,data_n)        
#single_summary_df_other.to_excel('回归系数表及列线图数据集//seed0其他因子系数表(无逐步回归).xlsx')
#single_summary_df_tn.to_excel('回归系数表及列线图数据集//seed0TN因子系数表(无逐步回归).xlsx')
            





























