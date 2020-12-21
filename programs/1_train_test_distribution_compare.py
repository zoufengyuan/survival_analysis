# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:52:50 2020

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
from scipy.stats import ttest_ind
from scipy.stats import levene
from sklearn.utils import shuffle
from matplotlib.backends.backend_pdf import PdfPages

data_n = pd.read_excel('不同seed划分的数据集//data_0.xlsx')
n = data_n.shape[0]
#data_n = data_n[need_vars]
train = data_n[:int(n*0.7)]
test = data_n[int(n*0.7):]

bins = [-0.00001,2,4,10]
data_n['专科检查-原发肿瘤最大径(cm)_cut'] = pd.cut(data_n['专科检查-原发肿瘤最大径(cm)'],bins)
data_n['CT_MRI-原发肿瘤最大径(cm)_cut'] = pd.cut(data_n['CT_MRI-原发肿瘤最大径(cm)'],bins)
compared_vars = ['性别','年龄','既往史-颌面部放射线接触史',
             '个人史-吸烟史','专科检查-淋巴结肿大','CT_MRI-淋巴结转移',
             '病理类型-鳞癌：分化程度','专科检查-原发肿瘤主体部位',
             '是否死亡','病理分期-N分期','病理分期-T分期','专科检查-原发肿瘤最大径(cm)_cut','CT_MRI-原发肿瘤最大径(cm)_cut']
lianxu_vars = ['专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','生存时间(天)']
train = data_n[:int(n*0.7)]
test = data_n[int(n*0.7):]

p_value_dict = {}
all_df = pd.DataFrame()
for var in compared_vars:
    train_fenbu = train[var].value_counts().reset_index()
    train_fenbu.rename(columns = {var:'train'},inplace = True)
    train_fenbu_zhanbi = train[var].value_counts(normalize = True).reset_index()
    train_fenbu_zhanbi.rename(columns = {var:'train_rate'},inplace = True)
    test_fenbu = test[var].value_counts().reset_index()
    test_fenbu.rename(columns = {var:'test'},inplace = True)
    test_fenbu_zhanbi = test[var].value_counts(normalize = True).reset_index()
    test_fenbu_zhanbi.rename(columns = {var:'test_rate'},inplace = True)
    for df in [train_fenbu_zhanbi,test_fenbu,test_fenbu_zhanbi]:
        train_fenbu = pd.merge(train_fenbu,df,on = 'index',how = 'outer')
    train_fenbu.rename(columns = {'index':'Subtype'},inplace = True)
    train_fenbu['Characteristics'] = [var]*len(train_fenbu)
    #train_fenbu.to_excel('训练测试集变量分布//'+var+'.xlsx',index = None)
    all_df = all_df.append(train_fenbu)
    ks_test,p_value = ks_2samp(train[var], test[var])
    p_value_dict[var] = p_value

all_df['train_rate'] = all_df['train_rate'].apply(lambda x:'('+str(round(x*100,2))+'%'+')')
all_df['test_rate'] = all_df['test_rate'].apply(lambda x:'('+str(round(x*100,2))+'%'+')')
all_df['train'] = all_df['train'].apply(lambda x:str(x))
all_df['test'] = all_df['test'].apply(lambda x:str(x))
all_df['train'] = all_df['train']+all_df['train_rate']
all_df['test'] = all_df['test']+all_df['test_rate']

all_df.to_excel('训练测试集变量分布//训练测试集分布汇总.xlsx',index = None)
result_dict = {}
for var in lianxu_vars:
    train_mean = train[var].mean()
    test_mean = test[var].mean()
    _,levene_p = levene(train[var],test[var])
    if levene_p< 0.05:
        _,p_value = ttest_ind(train[var],test[var],equal_var = False)
    else:
        _,p_value = ttest_ind(train[var],test[var])
    result_dict[var] = [train_mean,test_mean,p_value]
    
    
    
    