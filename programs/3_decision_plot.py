# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:46:26 2020

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
def forward_stepwise(data,vars_list,target_var,time_var,threshold_in = 0.05,verbose = True,changed = True):
    vars_list = [var.strip() for var in vars_list]
    vars_list.remove(target_var)
    vars_list.remove(time_var)
    included = []
    while changed == True:
        changed=False
        excluded = list(set(vars_list)-set(included))
        print(excluded)
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            X = data[included+[new_column]+[target_var]+[time_var]]
            X = X.dropna(thresh = X.shape[1])
            model = CoxPHFitter(penalizer = 15)
            model.fit(X,duration_col=time_var,event_col=target_var,show_progress=True,step_size = 1)
            new_pval[new_column] = model._compute_p_values()[-1]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)

            model_data = data[included+[target_var]+[time_var]]
            model_data = model_data.dropna(thresh = model_data.shape[1])
            tmp_model = CoxPHFitter(penalizer = 15)
            tmp_model.fit(X,duration_col=time_var,event_col=target_var,show_progress=True,step_size = 1)
            
            #included = list(tmp_model._compute_p_values()[tmp_model._compute_p_values()<0.05].index)
            if 'const' in included:
                included.remove('const')
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
    model = CoxPHFitter(penalizer = 2)  
    model.fit(X,duration_col=time_var,event_col=target_var,show_progress=True,step_size = 1)
    return included,model,model_data


def model_run(train,test,type_name):
    '''
    scaler = MinMaxScaler()
    scaler_vars = ['专科检查-淋巴结肿大','CT_MRI-淋巴结转移',
                 '专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)']
    train[scaler_vars] = scaler.fit_transform(train[scaler_vars])
    test[scaler_vars] = scaler.transform(test[scaler_vars])
    '''
    '''
    scaler = MinMaxScaler()
    scaler_vars = ['病理分期-N分期','病理分期-T分期']
    train[scaler_vars] = scaler.fit_transform(train[scaler_vars])
    test[scaler_vars] = scaler.transform(test[scaler_vars])
    '''
    
    
    #cox模型(逐步回归)
    vars_list = list(train.columns)
    included,model,model_data = forward_stepwise(train,vars_list,'是否死亡','生存时间(天)',threshold_in = 0.05,verbose = True,changed = True)
    train = train[included+['生存时间(天)','是否死亡']]
    test = test[included+['生存时间(天)','是否死亡']]
    print(train.columns)
    
    Cox_train_Cindex,Cox_test_Cindex,cph = Cox_Model(train,test)
    print('Cox_train_Cindex:%f,Cox_test_Cindex:%f'%(Cox_train_Cindex,Cox_test_Cindex))
    #print(included)
    summary_df = summary(cph)
    #summary_df.to_excel('回归系数表及列线图数据集//seed0'+type_name+'因子系数表(逐步回归).xlsx')
    #train.to_excel('回归系数表及列线图数据集//seed0'+type_name+'因子列线图train数据(逐步回归).xlsx',index = None)
    #test.to_excel('回归系数表及列线图数据集//seed0'+type_name+'因子列线图test数据(逐步回归).xlsx',index = None)
    return summary_df,train,test,cph
def dca(data,day,type_name):
    data['性别'][data['性别'] == 2] = 0
    data['年龄'].replace({1:0,2:1},inplace = True)
    for var in ['专科检查-原发肿瘤主体部位']:
        tmp_df = pd.get_dummies(data[var],prefix = var.split('-')[0],drop_first = True)
        data = pd.concat([data,tmp_df],axis = 1)
    data = data.drop(['专科检查-原发肿瘤主体部位'],axis = 1)
    n = data.shape[0]
    train = data[:int(n*0.7)]
    test = data[int(n*0.7):]
    summary_df,train,test,cph = model_run(train,test,type_name)
    #print(cph.predict_survival_function(test).index.to_list())
    probs = 1 - np.array(cph.predict_survival_function(test).loc[day]) 
    actual = test['是否死亡']
    fraction_of_positives,mean_predicted_value = calibration_curve(actual,probs,n_bins = 5,normalize = False)
    return fraction_of_positives,mean_predicted_value,train,test

data_n = pd.read_excel('不同seed划分的数据集//data_0.xlsx')
need_vars = ['性别','年龄','既往史-颌面部放射线接触史',
             '个人史-吸烟史','专科检查-淋巴结肿大','CT_MRI-淋巴结转移','专科检查-原发肿瘤主体部位',
             '专科检查-原发肿瘤最大径(cm)','CT_MRI-原发肿瘤最大径(cm)','病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度',
             '生存时间(天)','是否死亡']
need_vars_2 = ['性别','年龄','既往史-颌面部放射线接触史','专科检查-原发肿瘤主体部位',
             '个人史-吸烟史','病理分期-N分期','病理分期-T分期','病理类型-鳞癌：分化程度',
             '生存时间(天)','是否死亡']

data_other = data_n[need_vars]
data_tn = data_n[need_vars_2]
#predict_prob = cph.predict_partial_hazard(test)           
def positive_num(thresh,test,predict_prob,days):
    n = test.shape[0]
    predict_result = [1 if x >= thresh else 0 for x in predict_prob.loc[days]]
    #predict_result = [1 if x >= thresh else 0 for x in predict_prob[0]]
    confusion = confusion_matrix(predict_result,test['是否死亡'])
    true_positive = confusion[1][1]/n
    false_positive = confusion[1][0]/n
    benfit = true_positive-false_positive*(thresh/(1-thresh))
    return benfit
def Decision_Curve(test,predict_prob,days):
    n = test.shape[0]
    thresh_list = np.arange(0,1,0.05)
    benfit_list = np.array([positive_num(x,test,predict_prob,days) for x in thresh_list])
    benifit_none = np.array([0]*len(thresh_list))
    all_true_positive = test['是否死亡'].value_counts().loc[1]/n
    all_false_positive = test['是否死亡'].value_counts().loc[0]/n
    benifit_all = []
    for i in thresh_list:
        value = all_true_positive-all_false_positive*(i/(1-i))
        benifit_all.append(value)
    benifit_all = np.array(benifit_all)
    return thresh_list,benfit_list,benifit_none,benifit_all

#day = 372
#day = 1098
day = 1828
mean_predicted_value_other,fraction_of_positives_other,train_other,test_other = dca(data_other,day,'其他')
mean_predicted_value_tn,fraction_of_positives_tn,train_tn,test_tn = dca(data_tn,day,'TN')
Cox_train_Cindex_other,Cox_test_Cindex_other,cph_other = Cox_Model(train_other,test_other)
Cox_train_Cindex_tn,Cox_test_Cindex_tn,cph_tn = Cox_Model(train_tn,test_tn)

def dca_plot(test_other,test_tn,day,year,cph_other,cph_tn):
    predict_prob_other = 1-cph_other.predict_survival_function(test_other)  
    predict_prob_tn = 1-cph_tn.predict_survival_function(test_tn) 
    
    index_other = set(test_other.index.to_list())
    index_tn = set(test_tn.index.to_list())
    index_all = index_other&index_tn
    test_tn = test_tn.loc[index_all]
    predict_prob_tn = predict_prob_tn.T.loc[index_all].T
    test_other = test_other.loc[index_all]
    predict_prob_other = predict_prob_other.T.loc[index_all].T
    thresh_list,benfit_list_other,benifit_none,benifit_all = Decision_Curve(test_other,predict_prob_other,day)
    thresh_list,benfit_list_tn,benifit_none,benifit_all = Decision_Curve(test_tn,predict_prob_tn,day)
    
    with PdfPages('决策曲线图.pdf') as pdf:
        font = {'family' : 'Times New Roman',
                 'weight' : 'normal',
                 'size'   : 10
                 }
        plt.plot(thresh_list,benfit_list_other,label = '%d-year survival probability(Clinical prognostic model)'%(year))
        plt.plot(thresh_list,benfit_list_tn,label = '%d-year survival probability(Pathological prognostic model)'%(year))
        plt.plot(thresh_list,benifit_none,label = 'None',linestyle = ':')
        plt.plot(thresh_list,benifit_all,label = 'All',linestyle = ':',)
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.xlabel('Threshold probability',font)
        plt.ylabel('Net benefit',font)
        plt.title('Decision Curve',font)
        plt.yticks(np.arange(-0.1,0.7,0.1))
        plt.ylim(-0.1,0.7)
        plt.yticks(fontproperties = 'Times New Roman', size = 10)
        plt.xticks(fontproperties = 'Times New Roman', size = 10)
        plt.legend(fontsize=7,prop = {"family" : "Times New Roman"},frameon = False)
        #plt.title(tested_var+' survival curve with chi_value '+str(round(results.test_statistic,3))+' and p_value '+str(round(results.p_value,3)))
        pdf.savefig()
        plt.close()
dca_plot(test_other,test_tn,day,5,cph_other,cph_tn)
