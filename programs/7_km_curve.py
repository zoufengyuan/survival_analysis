# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:49:32 2020

@author: 邹风院
"""
'''
画生存曲线图
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter,KaplanMeierFitter 
from lifelines.statistics import logrank_test,multivariate_logrank_test
from matplotlib import font_manager
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sklearn.preprocessing import MinMaxScaler
from libtiff import TIFF
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve,auc

def sign_test(data,tested_var):
    global vars_translation
    global zimu_translation
    global tran_dict
    data = data[["生存时间(天)","是否死亡",tested_var]]
    data = data.dropna(thresh = data.shape[1])
    groups = data[tested_var]
    item = sorted(data[tested_var].unique())
    if len(item) == 2:
        whether_multi_test = False
    else:
        whether_multi_test = True
    fenzu_dict = {}
    for value in item:
        fenzu_dict[value] = (groups == value)
    print(fenzu_dict)
    kmf = KaplanMeierFitter()
    T = data["生存时间(天)"]
    E = data["是否死亡"]
    color_list = ['blue','red','darkcyan','violet','pink']
    if tested_var == 'CT_MRI-原发肿瘤最大径(cm)_cut':
        tt = list(fenzu_dict.keys())
        tmp_keylist = [tt[0],tt[1],tt[2]]
        print(tmp_keylist)
    else:
        tmp_keylist = sorted(list(fenzu_dict.keys()))
    with PdfPages('KM曲线图//'+zimu_translation[tested_var]+'-'+tested_var+'.pdf') as pdf:
        for i,key in enumerate(tmp_keylist):
            if i == 0:
                kmf.fit(T[fenzu_dict[key]], event_observed=E[fenzu_dict[key]],label = tran_dict[tested_var][key])
                a1 = kmf.plot(ci_show = False,linewidth=1.5,color = color_list[i])
            else:
                kmf.fit(T[fenzu_dict[key]], event_observed=E[fenzu_dict[key]],label = tran_dict[tested_var][key])
                kmf.plot(ax = a1,ci_show = False,linewidth=1.5,color = color_list[i])
        if whether_multi_test:
            results = multivariate_logrank_test(T,E,groups)
            results.print_summary()
        else:
            results = logrank_test(T[fenzu_dict[item[0]]], T[fenzu_dict[item[1]]], event_observed_A=E[fenzu_dict[item[0]]], event_observed_B=E[fenzu_dict[item[1]]])
            results.print_summary()
        font = {'family' : 'Times New Roman',
                 'weight' : 'normal',
                 'size'   : 12
                 }
    
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.xlabel('Survival time (days)',font)
        plt.ylabel('Overall survival probability',font)
        plt.title(vars_translation[tested_var],font)
        plt.legend(fontsize=7,prop = {"family" : "Times New Roman"},frameon = False)
        plt.yticks(fontproperties = 'Times New Roman', size = 10)
        plt.xticks(fontproperties = 'Times New Roman', size = 10)
        if results.p_value<0.001:
            text = 'p < 0.001'
        else:
            text = 'p = '+str(round(results.p_value,3))
        if tested_var in ['CT_MRI-原发肿瘤最大径(cm)_cut','专科检查-淋巴结肿大','CT_MRI-淋巴结转移','病理分期-T分期','病理分期-N分期','既往史-颌面部放射线接触史','个人史-吸烟史']:
            plt.text(500, 0.1, r'$%s$'%text,fontdict={'size': 12,'family':'Times New Roman'})
        else:
            plt.text(460, 0.28, r'$%s$'%text,fontdict={'size': 12,'family':'Times New Roman'})
        pdf.savefig()
        #plt.close()
    plt.savefig('KM曲线图//'+zimu_translation[tested_var]+'-'+tested_var+'.tiff', bbox_inches='tight')
    return results

vars_translation = {'性别':'Gender',
                    '年龄':'Age',
                    '专科检查-原发肿瘤主体部位':'Primary site',
                    '专科检查-原发肿瘤最大径(cm)_cut':'PE-T',
                    'CT_MRI-原发肿瘤最大径(cm)_cut':'IE-T',
                    '专科检查-淋巴结肿大':'PE-N',
                    'CT_MRI-淋巴结转移':'IE-N',
                    '病理分期-T分期':'P-T',
                    '病理分期-N分期':'P-N',
                    '病理类型-鳞癌：分化程度':'Histologic grade',
                    '既往史-颌面部放射线接触史':'Radiotherapy history',
                    '个人史-吸烟史':'Smoking history',
                    'risk':'Risk'}
zimu_translation = {'性别':'A',
                    '年龄':'B',
                    '专科检查-原发肿瘤主体部位':'C',
                    '专科检查-原发肿瘤最大径(cm)_cut':'D',
                    'CT_MRI-原发肿瘤最大径(cm)_cut':'E',
                    '专科检查-淋巴结肿大':'F',
                    'CT_MRI-淋巴结转移':'G',
                    '病理分期-T分期':'H',
                    '病理分期-N分期':'I',
                    '病理类型-鳞癌：分化程度':'J',
                    '既往史-颌面部放射线接触史':'K',
                    '个人史-吸烟史':'L',
                    'risk':'M'}

data_n = pd.read_excel('原始数据(已完成数据填补).xlsx')#样本量683
bins = [-0.00001,2,4,10]
data_n['专科检查-原发肿瘤最大径(cm)_cut'] = pd.cut(data_n['专科检查-原发肿瘤最大径(cm)'],bins)
data_n['CT_MRI-原发肿瘤最大径(cm)_cut'] = pd.cut(data_n['CT_MRI-原发肿瘤最大径(cm)'],bins)
tran_dict = {}
tran_dict['性别']={1:'Male',2:'Female'}
tran_dict['既往史-颌面部放射线接触史']={0:'No',1:'Yes',2:'Not clearly'}
tran_dict['个人史-吸烟史']={0:'No',1:'Yes',2:'Not clearly'}
tran_dict['专科检查-原发肿瘤主体部位']={1:'Tongue',2:'Floor of mouth',3:'Gingiva',4:'Hard palate',5:'Others'}
tran_dict['专科检查-淋巴结肿大']={0:'N0',1:'N1',2:'N2'}
tran_dict['CT_MRI-淋巴结转移']={0:'N0',1:'N1',2:'N2'}
tran_dict['病理分期-N分期']={0:'N0',1:'N1',2:'N2'}
tran_dict['病理分期-T分期']={1:'T1',2:'T2',3:'T3',4:'T4'}
tran_dict['年龄']={1:'< 60',2:'≥ 60'}
tmp_1 = data_n['专科检查-原发肿瘤最大径(cm)_cut'].unique()
tran_dict['专科检查-原发肿瘤最大径(cm)_cut']={tmp_1[1]:'(0,2] cm',tmp_1[2]:'(2,4] cm',tmp_1[0]:'>4 cm'}
tmp_2 = data_n['CT_MRI-原发肿瘤最大径(cm)_cut'].unique()
tran_dict['CT_MRI-原发肿瘤最大径(cm)_cut']={tmp_1[1]:'(0,2] cm',tmp_1[2]:'(2,4] cm',tmp_1[0]:'>4 cm'}
tran_dict['病理类型-鳞癌：分化程度']={1:'Well differentiated',2:'Moderately differentiated',3:'Poorly differentiated',4:'Others'}
tran_dict['risk'] = {0:'Low risk',1:'High risk'}
#data_n['性别'][data_n['性别'] == 2] = 0
#data_n['年龄'][data_n['年龄'] == 2] = 0
#for key in vars_translation.keys():
#for var in vars_translation:
#    results = sign_test(data_n,var)
results = sign_test(data_n,'CT_MRI-原发肿瘤最大径(cm)_cut')
'''
train_data = pd.read_table(r'回归系数表及列线图数据集/其他因子含得分train数据集.txt',sep = '\t')
test_data = pd.read_table(r'回归系数表及列线图数据集/其他因子含得分test数据集.txt',sep = '\t')


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return y,optimal_threshold, point

def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    y,optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return y,fpr, tpr, roc_auc, optimal_th, optimal_point
y,fpr, tpr, roc_auc, median, optimal_point = ROC(train_data['是否死亡'], train_data['points'])
plt.figure(1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
'''
'''
fpr, tpr, thresholds = roc_curve(train_data['是否死亡'], train_data['points'], pos_label=1)
add_score = fpr+tpr
median = thresholds[add_score.argmax()]
plt.plot(fpr, tpr)
plt.show()
'''
'''
#median = train_data['points'].median()
train_data['risk'] = train_data['points'].apply(lambda x:1 if x > median else 0)
test_data['risk'] = test_data['points'].apply(lambda x:1 if x > median else 0)
#results = sign_test(train_data,'risk')
#results = sign_test(test_data,'risk')

'''



      
        
        
        
        
        
        
        
        
        
        
        
            
    
    
    


