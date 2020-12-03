# Machine Learning - Using McIntosh Evolutions of Sunspot Groups to Predict Flares in Solar Cycle 23 using Solar Cycle 22

# Load libraries
from __future__ import print_function
import pandas as pd
import numpy as np
# load additional module
import pickle
#import graphviz
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
# Import metrics for validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
# Import machine learning algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Import preprocessing functions for encoding data
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from calibration_curve_plot import plot_calibration_curve
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import ss_custom


# Allowed McIntosh class integer values

allowed_mci =  ['000',\
                   '101', '102', \
                   '210', '220', '230', '240', '250', \
                   '311', '312', '321', '322', '331', '332',\
                   '341', '342', '351', '352',\
                   '411', '412', '413', '421', '422', '423', '431', '432', '433', \
                   '441', '442', '443', '451', '452', '453', \
                   '511', '512', '521', '522', '523', '531', '532', '533', \
                   '541', '542', '543', '551', '552', '553', \
                   '611', '612', '621', '622', '623', '631', '632', '633', \
                   '641', '642', '643', '651', '652', '653']

zur_arr = ['A','B', 'H', 'C', 'D', 'E', 'F']
pen_arr= ['X', 'R', 'S', 'A','H', 'K']
dist_arr=['X', 'O', 'I' , 'C']

label_enc = preprocessing.LabelEncoder()
le_mcint = label_enc.fit_transform(allowed_mci)
ohe_mcint = OneHotEncoder(sparse=False)
ohe_mcint_fit = ohe_mcint.fit_transform(le_mcint.reshape(-1,1))

# Encode data using integer encoding and vector (OHE)
le = preprocessing.LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
# Load dataset for training (Cycle 22)
filename_train = "mcint_ml22.csv"
names = ['mcint','mcint_evol', 'class']
df_train = pd.read_csv(filename_train, names=names, dtype={'mcint':str, 'mcint_evol':str, 'class':np.float64})


# Transform categorical variables to integer and one hot encoded (training)
df_train['mcint_enc']=le.fit_transform(df_train['mcint'])
df_train['mcint_evol_enc']=le.fit_transform(df_train['mcint_evol'])
df_train['Zpc_start']= df_train['mcint_evol'].str[0:3]
df_train['Zpc_end']=df_train['mcint_evol'].str[3:6]
df_train['Z1'],df_train['p1'],df_train['c1'],df_train['Z2'],df_train['p2'],df_train['c2'] = \
  [df_train['mcint_evol'].str[0],df_train['mcint_evol'].str[1],df_train['mcint_evol'].str[2],df_train['mcint_evol'].str[3],df_train['mcint_evol'].str[4],df_train['mcint_evol'].str[5]]
#df_train['c2'],df_train['p2'],df_train['Z2'],df_train['c1'],df_train['p1'],df_train['Z1'] = \
# [df_train['mcint_evol'].str[5],df_train['mcint_evol'].str[4],df_train['mcint_evol'].str[3],df_train['mcint_evol'].str[2],df_train['mcint_evol'].str[1],df_train['mcint_evol'].str[0]]
onehot_tr_mcevol = onehot_encoder.fit_transform(df_train['mcint_evol_enc'].values.reshape(len(df_train),1))

df_train_dummies = pd.get_dummies(df_train[df_train.columns[7:13]])

# Load test set (Cycle 23)
filename_test = "mcint_ml23.csv"
df_test = pd.read_csv(filename_test, names=names, \
                      dtype={'mcint':str, 'mcint_evol':str,\
                             'class':np.float64})

# Encoding categorical variables                              
df_test['mcint_enc']=le.fit_transform(df_test['mcint'])
df_test['mcint_evol_enc']=le.fit_transform(df_test['mcint_evol'])
df_test['Zpc_start']= df_test['mcint_evol'].str[0:3]
df_test['Zpc_end']=df_test['mcint_evol'].str[3:6]
df_test['Z1'],df_test['p1'],df_test['c1'],df_test['Z2'],df_test['p2'],df_test['c2'] = \
  [df_test['mcint_evol'].str[0],df_test['mcint_evol'].str[1],df_test['mcint_evol'].str[2],df_test['mcint_evol'].str[3],df_test['mcint_evol'].str[4],df_test['mcint_evol'].str[5]]

onehot_te_mcevol= onehot_encoder.fit_transform(df_test['mcint_evol_enc'].values.reshape(len(df_test),1))
df_test_dummies = pd.get_dummies(df_test[df_test.columns[7:13]])
#Prints the first 20 rows
print(df_train.head(20))


# class distribution (check number of classes that each entry falls into)
print(df_train.groupby('class').size())


# Split-out validation df values
array_train = df_train.values
array_test = df_test.values

method_test = input("Which method would you like to use? ")
#method_test='sep_zpc'

# Use full starting class information encoded
if method_test == 'static':
    a = 0
    b= 1 
# Use full final class information encoded
elif method_test == 'evol':
    a = 1 
    b= 2 
# Use full starting & ending class information encoded
elif method_test == 'both':
    a = 0 
    b = 2 
elif method_test == 'sep_zpc':
    a= 7 
    b = 13 
elif method_test == 'zpc1':
        a= 5 
        b= 6 
elif method_test == 'zpc1_sep':
        a= 7 
        b= 10 
elif method_test == 'zpc2_sep':
        a= 10 
        b= 13 
elif method_test == 'zpc2':
    a= 6 
    b = 7 
elif method_test == 'zpc_both':
    a=5 
    b= 7
elif method_test =='choose':
  a=int(input('Which start column?'))
  b=int(input('Which end column?') )
elif method_test == 'one_hot_sep':
    X_train = df_train_dummies
    Y_train = df_train['class']
    X_test= df_test_dummies
    Y_test = df_test['class']
else:
    print('Incorrect method choice, pls restart')

if method_test == 'one_hot_sep':
    print('One Hot Encoding')
else:
    X_train = df_train[df_train.columns[a:b]]
    Y_train = df_train['class']
    X_test = df_test[df_test.columns[a:b]]
    Y_test = df_test['class']

    
# Test options and evaluation metric

scoring = 'accuracy'

# Add ml algorithms to list
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))

# Evaluate each model in turn
colors = ['pink', 'lightblue', 'lightgreen', 'peachpuff','mediumpurple']
results = []
names = []
tss_full=[]
bss_full=[]
fpr_full=[]
tpr_full=[]
thresholds_full=[]
auc_roc=[]
rankings=[]
importance=[]
i=0
# Loop over models
for name, model in models:
    
    bss=[]
    tss = []
    names.append(name)
    n_feat = X_train.shape[1]
    x_val = (X_train.values)
    # Reshape input variable array for inputting to cross_val_score
    x_train = X_train.values.reshape(len(X_train),n_feat)
    y_train = Y_train
    x_test = X_test.values.reshape(len(X_test),n_feat)
    y_test = Y_test

   # Keeps class balance in each train/test fold

    
    
    # Calculate BSS and TSS values
    probs = model.fit(x_train,y_train).predict_proba(x_test)
    predicted=  model.fit(x_train,y_train).predict(x_test)
    conf_mat = confusion_matrix(y_test,predicted)
    if hasattr(model, "feature_importances_"):
        importance.append([name,model.feature_importances_])
        #importance.append(model.coef_)
    elif hasattr(model, "coef_"):
        importance.append([name, model.coef_])
    else:
        importance.append([name,'none'])
    bss = ss_custom.bss_calc(y_test,probs[:,1])
    tss = ss_custom.tss_calc(conf_mat)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
    auc_roc.append(auc(fpr, tpr))

    
    plot_calibration_curve(model, name, 1,
                       x_train, y_train,
                       x_test, y_test,method_test,colors[i])
    plt.savefig(name+'_reliability_diagram_22_23_'+method_test+'.eps')
    fpr_full.append(fpr)
    tpr_full.append(tpr)
    thresholds_full.append(thresholds)
    bss_full.append(bss)
    tss_full.append(tss)
    i += 1  
# -----------------------------------------------------------------------------
plt.close()
# ------------------------------------------------------------------------------
# Plot ROC Curve
j=0
sort_auc=list(np.argsort(auc_roc))
sort_auc.reverse()
names_auc=[x for _,x in sorted(zip(sort_auc,names))]
fig1 = plt.figure()  # create a figure object
ax1 = fig1.add_subplot(1, 1, 1)  # create an axes object in the figure
ax1.plot([0, 1], [0, 1], linestyle=':', lw=2, color='k',label='No Skill', alpha=.8)
for fpr, tpr,auc in zip(fpr_full,tpr_full,auc_roc):

  ax1.plot(fpr, tpr, color=colors[j],
      label=r'ROC$_{\mathrm{%s}}$ (AUC = %0.2f)'%(names[j],auc),
      lw=2)
  j = j +1

ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlabel('POFD')
ax1.set_ylabel('POD')
ax1.set_title('ROC Curves')
handles,labels = ax1.get_legend_handles_labels()

handles = [handles[1], handles[2],handles[5],handles[4], handles[3]]
labels = [labels[1],labels[2],labels[5],labels[4],labels[3]]

ax1.legend(handles,labels)
#ax1.legend(loc="lower right")
fig1.savefig('ROC_curves_'+method_test+'_22_23.eps')

   # fig1.show()
    
    
# Plot reliability diagram

#-------------------------------------------------------------------------------
# Plot box plots to compare values of chosen scorer in cross_val_score
"""
means = [np.mean(array) for array in results]
best_algorithm = names[means.index(max(means))]
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_ylabel(scoring)
plt.figtext(0.15,0.9,'Top Algorithm = '+best_algorithm)
fig.savefig('algorithm_comp_'+scoring+'_'+method_test+'_'+cycle_test+'_'+enc_str+'.eps',dpi=300)
"""


# Plot box plots of BSS values for each algorithm
best_algorithm_bss = names[bss_full.index(max(bss_full))]
fig_bss = plt.figure()
fig_bss.suptitle('Algorithm BSS Comparison')
ax_bss = fig_bss.add_subplot(111)
bplot1=plt.scatter((np.arange(len(bss_full)))+1,bss_full,color=colors,edgecolors='k')
ax_bss.axhline(.09,linestyle=':',color='k',lw=0.8,label='Uncorrected Poisson BSS')
ax_bss.axhline(.2,linestyle='--',color='k',lw=0.8,label='Corrected Poisson BSS')
ax_bss.set_xticks((np.arange(len(bss_full)))+1)
ax_bss.set_xticklabels(names)
ax_bss.set_ylabel('BSS')
ax_bss.set_xlabel('Algorithm')
ax_bss.set_xlim(0.5,5.5)
ax_bss.set_ylim([0.0, .4])
ax_bss.legend()
plt.figtext(0.15,0.9,'Top Algorithm = '+best_algorithm_bss)
fig_bss.savefig('algorithm_comp_bss_'+method_test+'_22_23.eps')

# Plot box plots of each algorithm of TSS values

best_algorithm_tss = names[tss_full.index(max(tss_full))]
fig_tss = plt.figure()
fig_tss.suptitle('Algorithm TSS Comparison')
ax_tss = fig_tss.add_subplot(111)
bplot2=plt.scatter((np.arange(len(tss_full)))+1,tss_full,color=colors,edgecolors='k')
ax_tss.set_xticks((np.arange(len(tss_full)))+1)
ax_tss.set_xticklabels(names)
ax_tss.set_ylabel('TSS')
ax_tss.axhline(0.45,linestyle='-.',color='k',lw=0.8,label='Poisson TSS')
ax_tss.set_xlabel('Algorithm')
ax_tss.set_ylim([0.3, .8])
ax_tss.set_xlim(0.5,5.5)
ax_tss.legend()
plt.figtext(0.15,0.9,'Top Algorithm = '+best_algorithm_tss)
fig_tss.savefig('algorithm_comp_tss_'+method_test+'_22_23.eps')
#plt.show()

tss_arr = list(zip(names,tss_full))
bss_arr = list(zip(names,bss_full))

# Plot feature importance

feature_score_full = [x[1:][0] for x in importance]
lr_feats_full = np.array(feature_score_full[0])
lda_feats_full = np.array(feature_score_full[1])
cart_feats_full = np.array(feature_score_full[3])
rfc_feats_full = np.array(feature_score_full[4])


# open file and read the content in a list
with open('kfold_feature_importance.data', 'rb') as filehandle:  
    # read the data as binary data stream
    feats_kfold = pickle.load(filehandle)
lr_feats_kfold = np.array(feats_kfold[0:10])
lda_feats_kfold = np.array(feats_kfold[10:20])
cart_feats_kfold = np.array(feats_kfold[30:40])
rfc_feats_kfold = np.array(feats_kfold[40:50])

lr_mean_ft  =  lr_feats_kfold.mean(axis=0)
lda_mean_ft = lda_feats_kfold.mean(axis=0)
cart_mean_ft= cart_feats_kfold.mean(axis=0)
rfc_mean_ft = rfc_feats_kfold.mean(axis=0)


fig_imp,ax_imp = plt.subplots(4,2, figsize=(8,8))


bx1=ax_imp[0,0].boxplot(lr_feats_kfold.reshape(10,6),patch_artist=True)
for patch in bx1['boxes']:
        patch.set_facecolor('pink')
bx2=ax_imp[1,0].boxplot(lda_feats_kfold.reshape(10,6),patch_artist=True)
for patch in bx2['boxes']:
        patch.set_facecolor('lightblue')
bx3=ax_imp[2,0].boxplot(cart_feats_kfold.reshape(10,6),patch_artist=True)
for patch in bx3['boxes']:
        patch.set_facecolor('peachpuff')
bx4=ax_imp[3,0].boxplot(rfc_feats_kfold.reshape(10,6),patch_artist=True)
for patch in bx4['boxes']:
        patch.set_facecolor( 'mediumpurple')
ax_imp[0,0].set_xticklabels(X_train.columns)
ax_imp[1,0].set_xticklabels(X_train.columns)
ax_imp[2,0].set_xticklabels(X_train.columns)
ax_imp[3,0].set_xticklabels(X_train.columns)
ax_imp[0,0].set_title('K-Fold')
ax_imp[0,0].text(0.05,0.85,'(a) LR',transform=ax_imp[0,0].transAxes)
ax_imp[1,0].text(0.05,0.85,'(b) LDA',transform=ax_imp[1,0].transAxes)
ax_imp[2,0].text(0.05,0.85,'(c) CART',transform=ax_imp[2,0].transAxes)
ax_imp[3,0].text(0.05,0.85,'(d) RFC',transform=ax_imp[3,0].transAxes)


ax_imp[0,1].bar(np.arange(n_feat),lr_feats_full[0],tick_label=X_train.columns,color='pink')
ax_imp[1,1].bar(np.arange(n_feat),lda_feats_full[0],tick_label=X_train.columns,color='lightblue')
ax_imp[2,1].bar(np.arange(n_feat),cart_feats_full,tick_label=X_train.columns,color='peachpuff')
ax_imp[3,1].bar(np.arange(n_feat),rfc_feats_full,tick_label=X_train.columns,color='mediumpurple')
ax_imp[0,0].set_ylim(min(lr_feats_full[0])-0.1,1.)
ax_imp[0,1].set_ylim(min(lr_feats_full[0])-0.1,1.)
ax_imp[1,0].set_ylim(-0.2,1.2)
ax_imp[1,1].set_ylim(-0.2,1.2)
ax_imp[2,1].set_ylim(0,.65)
ax_imp[2,0].set_ylim(0,.65)
ax_imp[0,1].set_title('SC22->SC23')
ax_imp[3,1].set_ylim(0,.4)
ax_imp[3,0].set_ylim(0,.4)
ax_imp[0,1].text(0.05,0.85,'(e) LR',transform=ax_imp[0,1].transAxes)
ax_imp[1,1].text(0.05,0.85,'(f) LDA',transform=ax_imp[1,1].transAxes)
ax_imp[2,1].text(0.05,0.85,'(g) CART',transform=ax_imp[2,1].transAxes)
ax_imp[3,1].text(0.05,0.85,'(h) RFC',transform=ax_imp[3,1].transAxes)

fig_imp.suptitle('Feature Importance')
fig_imp.savefig('feat_importances_'+method_test+'_22_23.eps')

    
plt.show()
plt.close()
