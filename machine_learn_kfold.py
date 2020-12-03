#First machine learning project


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


# transform categorical variables to integer and one hot encoded (training)
df_train['mcint_enc']=le.fit_transform(df_train['mcint'])
df_train['mcint_evol_enc']=le.fit_transform(df_train['mcint_evol'])
df_train['Zpc_start']= df_train['mcint_evol'].str[0:3]
df_train['Zpc_end']=df_train['mcint_evol'].str[3:6]
df_train['z1'],df_train['p1'],df_train['c1'],df_train['z2'],df_train['p2'],df_train['c2'] = \
  [df_train['mcint_evol'].str[0],df_train['mcint_evol'].str[1],df_train['mcint_evol'].str[2],df_train['mcint_evol'].str[3],df_train['mcint_evol'].str[4],df_train['mcint_evol'].str[5]]
#df_train['c2'],df_train['p2'],df_train['z2'],df_train['c1'],df_train['p1'],df_train['z1'] = \
# [df_train['mcint_evol'].str[5],df_train['mcint_evol'].str[4],df_train['mcint_evol'].str[3],df_train['mcint_evol'].str[2],df_train['mcint_evol'].str[1],df_train['mcint_evol'].str[0]]
onehot_tr_mcevol = onehot_encoder.fit_transform(df_train['mcint_evol_enc'].values.reshape(len(df_train),1))

df_dummies = pd.get_dummies(df_train[df_train.columns[7:13]])

# Load test set (Cycle 23)
filename_test = "mcint_ml23.csv"
df_test = pd.read_csv(filename_test, names=names, \
                      dtype={'mcint':str, 'mcint_evol':str,\
                             'class':np.float64})

# Encoding categorical variables                              
df_test['mcint_enc']=le.fit_transform(df_test['mcint'])
df_test['mcint_evol_enc']=le.fit_transform(df_test['mcint_evol'])
df_test['z1'],df_test['p1'],df_test['c1'],df_test['z2'],df_test['p2'],df_test['c2'] = \
  [df_test['mcint_evol'].str[0],df_test['mcint_evol'].str[1],df_test['mcint_evol'].str[2],df_test['mcint_evol'].str[3],df_test['mcint_evol'].str[4],df_test['mcint_evol'].str[5]]

onehot_te_mcevol= onehot_encoder.fit_transform(df_test['mcint_evol_enc'].values.reshape(len(df_test),1))

#Prints the first 20 rows
#print(df_train.head(20))


# some stats of each attribute 
#print(df_train.describe())

# class distribution (check number of classes that each entry falls into)
#print(df_train.groupby('class').size())


# Split-out validation df values
array_train = df_train.values

cycle_test = input("Which cycle to train and test on? ")
method_test = input("Which method would you like to use? ")
encode = input("Would you like to encode? ")

if encode == 'yes':
    encode_n = 3
    enc_str = 'enc'
elif encode == 'no':
    encode_n = 0
    enc_str = 'no_enc'
else:
    print('Not a valid encode choice, please enter yes or no')

if method_test == 'static':
    a = 0 + encode_n
    b= 1 + encode_n
elif method_test == 'evol':
    a = 1 + encode_n
    b= 2 + encode_n
elif method_test == 'both':
    a = 0 + encode_n
    b = 2 + encode_n
elif method_test == 'sep_zpc':
    a= 7 + encode_n
    b = 13 + encode_n
elif method_test == 'zpc1':
        a= 5 + encode_n
        b= 6 + encode_n
elif method_test == 'zpc1_sep':
        a= 7 + encode_n
        b= 10 + encode_n
elif method_test == 'zpc2_sep':
        a= 10 + encode_n
        b= 13 + encode_n
elif method_test == 'zpc2':
    a= 6 + encode_n
    b = 7 + encode_n
elif method_test == 'zpc_both':
    a=5 + encode_n
    b= 7 + encode_n
elif method_test == 'one_hot_sep':
    X_train = df_dummies
    Y_train = df_train['class']
else:
    print('Incorrect method choice, pls restart')

if cycle_test == '22':
#Cycle 22 train
    if method_test == 'one_hot_sep':
        print('one hot')
    else:
        X_train= df_train[df_train.columns[a:b]]
        Y_train = df_train['class']
        X_test= df_train[df_train.columns[a:b]]
        Y_test = df_train['class']
    
elif cycle_test == '23':
    #Cycle 23 test
    X_train = df_test[df_train.columns[a:b]]
    Y_train = df_test['class']
    X_test = df_test[df_train.columns[a:b]]
    Y_test = df_test['class']

    
# Test options and evaluation metric

scoring = 'accuracy'

# Add ml algorithms to list
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(probability=True)))
models.append(('RFC', RandomForestClassifier()))
# evaluate each model in turn

results = []
names = []
tss_full=[]
bss_full=[]
rankings=[]
importance=[]
cv = StratifiedKFold(n_splits=10)
j = 0
colors = ['pink', 'lightblue', 'lightgreen', 'peachpuff','mediumpurple']
# Loop over models and cross validate them using BSS and TSS
for name, model in models:
  
    bss=[]
    tss = []
    names.append(name)
    n_feat = X_train.shape[1]
    x_val = (X_train.values)
    # Reshape input variable array for inputting to cross_val_score
    x_arr = x_val.reshape(len(x_val),n_feat)
    y_arr = Y_train.values
   # Keeps class balance in each train/test fold

    i=0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig_roc, ax_roc = plt.subplots()
    # Calculate BSS and TSS values
    for train_ind, test_ind in cv.split(x_arr, y_arr):
        #print("TRAIN:", train_ind, "TEST:", test_ind)
        probs = model.fit(x_arr[train_ind],y_arr[train_ind]).predict_proba(x_arr[test_ind])
        predicted=  model.fit(x_arr[train_ind],y_arr[train_ind]).predict(x_arr[test_ind])
        conf_mat = confusion_matrix(y_arr[test_ind],predicted)
        if hasattr(model, "feature_importances_"):
            importance.append([name,model.feature_importances_])
            #importance.append(model.coef_)
        elif hasattr(model, "coef_"):
            importance.append([name, model.coef_])
        else:
            importance.append([name,'none'])
        bss.append(ss_custom.bss_calc(y_arr[test_ind],probs[:,1]))
        tss.append(ss_custom.tss_calc(conf_mat))
        fpr, tpr, thresholds = roc_curve(y_arr[test_ind], probs[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax_roc.plot(fpr, tpr, lw=1,alpha=0.5)
        i += 1
        #plot_calibration_curve(model, name, 1,
         #                  x_arr[train_ind], y_arr[train_ind],
          #                 x_arr[test_ind], y_arr[test_ind],method_test,colors[j])
        #plt.savefig(name+'_'+str(i)+'_reliability_diagram_'+method_test+'_'+cycle_test+'.eps')
    bss_full.append(bss)
    tss_full.append(tss)
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Plot ROC Curve

    ax_roc.plot([0, 1], [0, 1], linestyle=':', lw=2, color='k',label='No Skill', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    print(std_auc)
    #tss_roc = mean_tpr - mean_fpr
    #max_tss_roc = tss_roc.argmax()
    #ax_roc.plot(mean_fpr, mean_tpr, color='b',
        # label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         #lw=2, alpha=.8)
    #plt.scatter(mean_fpr[max_tss_roc],mean_tpr[max_tss_roc], label='Max TSS =%.2f'%(tss_roc[max_tss_roc]))
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
       #          label=r'$\pm$ 1 std. dev.')

    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('POFD')
    ax_roc.set_ylabel('POD')
    ax_roc.set_title('ROC '+name)
    ax_roc.legend(loc="lower right")
    ax_roc.text(0.75,0.1, 'AUC=%.2f $\pm$ %0.3f'%(mean_auc,std_auc))
    fig_roc.savefig('ROC_curve_'+name+'_'+method_test+'_'+cycle_test+'.eps',dpi=300)
    plt.close()
   # fig1.show()
    
    j += 1
# Plot reliability diagram

#-------------------------------------------------------------------------------


# Plot box plots of BSS values for each algorithm
means_bss = [np.mean(array) for array in bss_full]
best_algorithm_bss = names[means_bss.index(max(means_bss))]
fig_bss = plt.figure(1)
fig_bss.suptitle('Algorithm BSS Comparison')
ax_bss = fig_bss.add_subplot(111)
bplot1=plt.boxplot(bss_full,patch_artist=True)
#a =[item.get_ydata() for item in bplot1['boxes']]
locs, labels = plt.xticks() 
ax_bss.set_xticklabels(names)
ax_bss.set_ylabel('BSS')
ax_bss.set_xlabel('Algorithm')
ax_bss.set_ylim([0.0, .4])
plt.figtext(0.15,0.9,'Top Algorithm = '+best_algorithm_bss)
for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
fig_bss.savefig('algorithm_comp_bss_'+method_test+'_'+cycle_test+'.eps',dpi = 300)

# Plot box plots of each algorithm of TSS values
means_tss = [np.mean(array) for array in tss_full]
best_algorithm_tss = names[means_tss.index(max(means_tss))]
fig_tss = plt.figure()
fig_tss.suptitle('Algorithm TSS Comparison')
ax_tss = fig_tss.add_subplot(111)
bplot2=plt.boxplot(tss_full,patch_artist=True)
ax_tss.set_xticklabels(names)
ax_tss.set_ylabel('TSS')
ax_tss.set_xlabel('Algorithm')
ax_tss.set_ylim([0.3, .8])
plt.figtext(0.15,0.9,'Top Algorithm = '+best_algorithm_tss)
for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
fig_tss.savefig('algorithm_comp_tss_'+method_test+'_'+cycle_test+'.eps',dpi = 300)
#plt.show()

tss_arr = list(zip(names,tss_full))
bss_arr = list(zip(names,bss_full))

# Plot feature importance

feature_score = [x[1:][0] for x in importance]
with open('kfold_feature_importance.data', 'wb') as filehandle:  
    # store the data as binary data stream
    pickle.dump(feature_score, filehandle)
lr_feats = np.array(feature_score[0:10])
lda_feats = np.array(feature_score[10:20])
cart_feats = np.array(feature_score[30:40])
rfc_feats = np.array(feature_score[40:50])

lr_mean_ft =  lr_feats.mean(axis=0)
lda_mean_ft = lda_feats.mean(axis=0)
cart_mean_ft= cart_feats.mean(axis=0)
rfc_mean_ft= rfc_feats.mean(axis=0)


fig_imp,ax_imp = plt.subplots(2,2)


bx1=ax_imp[0,0].boxplot(lr_feats.reshape(10,6),patch_artist=True)
for patch in bx1['boxes']:
        patch.set_facecolor('pink')
bx2=ax_imp[0,1].boxplot(lda_feats.reshape(10,6),patch_artist=True)
for patch in bx2['boxes']:
        patch.set_facecolor('lightblue')
bx3=ax_imp[1,0].boxplot(cart_feats.reshape(10,6),patch_artist=True)
for patch in bx3['boxes']:
        patch.set_facecolor('peachpuff')
bx4=ax_imp[1,1].boxplot(rfc_feats.reshape(10,6),patch_artist=True)
for patch in bx4['boxes']:
        patch.set_facecolor( 'mediumpurple')
ax_imp[0,0].set_xticklabels( X_train.columns)
ax_imp[0,1].set_xticklabels(X_train.columns)
ax_imp[1,0].set_xticklabels(X_train.columns)
ax_imp[1,1].set_xticklabels(X_train.columns)
ax_imp[0,0].text(0.05,0.85,'LR',transform=ax_imp[0,0].transAxes)
ax_imp[0,1].text(0.05,0.85,'LDA',transform=ax_imp[0,1].transAxes)
ax_imp[1,0].text(0.05,0.85,'CART',transform=ax_imp[1,0].transAxes)
ax_imp[1,1].text(0.05,0.85,'RFC',transform=ax_imp[1,1].transAxes)
"""
ax_imp[0,0].bar(np.arange(n_feat),lr_mean_ft[0]/lr_mean_ft.max(),tick_label=X_train.columns)
ax_imp[0,1].bar(np.arange(n_feat),lda_mean_ft[0]/lda_mean_ft.max(),tick_label=X_train.columns,color='darkorchid')
ax_imp[1,0].bar(np.arange(n_feat),cart_mean_ft/cart_mean_ft.max(),tick_label=X_train.columns,color='coral')
ax_imp[1,1].bar(np.arange(n_feat),rfc_mean_ft/rfc_mean_ft.max(),tick_label=X_train.columns,color='forestgreen')

ax_imp[0,0].text(0,0.85,'LR')
ax_imp[0,1].text(0,0.85,'LDA')
ax_imp[1,0].text(0,0.85,'CART')
ax_imp[1,1].text(0,0.85,'RFC')
"""
fig_imp.suptitle('Feature Importance (K-Fold)')
fig_imp.savefig('feat_importances_'+method_test+'_'+cycle_test+'.eps')

        
# Make predictions using one hot encoding
validation_size=0.2
seed=9
#df_values = df_train.values
#x_data = onehot_tr_mcevol
#x_data = X_train
x_data = df_dummies
#x_data = df_values[:,1:2].reshape(len(df_train),1)
y_data = df_train['class'].values

"""
bss_test =[]
tss_test = []
importance =[]
for name, ml_algo in models:

    print('Running '+name+' one hot encoding')
    X_train_split, X_test_split, Y_train_split, Y_test_split = model_selection.train_test_split(x_data, y_data, \
                                                                                                test_size=validation_size,random_state=seed)
    ml_algo.fit(X_train_split,Y_train_split)       
    predictions = ml_algo.predict(X_test_split)
    predict_probs = ml_algo.predict_proba(X_test_split)
    try:
        importance.append(ml_algo.feature_importances_)
        importance.append(ml_algo.coef_)
    except:
        pass
    bss_test.append(ss_custom.bss_calc(Y_test_split,predict_probs[:,1]))
    tss_test.append(ss_custom.tss_calc(confusion_matrix(Y_test_split, predictions)))
    

means_bss_test = [np.mean(array) for array in bss_test]
best_algorithm_bss_test = names[means_bss_test.index(max(means_bss_test))]

fig_bss_ohe = plt.figure()
fig_bss_ohe.suptitle('Algorithm BSS Comparison')
ax_bss_ohe = fig_bss_ohe.add_subplot(111)
#bplot1=plt.boxplot(bss_full_test,patch_artist=True)
plt.scatter(names,bss_test)
#ax_bss_ohe.set_xticklabels(names)
ax_bss_ohe.set_ylabel('BSS')
ax_bss_ohe.set_xlabel('Algorithm')
plt.figtext(0.2,0.9,'Top Algorithm = %s (BSS+%d)'%(best_algorithm_bss_test,max(means_bss_Test))

fig_bss_ohe.savefig('algorithm_comp_bss_ohe_'+method_test+'_'+cycle_test+'.eps',dpi = 300)
fig_tss_ohe = plt.figure()
fig_tss_ohe.suptitle('Algorithm TSS Comparison')
ax_tss_ohe = fig_tss_ohe.add_subplot(111)
#bplot1=plt.boxplot(tss_full_test,patch_artist=True)
plt.scatter(names,tss_test)
#ax_tss_ohe.set_xticklabels(names)
ax_tss_ohe.set_ylabel('TSS')
ax_tss_ohe.set_xlabel('Algorithm')
#plt.figtext(0.2,0.9,'Top Algorithm = '+best_algorithm_tss)
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)

fig_tss_ohe.savefig('algorithm_comp_tss_ohe_'+method_test+'_'+cycle_test+
                        '.eps',dpi = 300)

auc_test = []
fig_3 = plt.figure()
ax_3 = fig_3.add_subplot(111)
fpr_test, tpr_test, thresholds_test = roc_curve(Y_test_split,
                                                predict_probs[:, 1])
roc_auc_test = auc(fpr_test, tpr_test)
ax_3.plot(fpr_test, tpr_test, lw=1,label='ROC (AUC = %0.2f)' % (roc_auc_test))
ax_3.plot([0, 1], [0, 1], linestyle='--',
          lw=2, color='r',label='Luck',
          alpha=.8)
ax_3.set_xlabel('False Positive Rate')
ax_3.set_ylabel('True Positive Rate')
ax_3.set_title('ROC')
ax_3.legend(loc="lower right")
fig_3.savefig('ROC_curve_cycle2223.eps')
print(accuracy_score(Y_test_split, predictions))
print(classification_report(Y_test_split, predictions))

plot_calibration_curve(RandomForestClassifier(), "RFC", 1,
                       X_train_split, Y_train_split,
                       X_test_split, Y_test_split)
                       
#plt.show(fig_3)
#plt.close('all')


# Check feature importance using RFE
X= df_train[df_train.columns[7:13]]
#X = df_train.drop(['class'],axis=1)
Y = df_train['class']

model = RandomForestClassifier()
rfe = RFE(model,1)

rfe = rfe.fit(X,Y)
print(rfe.support_)
print(rfe.ranking_)
"""
plt.show()
plt.close()
