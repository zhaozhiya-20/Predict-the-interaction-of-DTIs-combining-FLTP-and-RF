# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
import matplotlib.pyplot as plt
from numpy import interp
mean_fpr = np.linspace(0, 1, 100)

input_path = r'E:\论文\DTI\FLTP_V2\data\IC_N.csv'
input_path1 = r'E:\论文\DTI\FLTP_V2\data\IC_P.csv'
input_data = pd.read_csv(input_path)
input_data1 = pd.read_csv(input_path1)

input0 = np.array(input_data)
input1 = np.array(input_data1)
input = np.vstack((input0, input1))  # 按照列合并矩阵

index = [i for i in range(2950)]
random.shuffle(index)
input_new = input[index]

x = input_new[:, 1:1000]
y = input_new[:, 0]


print("********************************")
print("N_EN的尺寸:", input0.shape)
print("P_EN的尺寸:", input1.shape)
print("合并后矩阵的尺寸:", input.shape)
print("乱序后矩阵的尺寸:", input_new.shape)
print("*type_X", type(x))
print("*X_shape:", x.shape)
print("*type_Y", type(y))
print("*Y_shape:", y.shape)
print("********************************")




KF = KFold(n_splits=5)

X_train = []
Y_train = []
X_test = []
Y_test = []

for train_index, test_index in KF.split(x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train.append(x_train)
    Y_train.append(y_train)
    X_test.append(x_test)
    Y_test.append(y_test)



Y_hat = []
Real_data = []
Predict_data = []

clf = svm.SVC(C=0.7, kernel='poly', gamma=40, decision_function_shape='ovo', probability=True)
for q in range(0, 5):
    clf.fit(X_train[q], Y_train[q])
    y = clf.predict(X_test[q])
    y11 = clf.predict_proba(X_test[q])
    Real_data.append(Y_test[q])
    Predict_data.append(y11[:, 1])
    score = clf.score(X_test[q], Y_test[q])
    print(y)

    C2 = confusion_matrix(Y_test[q], y, labels=[0, 1])
    print(C2)
    print(C2.ravel())
    C = C2.ravel()
    MCC = (C[0] * C[3] - C[1] * C[2]) / ((C[3] + C[1]) * (C[3] + C[2]) * (C[0] + C[1]) * (C[0] + C[2])) ** 0.5
    Sen = C[3] / (C[3] + C[2])
    Acc = (C[3] + C[0]) / (C[3] + C[0] + C[1] + C[2])
    Pre = C[3] / (C[1] + C[3])
    Spec = C[0] / (C[0] + C[1])
    print("******************************************")
    print("Acc:", Acc)
    print("Pre:", Pre)
    print("Sen:", Sen)
    print("Spec:", Spec)
    print("MCC:", MCC)
    # print(classification_report(y, final_prediction, digits=4))  # Pre、Sen(recall)
    print("******************************************")
print("len_Real_data[0]:", len(Real_data[0]))
print("len_Predict_data:", len(Predict_data))
print("Predict_data[0]", Predict_data[0])

tprs = []

fpr, tpr, threshold = roc_curve(Real_data[0], Predict_data[0])
tprs.append(interp(mean_fpr, fpr, tpr))
fpr1, tpr1, threshold1 = roc_curve(Real_data[1], Predict_data[1])
tprs.append(interp(mean_fpr, fpr, tpr))
fpr2, tpr2, threshold2 = roc_curve(Real_data[2], Predict_data[2])
tprs.append(interp(mean_fpr, fpr, tpr))
fpr3, tpr3, threshold3 = roc_curve(Real_data[3], Predict_data[3])
tprs.append(interp(mean_fpr, fpr, tpr))
fpr4, tpr4, threshold4 = roc_curve(Real_data[4], Predict_data[4])
tprs.append(interp(mean_fpr, fpr, tpr))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)

roc_auc = auc(fpr, tpr)
print(roc_auc)
roc_auc1 = auc(fpr1, tpr1)
print(roc_auc1)
roc_auc2 = auc(fpr2, tpr2)
print(roc_auc2)
roc_auc3 = auc(fpr3, tpr3)
print(roc_auc3)
roc_auc4 = auc(fpr4, tpr4)
print(roc_auc4)  # 计算auc的值
average_auc = (roc_auc+roc_auc1+roc_auc2+roc_auc3+roc_auc4)/5
lw = 4
fig = plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC_Fold1  (area = %0.3f)' % roc_auc)
plt.plot(fpr, tpr, color='black', lw=lw, label='1st fold = %0.4f' % roc_auc)
plt.plot(fpr1, tpr1, color='blue', lw=lw, label='2nd fold = %0.4f' % roc_auc1)
plt.plot(fpr2, tpr2, color='red', lw=lw, label='3rd fold = %0.4f' % roc_auc2)
plt.plot(fpr3, tpr3, color='yellow', lw=lw, label='4th fold = %0.4f' % roc_auc3)
plt.plot(fpr4, tpr4, color='green', lw=lw, label='5th fold = %0.4f' % roc_auc4)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.text(0.22, 0.05, "Average AUC = %0.4f" % average_auc, size=34, alpha=1)
plt.xlabel('1-Specificity', fontsize=30)
plt.ylabel('Sensitivity', fontsize=30)
# plt.title('Receiver operating characteristic example')
plt.legend(loc="right", fontsize=26)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.tick_params(labelsize=24)
plt.show()

# fig.savefig(r'E:\论文\DTI\FLTP_V2\NEW\SVM\TIFF\EN_SVM.tif', dpi=600, format='tif')

# np.savetxt(r"E:\论文\DTI\FLTP_V2\NEW\SVM\TFR\EN_Fpr.csv", mean_fpr, delimiter=",")
# np.savetxt(r"E:\论文\DTI\FLTP_V2\NEW\SVM\TFR\EN_Tpr.csv", mean_tpr, delimiter=",")

