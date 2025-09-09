import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK

from sklearn.metrics import roc_auc_score
import numpy as np
import os

# Define random seed
torch.manual_seed(1447)
np.random.seed(1447)

##---------------------Loading the Saved Features---------------------------------------------

train_save_dir = "extracted_features"
valid_save_dir = "extracted_features"
# Load training features
train_features_current = np.load(os.path.join(train_save_dir, 'features_current_train.npy'))
train_features_video = np.load(os.path.join(train_save_dir, 'features_video_train.npy'))
train_labels = np.load(os.path.join(train_save_dir, 'labels_train.npy'))

# Load validation features
valid_features_current = np.load(os.path.join(valid_save_dir, 'features_current_test.npy'))
valid_features_video = np.load(os.path.join(valid_save_dir, 'features_video_test.npy'))
valid_labels = np.load(os.path.join(valid_save_dir, 'labels_test.npy'))


print("Loaded training features current shape:", train_features_current.shape)
print("Loaded training features video shape:", train_features_video.shape)
print("Loaded training labels shape:", train_labels.shape)
print("Loaded validation features current shape:", valid_features_current.shape)
print("Loaded validation features video shape:", valid_features_video.shape)
print("Loaded validation labels shape:", valid_labels.shape)
# # Prepare dataset
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)  # X: [n_samples, n_features], y: [n_samples, 1]
# X = np.vstack((train_features_video, valid_features_video))
n_class = len(np.unique(train_labels))  # Num. of class

# # split train-test
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
#     x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
# ))


x_train = train_features_video
x_test = valid_features_video
y_train = train_labels
y_test = valid_labels #valid_labels

X = np.vstack((x_train, x_test))
# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
    x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
))

# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Define TSK model parameters
n_rule = 10  # Num. of rules
lr = 0.001  # learning rate
weight_decay = 1e-8
consbn = True
order = 1

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

# ----------------- optimizer ----------------------------
ante_param, other_param = [], []
for n, p in model.named_parameters():
    if "center" in n or "sigma" in n:
        ante_param.append(p)
    else:
        other_param.append(p)
optimizer = AdamW([
    {'params': ante_param, "weight_decay": 0},
    {'params': other_param, "weight_decay": weight_decay}
], lr=lr)

# ----------------- split 10% data for earlystopping -----------------
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# ----------------- define the earlystopping callback -----------------
EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=40, save_path="tmp.pkl")

wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
              epochs=30, callbacks=[EACC], ur=0, ur_tau=1/n_class)
wrapper.fit(x_train, y_train)
wrapper.load("tmp.pkl")

y_pred = wrapper.predict(x_test).argmax(axis=1)
print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))


# Assuming wrapper.predict(x_test) returns a NumPy array
outputs = wrapper.predict(x_test)

# Convert NumPy array to PyTorch tensor
outputs_tensor = torch.tensor(outputs)

# Apply softmax to the tensor to get the probability of the positive class
probabilities = torch.softmax(outputs_tensor, dim=1)[:, 1]  # Get probability of the positive class

# If you need to convert the result back to a NumPy array
probabilities_np = probabilities.numpy()

# Calculate AUC using scikit-learn
auc = roc_auc_score(y_test, probabilities_np)

class ClassificationMetric:
    # 本代码计算的指标仅适用于二分类
    def __init__(self, numClass=2):
        # assert numClass == 2, 'numClass must be 2'
        assert numClass == 2
        self.numClass = numClass
        # confusionMatrix = [[TN, FP],
        #                    [FN, TP]]
        # 行为label，列为预测
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, clsPredict, clsLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (clsLabel >= 0) & (clsLabel < self.numClass)
        label = self.numClass * clsLabel[mask] + clsPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, clsPredict, clsLabel):
        assert clsPredict.shape == clsLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(clsPredict, clsLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def Accuracy(self):
        # 所有类别的分类准确率
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + 1e-6)
        return acc

    def F1Score(self):
        # F1 score = 2 * Precision * Recall / (Precision + Recall)
        p = self.Precision()
        r = self.Recall()
        f1 = 2 * p * r / (p + r + 1e-5)
        return f1

    def Precision(self):
        # 精准率，预测的正样本中有多少是真实的正样本。值越大，性能越好
        # Precision =  TP / (TP + FP)
        p = self.confusionMatrix[1][1] / (self.confusionMatrix[1][1] + self.confusionMatrix[0][1] + 1e-6)
        return p

    def Recall(self):
        # 召回率，正样本被预测为正样本占总的正样本的比例。值越大，性能越好
        # Recall = TP / (TP + FN))
        r = self.confusionMatrix[1][1] / (self.confusionMatrix[1][1] + self.confusionMatrix[1][0] + 1e-6)
        return r

    def FalsePositiveRate(self):
        # 假阳率、误报率, 负样本被预测为正样本占总的负样本的比例。值越小, 性能越好
        # FPR = FP / (FP + TN)
        fpr = self.confusionMatrix[0][1] / (self.confusionMatrix[0][1] + self.confusionMatrix[0][0] + 1e-6)
        return fpr

    def FalseNegativeRate(self):
        # 假阴率、漏报率，正样本被预测为负样本占总的正样本的比例。值越小，性能越好
        # FNR = FN / (TP + FN)
        fnr = self.confusionMatrix[1][0] / (self.confusionMatrix[1][1] + self.confusionMatrix[1][0] + 1e-6)
        return fnr
    
# 创建一个 ClassificationMetric 对象
metric = ClassificationMetric()

y_pred = y_pred.astype(int)
y_test = y_test.astype(int)        # 将预测结果和真实标签加入到混淆矩阵中
metric.addBatch(y_test, y_pred)


# 计算并打印各类评估指标
print("Accuracy: ", metric.Accuracy())
print("F1 Score: ", metric.F1Score())
print("False Positive Rate: ", metric.FalsePositiveRate())
print("False Negative Rate: ", metric.FalseNegativeRate())

# Calculate total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')
