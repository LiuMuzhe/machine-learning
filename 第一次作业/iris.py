# %%
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df['species'] = df['target'].apply(lambda x: data.target_names[x])

df.head()

# %%
import seaborn as sns
import matplotlib.pylab as plt
sns.pairplot(df, hue="species")

# %%
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

feature_names_cn = ['萼片长度 (cm)', '萼片宽度 (cm)', '花瓣长度 (cm)', '花瓣宽度 (cm)']

for i in range(4):
    data_by_class = []
    for j in range(3):
        class_data = df[df['target'] == j].iloc[:, i].values
        data_by_class.append(class_data)
    
    bp = axes[i].boxplot(data_by_class, 
                          labels=data.target_names,
                          patch_artist=True,
                          medianprops=dict(color='red', linewidth=2))
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[i].set_title(f'{feature_names_cn[i]}', fontsize=12)
    axes[i].set_xlabel('鸢尾花种类', fontsize=10)
    axes[i].set_ylabel(feature_names_cn[i], fontsize=10)
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %%
corr_m = df.drop(columns=["target","species"]).corr()
sns.heatmap(corr_m, annot=True, cmap="coolwarm", fmt=".2f")

plt.show()

# %%
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["target", "species"])
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(len(X))

# %% [markdown]
# # KNN（K近邻）

# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# 分割数据集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

param_grid_cnn = [
    {'weights': ['uniform'], 'n_neighbors': list(range(1, 11))}, 
]
# 初始化 KNN 分类器和网格搜索模型
model_knn = KNeighborsClassifier(n_neighbors=4)
grid_search_cnn = GridSearchCV(model_knn, param_grid_cnn, cv=5)

# 训练模型和调参
grid_search_cnn.fit(X_train, Y_train)
print("最佳参数：", grid_search_cnn.best_params_)
print("最佳准确率：", grid_search_cnn.best_score_)

best_knn = grid_search_cnn.best_estimator_

# 预测
y_pred = best_knn.predict(X_test)

# 评估模型
print("准确率：", accuracy_score(Y_test, y_pred))

cm_knn = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix(KNN):")
print(cm_knn)

report_knn = classification_report(Y_test, y_pred)
print("Classification Report(KNN):")
print(report_knn)

# %% [markdown]
# # 决策树

# %%
from sklearn.tree import DecisionTreeClassifier

# X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
param_grid_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 初始化 DT 分类器
model_dt = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(model_dt, param_grid_dt, cv=5)

# 训练模型和调参
grid_search_dt.fit(X_train, Y_train)
print("最佳参数：", grid_search_dt.best_params_)
print("最佳准确率：", grid_search_dt.best_score_)

best_dt = grid_search_dt.best_estimator_
# 预测
y_pred_dt = best_dt.predict(X_test)

# 评估模型
accuracy_dt = accuracy_score(Y_test, y_pred_dt)
print(f"Decision Tree Accuracy:{accuracy_dt}")

cm_dt = confusion_matrix(Y_test, y_pred_dt)
print("Confusion Matrix(KNN):")
print(cm_dt)

report_dt = classification_report(Y_test, y_pred_dt)
print("Classification Report(KNN):")
print(report_dt)

# %% [markdown]
# # SVM（支持向量机）

# %%
from sklearn.svm import SVC

param_grid_svm = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

# 初始化 SVM 分类器
model_svm = SVC(random_state=42)
grid_search_svm = GridSearchCV(model_svm, param_grid_svm, cv=5)

# 训练模型和调参
grid_search_svm.fit(X_train, Y_train)
print("最佳参数：", grid_search_svm.best_params_)
print("最佳准确率：", grid_search_svm.best_score_)

best_svm = grid_search_svm.best_estimator_
# 预测
y_pred_svm = best_svm.predict(X_test)

# 评估模型
accuracy_svm = accuracy_score(Y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

cm_svm = confusion_matrix(Y_test, y_pred_svm)
print("Confusion Matrix(KNN):")
print(cm_svm)

report_svm = classification_report(Y_test, y_pred_svm)
print("Classification Report(KNN):")
print(report_svm)

# %% [markdown]
# # 交叉验证

# %%
from sklearn.model_selection import cross_val_score
# KNN
print("KNN:")
cross_val_scores_knn = cross_val_score(best_knn, X_scaled, y, cv=5)
print(f"Cross-validation Scores: {cross_val_scores_knn}")
print(f"Mean CV Accuracy: {cross_val_scores_knn.mean():.4f}")
print()

# DT
print("DT:")
cross_val_scores_dt = cross_val_score(best_dt, X_scaled, y, cv=5)
print(f"Cross-validation Scores: {cross_val_scores_dt}")
print(f"Mean CV Accuracy: {cross_val_scores_dt.mean():.4f}")
print()

# SVM
print("SVM:")
cross_val_scores_svm = cross_val_score(best_svm, X_scaled, y, cv=5)
print(f"Cross-validation Scores: {cross_val_scores_svm}")
print(f"Mean CV Accuracy: {cross_val_scores_svm.mean():.4f}")


# %%
print(Y_train.value_counts())
print(Y_test.value_counts())
# print(y)


