import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Preprocessing ===============================================

# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("=== Preprocessing Q1 ===")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n=== Preprocessing Q2 ===")
print(X_train_scaled.mean(axis=0))
# We fit the scaler only on X_train to prevent data leakage from the test set.


# KNN ==============================-==========

# Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_unscaled = knn.predict(X_test)

print("\n=== KNN Q1 ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_unscaled)}")
print(classification_report(y_test, y_pred_unscaled, target_names=iris.target_names))

# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("=== KNN Q2 ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_scaled)}")
# On Iris, scaling has no discernible impact. Even without scaling, none of the 
# four features, sepal/petal length and width, dominates KNN's distance calculations 
# because they are all measured in centimeters and have comparable magnitudes. 
# Scaling usually greatly aids KNN on datasets where features have widely disparate sizes.

# Q3
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("\n=== KNN Q3 ===")
print(f"Fold scores: {cv_scores}")
print(f"Mean:        {cv_scores.mean()}")
print(f"Std:         {cv_scores.std()}")
# A single train/test split is less reliable than cross-validation. A single split is mostly 
# dependent on the 20% of samples that just so happened to fall into the test set; by chance, 
# you may receive an exceptionally simple or difficult partition. Because CV averages over 
# five distinct splits, the estimate is much less susceptible to any one fortunate or unfortunate split.

# Q4
print("\n=== KNN Q4 ===")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
k_scores = {}
for k in k_values:
    model_k = KNeighborsClassifier(n_neighbors=k)
    scores   = cross_val_score(model_k, X_train, y_train, cv=5)
    k_scores[k] = scores.mean()
    print(f"k={k}  mean CV accuracy={scores.mean()}")

best_k = max(k_scores, key=k_scores.get)
print(f"\nChosen k: {best_k}")
# On Iris, k=7 or k=9 typically win (or draw). Lower k values, such as k=1 overfit, result 
# in a very jagged decision boundary that memorizes noise. By allowing distant, unimportant 
# neighbors to vote, very high k values underfit. The delicious Spot is a moderate k that 
# strikes a balance between these two extremes; as cross-validation measures generalization 
# directly, the winning k is the best option.

# Classifier Evaluation =========================================================================

# Q1
cm = confusion_matrix(y_test, y_pred_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=True)
ax.set_title("KNN (k=5, unscaled) — Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/knn_confusion_matrix.png", dpi=150)
plt.close()
print("\n=== Classifier Evaluation Q1 ===")
print("Confusion matrix saved to outputs/knn_confusion_matrix.png")
# The most commonly confused pair is Versicolor and Virginica. Setosa is linearly 
# separable from the other two and is never misclassified, despite their significant 
# overlap in feature space.

# The sklearn API: Decision Trees =============================================================================

# Q1
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Decision Tree Q1 ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt)}")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))
# On Iris, decision tree accuracy is close to 97%, just like KNN.

# On this clean, little dataset, neither approach has a definite advantage.
# A Decision Tree is unaffected by scaled or unscaled data since trees divide on 
# individual feature thresholds rather than point distances, so multiplying a 
# feature by a constant only modifies the threshold proportionately; the tree's 
# structure remains the same.


# Logistic Regression and Regularization =========================================================

# Q1
print("\n=== Logistic Regression Q1 ===")
for C in [0.01, 1.0, 100]:
    lr = LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    lr.fit(X_train_scaled, y_train)
    total_coef = np.abs(lr.coef_).sum()
    print(f"C={C:6.2f}  total |coef| = {total_coef}")
# The total coefficient magnitude rises and the regularization strength falls 
# as C increases. The bias-variance tradeoff is at work when low C (strong regularization) 
# pushes coefficients toward zero, preventing the model from depending too 
# much on any one feature. Coefficients can overfit on noisy data when C is 
# high because it allows them to develop freely. In essence, regularization 
# penalizes the complexity of the model.


# PCA ==========================================

digits  = load_digits()
X_digits = digits.data
y_digits = digits.target
images   = digits.images

# Q1
print("\n=== PCA Q1 ===")
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape:   {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(14, 2))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit), fontsize=10)
    axes[digit].axis('off')
fig.suptitle("Sample digit images (one per class)", y=1.02)
plt.tight_layout()
plt.savefig("outputs/sample_digits.png", dpi=150, bbox_inches='tight')
plt.close()
print("Sample digit images saved to outputs/sample_digits.png")

# Q2
print("\n=== PCA Q2 ===")
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D projection of Digits dataset")
plt.tight_layout()
plt.savefig("outputs/pca_2d_projection.png", dpi=150)
plt.close()
print("PCA 2D projection saved to outputs/pca_2d_projection.png")
# Although not exactly, same-digit images do tend to cluster together. In 2D, 
# digits like 0 and 1 form relatively compact, well-separated clusters, while 
# others like 4, 7, and 9 exhibit more overlap, which makes sense given their 
# comparable stroke patterns. There should be some confusion because two 
# components only account for a small portion of the total variation.

# Q3
print("\n=== PCA Q3 ===")
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumvar) + 1), cumvar, linewidth=1.5)
plt.axhline(0.80, color='red', linestyle='--', label='80% threshold')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA — Cumulative Explained Variance (Digits)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png", dpi=150)
plt.close()
print("PCA variance explained plot saved to outputs/pca_variance_explained.png")
n_80 = np.searchsorted(cumvar, 0.80) + 1
print(f"Components needed for 80% variance: {n_80}")
# To account for 80% of the variance, about {n_80} components are required.
# The majority of the data is concentrated in a little portion of the 64 
# initial dimensions; the curve first climbs sharply before leveling out.

# Q4
print("\n=== PCA Q4 ===")
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values  = [2, 5, 15, 40]
n_samples = 5
fig, axes = plt.subplots(len(n_values) + 1, n_samples, figsize=(10, 10))

# Original row
for col in range(n_samples):
    axes[0, col].imshow(images[col], cmap='gray_r')
    axes[0, col].axis('off')
axes[0, 0].set_ylabel("Original", rotation=90, fontsize=9, labelpad=40)

# Reconstruction rows
for row, n in enumerate(n_values, start=1):
    for col in range(n_samples):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap='gray_r')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f"n={n}", rotation=90, fontsize=9, labelpad=40)

plt.suptitle("PCA Digit Reconstructions", y=1.01)
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", dpi=150, bbox_inches='tight')
plt.close()
print("PCA reconstructions saved to outputs/pca_reconstructions.png")
# Around n=15 components, digits start to become recognizable. They are hazy 
# blobs with hardly any recognizable shape at n=2. Rough outlines start to 
# ppear at n=5, but the numbers are still unclear. The majority of the digits 
# are easily readable at n=15, which corresponds to the point at which the 
# variance curve begins to flatten; the majority of the significant structure 
# is captured in the first ~15 components. The reconstruction and the original 
# are almost identical at n=40.