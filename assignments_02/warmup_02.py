#%%
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split
#%%
os.makedirs("outputs", exist_ok=True)

# --- scikit-learn API --- 

# Q1
#%%
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4 = model.predict([[4]])[0]
pred_8 = model.predict([[8]])[0]

print("scikit-learn Q1")
print(f"Predicted salary at 4 years: ${pred_4:,.2f}")
print(f"Predicted salary at 8 years: ${pred_8:,.2f}")
print(f"Slope (coef): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
#%%

# Q2
#%%
x = np.array([10, 20, 30, 40, 50])
print("\n--- scikit-learn Q2 ---")
print(f"Original shape: {x.shape}")

x_2d = x.reshape(-1, 1)
print(f"Reshaped shape: {x_2d.shape}")
#%%

'''
Because scikit-learn is built to handle several features at once, X 
must be 2D. We use shape (n, 1) even when there is only one feature 
because a 2D array has shape (n_samples, n_features). Because of this 
uniform interface, the same code can be used with one feature or one 
hundred.
'''

# Q3
#%%
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("\n--- scikit-learn Q3 ---")
print("Cluster centers:")
print(kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure(figsize=(7, 5))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="tab10", s=40, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="black", marker="X", s=180, zorder=5, label="Centers")
plt.title("K-Means Clustering (k=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/kmeans_clusters.png", dpi=150)
plt.close()
print("Saved: outputs/kmeans_clusters.png")
#%%

# --- Linear Regression ---

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
print("\n--- Linear Regression Q1 ---")
plt.figure(figsize=(7, 5))
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.8, edgecolors="k")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")
plt.tight_layout()
plt.savefig("outputs/cost_vs_age.png", dpi=150)
plt.close()
print("Saved: outputs/cost_vs_age.png")

#%%
'''
It is evident that there are two different bands of points. Smokers,
whose expenses are much higher, are represented by the upper band 
(red/warm). Non-smokers are represented by the lower band (blue/cool).
This implies that being a smoker has a significant additive impact on
medical expenses, independent of age.
'''

# Q2
print("\n--- Linear Regression Q2 ---")
X_age = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(X_age, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
#%%

# Q3
print("\n--- Linear Regression Q3 ---")
model_age = LinearRegression()
model_age.fit(X_train, y_train)

print(f"Slope (age coef): {model_age.coef_[0]:.2f}")
print(f"Intercept:        {model_age.intercept_:.2f}")

y_pred = model_age.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2   = model_age.score(X_test, y_test)

print(f"RMSE:  {rmse:,.2f}")
print(f"R²:    {r2:.4f}")
#%%

'''
The slope (~200) indicates that medical expenses rise by roughly 
$200 annually on average for every extra year of age.
'''

# Q4
print("\n--- Linear Regression Q4 ---")
X_full = np.column_stack([age, smoker])

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

r2_full = model_full.score(X_test_f, y_test_f)
print(f"R² (age only):         {r2:.4f}")
print(f"R² (age + smoker):     {r2_full:.4f}")
print(f"age coefficient:    {model_full.coef_[0]:.2f}")
print(f"smoker coefficient: {model_full.coef_[1]:.2f}")
#%%

'''
Adding the smoker flag significantly increases R^2. The smoker 
coefficient (~15000) indicates that, when age is held constant, smoking
is linked to an additional $15,000 in annual medical expenses.
'''

# Q5
print("\n--- Linear Regression Q5 ---")
y_pred_full = model_full.predict(X_test_f)

plt.figure(figsize=(6, 6))
plt.scatter(y_pred_full, y_test_f, alpha=0.75, edgecolors="k", linewidths=0.4)
min_val = min(y_pred_full.min(), y_test_f.min())
max_val = max(y_pred_full.max(), y_test_f.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost ($)")
plt.ylabel("Actual Cost ($)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png", dpi=150)
plt.close()
print("Saved: outputs/predicted_vs_actual.png")
#%%

'''
A point above the diagonal indicates that the actual cost exceeded the 
forecast. The model was underestimated. A point below the diagonal 
indicates that the model overestimated and the real cost was LOWER than 
anticipated.
'''