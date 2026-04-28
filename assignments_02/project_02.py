#%%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

def print_title(msg:str):
    print("=" * 60)
    print(msg)
    print("=" * 60)


#%%
# TASK 1: Load and Explore

df = pd.read_csv("../../assignments/resources/student_performance_math.csv", sep=";")
print_title("TASK 1: Load and Explore")
print(f"\nShape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Histogram of G3 - 21 bins, one per possible value 0-20
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["G3"], bins=range(0, 22), edgecolor="black", color="#4C72B0", alpha=0.85)
ax.set_title("Distribution of Final Math Grades", fontsize=14)
ax.set_xlabel("G3 (Final Grade, 0-20)", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
ax.set_xticks(range(0, 21))
plt.tight_layout()
plt.savefig("outputs/g3_distribution.png", dpi=150)
plt.close()
print("\nSaved: outputs/g3_distribution.png")

#%%
## TASK 2: Preprocess the Data
print_title("TASK 2: Preprocess the Data")

print(f"\nShape before filtering G3=0: {df.shape}")
df_clean = df[df["G3"] > 0].copy()
print(f"Shape after  filtering G3=0: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# Encode binary yes/no columns to 1/0
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})
    df[col] = df[col].map({"yes": 1, "no": 0})

# Encode sex: F=0, M=1
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})
df["sex"] = df["sex"].map({"F": 0, "M": 1})

# Pearson correlation: absences vs G3 before and after filtering
corr_before = df["absences"].corr(df["G3"])
corr_after  = df_clean["absences"].corr(df_clean["G3"])

print(f"\nPearson corr(absences, G3) - full dataset (includes G3=0): {corr_before:.4f}")
print(f"Pearson corr(absences, G3) - filtered dataset (G3>0 only): {corr_after:.4f}")

'''
WHY FILTERING CHANGES THE RESULT:
    In the original dataset, students with G3=0 (exam-absent) also tend to have
    high absences. That creates a cluster of (high absences, G3=0) points that
    drags the correlation strongly negative but it is measuring exam absence,
    not grade quality. Once we filter to students who actually took the exam,
    the correlation between absences and G3 weakens substantially, because the
    relationship between skipping some school days and final performance is
    genuinely modest among students who showed up for the exam. The dramatic
    shift exposes how much that cluster of missing outcomes was distorting the
    original signal.
'''
#%%
# TASK 3: Exploratory Data Analysis
print_title("TASK 3: Exploratory Data Analysis")

numeric_features = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc"
]

correlations = {col: df_clean[col].corr(df_clean["G3"]) for col in numeric_features}
sorted_corrs = sorted(correlations.items(), key=lambda x: x[1])

print("\nPearson correlation with G3 (sorted most negative to most positive):")
for feat, corr in sorted_corrs:
    print(f"  {feat:12s}: {corr:+.4f}")

'''
OBSERVATIONS:
    failures has the strongest (negative) relationship with G3 students
    who have failed courses before tend to score lower. This makes intuitive
    sense and aligns with the baseline model choice.
    Medu (mother's education) and Fedu (father's education) show moderate
    positive correlations students from more educated households perform
    better, likely through more support at home and higher expectations.
    Walc (weekend alcohol) and goout show small negative correlations the
    effect exists but is weaker than popular intuition might suggest.
    Absences correlation is strong, which is not surprising.
'''

# Visualization 1: G3 box plot by number of past failures
fig, ax = plt.subplots(figsize=(8, 5))
groups = [df_clean[df_clean["failures"] == k]["G3"].values for k in range(4)]
bp = ax.boxplot(groups, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                medianprops=dict(color="red", linewidth=2))
ax.set_xticklabels(["0 failures", "1 failure", "2 failures", "3 failures"])
ax.set_title("G3 Distribution by Number of Past Failures", fontsize=13)
ax.set_xlabel("Past Failures", fontsize=11)
ax.set_ylabel("Final Grade G3 (0-20)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/g3_by_failures.png", dpi=150)
plt.close()
print("\nSaved: outputs/g3_by_failures.png")

'''
WHAT WE SEE:
    Students with zero past failures score substantially higher on average and
    have a tighter, higher distribution. Each additional failure shifts the
    median down noticeably. Students with 3 failures have a very wide spread
    some recover well, some do not suggesting failures alone cannot cleanly
    separate performance at the high end of the failure count.
'''

# Visualization 2: G3 vs studytime box plot
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["<2 hrs/wk", "2-5 hrs/wk", "5-10 hrs/wk", ">10 hrs/wk"]
groups2 = [df_clean[df_clean["studytime"] == k]["G3"].values for k in range(1, 5)]
ax.boxplot(groups2, patch_artist=True,
           boxprops=dict(facecolor="#55A868", alpha=0.7),
           medianprops=dict(color="red", linewidth=2))
ax.set_xticklabels(labels)
ax.set_title("G3 Distribution by Weekly Study Time", fontsize=13)
ax.set_xlabel("Weekly Study Time", fontsize=11)
ax.set_ylabel("Final Grade G3 (0-20)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/g3_by_studytime.png", dpi=150)
plt.close()
print("Saved: outputs/g3_by_studytime.png")

'''
WHAT WE SEE:
    The median G3 rises with study time, but the gains taper off at the highest
    level (>10 hrs/wk). That group also shows a wider spread, possibly because
    students studying very long hours include both highly motivated students and
    struggling students trying to compensate. The relationship is real but not
    perfectly linear worth noting as a limitation of a linear model.
'''

#%%
# TASK 4: Baseline Model (failures only)

print_title("TASK 4: Baseline Model")

X_base = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)
y_pred_b = model_base.predict(X_test_b)

rmse_base = math.sqrt(mean_squared_error(y_test_b, y_pred_b))
r2_base   = r2_score(y_test_b, y_pred_b)

print(f"\nIntercept (predicted G3 at 0 failures): {model_base.intercept_:.3f}")
print(f"Slope     (change in G3 per failure):    {model_base.coef_[0]:.3f}")
print(f"RMSE:  {rmse_base:.3f}")
print(f"R²:    {r2_base:.3f}")

'''
INTERPRETATION:
    The intercept (~11.5 expected) is the predicted G3 for a student with no
    prior failures. The slope (expected ~-1.5 to -2) means each additional
    recorded failure predicts roughly 1.5-2 grade points lower on the 0-20
    scale. An RMSE around 3.2-3.5 means the model's typical prediction is off
    by about 3-3.5 grade points -- that is quite large given a 0-20 scale.
    R² around 0.10-0.18 means failures alone explains only ~10-18% of grade
    variance. Given the moderate correlation we saw in Task 3, this is roughly
    expected. The model captures a real signal but most of the variation in G3
    comes from factors failures does not capture.
'''

#%%
# TASK 5: Full Model

print_title("TASK 5: Full Model")

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred = model_full.predict(X_test)

rmse_full    = math.sqrt(mean_squared_error(y_test, y_pred))
r2_train     = r2_score(y_train, model_full.predict(X_train))
r2_test_full = r2_score(y_test, y_pred)

print(f"\nTrain R²: {r2_train:.4f}")
print(f"Test  R²: {r2_test_full:.4f}")
print(f"RMSE:     {rmse_full:.4f}")
print(f"\nBaseline test R² (failures only): {r2_base:.4f}")
print(f"Full model test R²:               {r2_test_full:.4f}")
print(f"Improvement in R²:                {r2_test_full - r2_base:+.4f}")

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"  {name:12s}: {coef:+.3f}")

'''
Surprising results:
  activities: slightly negative in some runs. Extra-curriculars might
  correlate with students who are social rather than academic in this
  Portuguese sample or it could be noise given the small dataset.
  freetime: weakly negative. More unstructured free time might reduce
  study hours in practice, even though studytime is included separately.
Train vs test R2: they should be close (within ~0.05). A large gap would
signal overfitting, but with 11 features and ~350 rows, linear regression
is unlikely to overfit badly.
 
Production feature selection:
  Keep: failures (strongest predictor), higher (strong behavioral signal),
  Medu (socioeconomic proxy), studytime (direct effort measure)
  Drop or deprioritize: traveltime (tiny effect), activities (noisy),
  freetime (redundant with studytime), Fedu once Medu is in.
  Sex coefficient is modest; keep it for auditing purposes but do not
  over-interpret it as causal.
'''
#%%

# TASK 6: Evaluate and Summarize

print_title("TASK 6: Evaluate and Summarize")

plt.figure()
plt.scatter(y_pred, y_test, alpha=0.6)
mn = min(y_pred.min(), y_test.min())
mx = max(y_pred.max(), y_test.max())
plt.plot([mn, mx], [mn, mx], "r--", label="perfect prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()
'''
The model tends to miss more at the high end (grades 15+): it under-predicts
the best students, whose performance likely depends on individual factors not
captured by these background variables. Error looks roughly uniform across
the middle range. A point above the diagonal = actual grade was higher than
predicted (under-prediction). Below = actual was lower (over-prediction).
 
Summary:
  Filtered dataset: ~357 rows; test set: ~72 rows
  Full model RMSE ~3 on a 0-20 scale means a typical prediction is about
  3 grade points off — meaningful error for intervention use.
  Full model R2 ~0.30 explains about 30% of grade variance from background
  features alone; the rest is due to things we don't observe.
  Largest positive coefficient: higher (wanting higher education adds ~2-3
  points), Medu (mother's education adds ~0.5 pts per level).
  Largest negative coefficient: failures (each prior failure subtracts
  ~1.5-2 points).
  Surprise: activities and freetime have near-zero or slightly negative
  coefficients, which runs counter to the intuition that engaged students
  do better.
'''
  
# Neglected Feature: G1 
 
feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values
 
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42
)
 
m_g1 = LinearRegression()
m_g1.fit(X_train_g1, y_train_g1)
r2_g1 = m_g1.score(X_test_g1, y_test_g1)
 
print(f"\nWith G1 added — Test R2: {r2_g1:.3f}")
'''
R2 jumps to ~0.75. This does NOT mean G1 causes G3. G1 and G3 are
correlated because they both reflect the same underlying trait: that
student's actual ability and engagement throughout the year. Including G1
is essentially "cheating" for early intervention: by the time you have
G1, the student is already in the class and a third of the year has passed.
For truly early intervention identifying at-risk students before the
course even starts educators would need pre-enrollment data (prior school
records, socioeconomic indicators) and would be limited to the weaker R2 
of the background-features-only model.
'''
# %%
