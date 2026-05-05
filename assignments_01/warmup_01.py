# --- Pandas ---

# Pandas Q1
import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("Pandas Q1")
print(f"First 3 rows:\n{df.head(3)}", end="\n\n")
print(f"Num Rows: {len(df)}", end="\n\n")
print(f"Shape: {df.shape}", end="\n\n")
print(f"Col data types:\n{df.dtypes}", end="\n\n")

# Pandas Q2
print("\nPandas Q2")
print(df.query("passed == True and grade > 80"))

# Pandas Q3
print("\nPandas Q3")
df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4
print("\nPandas Q4")
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5
print("\nPandas Q5")
city_grouped_mean = df.groupby("city")["grade"].mean()
print(city_grouped_mean)

# Pandas Q6
print("\nPandas Q6")
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7
print("\nPandas Q7")
sorted_df = df.sort_values("grade", ascending=False)
print(sorted_df.head(3))


# --- NumPy ---
import numpy as np

# NumPy Q1
print("\nNumpy Q1")
arr = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Ndim: {arr.ndim}")

# NumPy Q2
print("\nNumpy Q2")
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Shape: {arr.shape}")
print(f"Size (total elements): {arr.size}")

# NumPy Q3
print("\nNumpy Q3")
top_left = arr[0:2, 0:2]
print(top_left)

# NumPy Q4
print("\nNumpy Q4")
zeros_3x4 = np.zeros((3, 4))
ones_2x5 = np.ones((2, 5))
print(f"3x4 zeros:\n{zeros_3x4}")
print(f"2x5 ones:\n{ones_2x5}")

# NumPy Q5
print("\nNumpy Q5")
arr = np.arange(0, 50, 5)
# Expected: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] — 10 values stepping by 5
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Mean: {np.mean(arr)}")
print(f"Sum: {np.sum(arr)}")
print(f"Std: {np.std(arr)}")

# NumPy Q6
print("\nNumpy Q6")
arr = np.random.normal(loc=0, scale=1, size=200)
print(f"Mean: {np.mean(arr)}")
print(f"Std:  {np.std(arr)}")


# --- Matplotlib ---
import matplotlib.pyplot as plt

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.figure(1)
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.figure(2)
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

# Matplotlib Q3
plt.figure(3)
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Matplotlib Q4
plt.figure(4)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

plt.show()


# --- Descriptive Statistics ---

# Descriptive Stats Q1
print("\nDescriptive Stats Q1")
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Std Dev: {np.std(data)}")

# Descriptive Stats Q2
print("\nDescriptive Stats Q2")
scores_500 = np.random.normal(65, 10, 500)
plt.figure(5)
plt.hist(scores_500, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3
print("\nDescriptive Stats Q3")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.figure(6)
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.show()

# Descriptive Stats Q4
print("\nDescriptive Stats Q4")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.figure(7)
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

# The exponential distribution is more skewed (positive skew).
# For the normal distribution, mean and median are essentially the same either works well.
# For the exponential distribution, the median is a better measure of central tendency
# because it is less influenced by the long right tail that pulls the mean upward.

# Descriptive Stats Q5
print("\nDescriptive Stats Q5")
from scipy import stats

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

for label, d in [("data1", data1), ("data2", data2)]:
    print(f"{label} -> Mean: {np.mean(d)}, Median: {np.median(d)}, Mode: {stats.mode(d, keepdims=True).mode[0]}")

# In data2, the outlier value 150 is far larger than the rest of the values.
# The mean gets pulled heavily toward that outlier, making it much higher than
# what a "typical" value in the dataset would suggest. The median only
# looks at the middle value and is not affected by how extreme the outlier is,
# so it stays close to 12, which better represents the center of the majority of values.


# # --- Hypothesis Testing ---

# Hypothesis Q1
print("\nHypothesis Q1")
from scipy.stats import ttest_ind

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")

# Hypothesis Q2
print("\nHypothesis Q2")
alpha = 0.05
if p_val < alpha:
    print("Result is statistically significant (p < 0.05) — we reject the null hypothesis.")
else:
    print("Result is NOT statistically significant (p >= 0.05) — we fail to reject the null hypothesis.")

# Hypothesis Q3
print("\nHypothesis Q3")
from scipy.stats import ttest_rel

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat_paired, p_val_paired = ttest_rel(before, after)
print(f"Paired t-statistic: {t_stat_paired}")
print(f"p-value: {p_val_paired}")

# Hypothesis Q4
print("\nHypothesis Q4")
from scipy.stats import ttest_1samp

scores_q4 = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat_1s, p_val_1s = ttest_1samp(scores_q4, popmean=70)
print(f"t-statistic: {t_stat_1s}")
print(f"p-value: {p_val_1s}")

# Hypothesis Q5
print("\nHypothesis Q5")
t_stat_1t, p_val_1t = ttest_ind(group_a, group_b, alternative='less')
print(f"One-tailed p-value (group_a < group_b): {p_val_1t}")

# Hypothesis Q6
print("\nHypothesis Q6")
mean_a = round(sum(group_a) / len(group_a), 2)
mean_b = round(sum(group_b) / len(group_b), 2)
print(
    f"Students in Group B scored meaningfully higher on average ({mean_b}) than students in Group A ({mean_a}). "
    f"This difference is unlikely to be due to chance (p = {p_val}), suggesting a real performance gap between the two groups."
)


# --- Correlation ---

# Correlation Q1
print("\nCorrelation Q1")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print(f"Full correlation matrix:\n{corr_matrix}")
print(f"Correlation coefficient: {corr_matrix[0, 1]}")
# Expected correlation is exactly 1.0 — x and y have a perfect positive linear relationship.
# Every value of y is exactly 2 * x, so as x increases by 1, y always increases by 2.
# There is zero deviation from the trend line, which gives a Pearson r of 1.

# Correlation Q2
print("\nCorrelation Q2")
from scipy.stats import pearsonr

x2 = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y2 = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
r, p = pearsonr(x2, y2)
print(f"Correlation coefficient: {r}")
print(f"p-value:                 {p}")

# Correlation Q3
print("\nCorrelation Q3")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print(df.corr())

# Correlation Q4
print("\nCorrelation Q4")
x4 = [10, 20, 30, 40, 50]
y4 = [90, 75, 60, 45, 30]
plt.figure()
plt.scatter(x4, y4)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5
print("\nCorrelation Q5")
import seaborn as sns

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# --- Pipelines ---

# Pipeline Q1
print("\nPipeline Q1")

arr_pipeline = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }

def data_pipeline(arr):
    dt = create_series(arr)
    dt = clean_data(dt)
    return summarize_data(dt)

result = data_pipeline(arr_pipeline)

print("Result's Keys: ")
for key, val in result.items():
    print(f"{key}: {val}")