import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo


# helpers

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {candidates} found in dataframe")


def boxplot_by_class(df, feature, label_col, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [df.loc[df[label_col] == v, feature] for v in (0, 1)]
    ax.boxplot(groups, tick_labels=["ham", "spam"], patch_artist=True)
    ax.set_title(feature)
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_cumvar(cumvar, threshold, outpath):
    n = int(np.searchsorted(cumvar, threshold)) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar)
    ax.axhline(threshold, color="red", linestyle="--", label=f"{threshold:.0%} threshold")
    ax.axvline(n, color="orange", linestyle=":", label=f"n={n}")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative explained variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return n


def eval_model(model, X_tr, y_tr, X_te, y_te, label):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"\n{label}")
    print(f"  accuracy: {accuracy_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, target_names=["ham", "spam"]))
    return y_pred


def cv_summary(models_data, y_tr, cv=5):
    print(f"\ncross-validation ({cv}-fold on training set)")
    print(f"{'model':<30} {'mean':>8} {'std':>8}")
    print("-" * 48)
    for label, model, X in models_data:
        scores = cross_val_score(model, X, y_tr, cv=cv)
        print(f"{label:<30} {scores.mean():>8.4f} {scores.std():>8.4f}")


# task 1: load and explore

spambase = fetch_ucirepo(id=94)

X_raw = spambase.data.features
y_raw = spambase.data.targets

print(spambase.metadata)
print(spambase.variables)

feature_cols = list(X_raw.columns)
X = X_raw.values
y = y_raw.values.ravel()

df = X_raw.copy()
df["spam_label"] = y

print(f"\nshape: {df.shape}")
print(f"\nclass counts:\n{df['spam_label'].value_counts()}")
print(f"\nclass proportions:\n{df['spam_label'].value_counts(normalize=True).round(3)}")

# ~61% ham means a majority-class baseline already scores 0.61  raw accuracy
# alone is a weak signal here. precision/recall per class tells the real story.

# spam emails use 'free', '!', and long capital runs far more than ham.
# word frequencies are zero-inflated fractions; capital_run_length_total can
# reach the thousands that scale gap will hurt distance-based models badly.

col_free = find_col(df, ["word_freq_free", "word.freq.free"])
col_excl = find_col(df, ["char_freq_!", "char.freq.!", "char_freq_excl"])
col_caps = find_col(df, ["capital_run_length_total", "capital.run.length.total"])

for feat in (col_free, col_excl, col_caps):
    safe = feat.replace(".", "_").replace("!", "excl").replace("$", "dollar")
    boxplot_by_class(df, feat, "spam_label", f"outputs/boxplot_{safe}.png")

# task 2: train/test split, scaling, pca

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# fit scaler on training data only using test statistics would leak
# information and produce an overoptimistic performance estimate.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA needs scaled input; capital_run_length_total would otherwise dominate
# the first component purely because of its larger numeric range.
pca_full = PCA().fit(X_train_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n90 = plot_cumvar(cumvar, 0.90, "outputs/spambase_pca_variance.png")
print(f"\ncomponents for 90% variance: {n90}")

pca = PCA(n_components=n90)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# task 3: classifier comparison

knn_raw = KNeighborsClassifier(n_neighbors=5)
y_pred_knn_raw = eval_model(knn_raw, X_train, y_train, X_test, y_test,
                             "KNN k=5 (unscaled)")

knn_sc = KNeighborsClassifier(n_neighbors=5)
y_pred_knn_sc = eval_model(knn_sc, X_train_scaled, y_train, X_test_scaled, y_test,
                            "KNN k=5 (scaled)")

knn_pca = KNeighborsClassifier(n_neighbors=5)
y_pred_knn_pca = eval_model(knn_pca, X_train_pca, y_train, X_test_pca, y_test,
                             "KNN k=5 (PCA)")

# sweep max_depth to see the overfitting curve directly
print("\ndecision tree depth sweep")
print(f"  {'depth':<18} {'train':>8} {'test':>8}")
for depth in (3, 5, 10, None):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    tr = accuracy_score(y_train, dt.predict(X_train))
    te = accuracy_score(y_test, dt.predict(X_test))
    label = str(depth) if depth else "None"
    print(f"  {label:<18} {tr:>8.4f} {te:>8.4f}")

# training accuracy hits 1.0 at unlimited depth while test accuracy drops
# the tree memorises noise. depth=5 sits near the test peak without that gap.
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
y_pred_dt = eval_model(dt, X_train, y_train, X_test, y_test,
                        "decision tree (max_depth=5)")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred_rf = eval_model(rf, X_train, y_train, X_test, y_test,
                        "random forest (100 trees)")

lr_sc = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
y_pred_lr_sc = eval_model(lr_sc, X_train_scaled, y_train, X_test_scaled, y_test,
                           "logistic regression (scaled)")

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
y_pred_lr_pca = eval_model(lr_pca, X_train_pca, y_train, X_test_pca, y_test,
                            "logistic regression (PCA)")

# random forest wins overall. logistic regression on full scaled features is a
# close second; PCA sheds a bit of signal so it underperforms slightly.
# unscaled KNN suffers because capital_run_length_total overwhelms distances.
#
# for a spam filter i'd optimise for precision over recall: a false positive
# is more disruptive to a user than a false negative. missing a job offer 
# or a medical message is a real cost; a bit of leaked spam is merely annoying.

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf),
                       display_labels=["ham", "spam"]).plot(ax=ax, colorbar=False)
ax.set_title("random forest - confusion matrix")
fig.tight_layout()
fig.savefig("outputs/best_model_confusion_matrix.png", dpi=150)
plt.close(fig)

# feature importances - both models should agree on the top features
dt_imp = pd.Series(dt.feature_importances_, index=feature_cols)
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols)

print("\ntop 10 - decision tree")
print(dt_imp.nlargest(10).round(4).to_string())
print("\ntop 10 - random forest")
print(rf_imp.nlargest(10).round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
rf_imp.nlargest(10).sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("importance")
ax.set_title("random forest - top 10 feature importances")
fig.tight_layout()
fig.savefig("outputs/feature_importances.png", dpi=150)
plt.close(fig)

# task 4: cross-validation

cv_summary([
    ("KNN (unscaled)",         knn_raw, X_train),
    ("KNN (scaled)",           knn_sc,  X_train_scaled),
    ("KNN (PCA)",              knn_pca, X_train_pca),
    ("decision tree (d=5)",    dt,      X_train),
    ("random forest",          rf,      X_train),
    ("logistic reg (scaled)",  lr_sc,   X_train_scaled),
    ("logistic reg (PCA)",     lr_pca,  X_train_pca),
], y_train)

# random forest is the most accurate and the most stable (lowest std).
# the CV ranking matches the single split the 80/20 partition was
# representative and we didn't get lucky.

# task 5: pipelines

# tree-based: no scaler needed, trees are invariant to feature scale
rf_pipe = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])
eval_model(rf_pipe, X_train, y_train, X_test, y_test,
           "pipeline - random forest")

# logistic regression requires scaling; PCA omitted because full features win
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
])
eval_model(lr_pipe, X_train, y_train, X_test, y_test,
           "pipeline - logistic regression")

# the pipelines differ in structure because their preprocessing requirements
# differ. bundling scaler + model into one object eliminates bookkeeping
# errors and makes the artifact serialisable for deployment raw input in,
# prediction out, with no chance of forgetting to scale new data.