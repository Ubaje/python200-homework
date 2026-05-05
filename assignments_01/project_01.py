import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from prefect import task, flow, get_run_logger

DATA_DIR   = "../../assignments/resources/happiness_project"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = list(range(2015, 2025))

NUMERIC_COLS = [
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
]

# Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    logger = get_run_logger()
    data_frames = []

    for year in YEARS:
        path = os.path.join(DATA_DIR, f"world_happiness_{year}.csv")
        df = pd.read_csv(path, sep=";", decimal=",")
        df["year"] = year
        df.rename(columns={"Ladder score": "Happiness score"}, inplace=True)
        df.columns = df.columns.str.strip()
        data_frames.append(df)
        logger.info(f"Loaded {year}: {len(df)} rows")

    merged = pd.concat(data_frames, ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "merged_happiness.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"Merged dataset saved to {out_path} — {len(merged)} total rows")

    return merged

# Task 2: Descriptive Statistics
@task
def descriptive_stats(df):
    logger = get_run_logger()

    mean = float(df["Happiness score"].mean())
    median = float(df["Happiness score"].median())
    std = float(df["Happiness score"].std())

    logger.info(f"Overall happiness score — mean: {mean:.3f}, median: {median:.3f}, std: {std:.3f}")

    logger.info("Mean happiness score by year:")
    by_year = df.groupby("year")["Happiness score"].mean()
    for year, val in by_year.items():
        logger.info(f"  {year}: {val:.3f}")

    logger.info("Mean happiness score by region:")
    by_region = df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    for region, val in by_region.items():
        logger.info(f"  {region}: {val:.3f}")

    return by_region


# Task 3: Visual Exploration
@task
def visual_exploration(df):
    logger = get_run_logger()

    # Histogram of all happiness scores
    plt.figure()
    plt.hist(df["Happiness score"].dropna(), bins=30)
    plt.title("Distribution of Happiness Scores (2015-2024)")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "happiness_histogram.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")

    # Boxplot by year
    years_sorted = sorted(df["year"].unique())
    data_by_year = [df[df["year"] == y]["Happiness score"].dropna().values for y in years_sorted]
    plt.figure(figsize=(14, 6))
    plt.boxplot(data_by_year, labels=years_sorted)
    plt.title("Happiness Score Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "happiness_by_year.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")

    # Scatter: GDP per capita vs happiness score
    plt.figure()
    plt.scatter(df["GDP per capita"], df["Happiness score"], alpha=0.3, s=10)
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")

    # Correlation heatmap
    numeric_df = df[["Happiness score"] + NUMERIC_COLS].dropna()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")


# Task 4: Hypothesis Testing
@task
def hypothesis_testing(df):
    logger = get_run_logger()

    # Test 1: Did happiness change between 2019 (pre-pandemic) and 2020 (pandemic onset)?
    scores_2019 = df[df["year"] == 2019]["Happiness score"].dropna()
    scores_2020 = df[df["year"] == 2020]["Happiness score"].dropna()

    t_stat, p_val = ttest_ind(scores_2019, scores_2020)
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info("--- Test 1: 2019 vs 2020 (pre vs pandemic onset) ---")
    logger.info(f"Mean happiness 2019: {mean_2019:.3f}")
    logger.info(f"Mean happiness 2020: {mean_2020:.3f}")
    logger.info(f"t-statistic: {t_stat:.4f}")
    logger.info(f"p-value: {p_val:.4f}")

    if p_val < 0.05:
        direction = "lower" if mean_2020 < mean_2019 else "higher"
        logger.info(
            f"Interpretation: The difference is statistically significant (p < 0.05). "
            f"Global happiness scores were {direction} in 2020 than in 2019, "
            f"suggesting the onset of the pandemic was associated with a measurable shift in well-being."
        )
    else:
        logger.info(
            f"Interpretation: The difference is not statistically significant (p = {p_val:.4f}). "
            f"We cannot conclude that happiness changed meaningfully between 2019 and 2020 based on this data."
        )

    # Test 2: Western Europe vs Sub-Saharan Africa — two regions that visually appear to sit at opposite ends of the regional happiness ranking
    region_a = "Western Europe"
    region_b = "Sub-Saharan Africa"
    scores_a = df[df["Regional indicator"] == region_a]["Happiness score"].dropna()
    scores_b = df[df["Regional indicator"] == region_b]["Happiness score"].dropna()

    t2, p2 = ttest_ind(scores_a, scores_b)
    logger.info(f"--- Test 2: {region_a} vs {region_b} ---")
    logger.info(f"Mean {region_a}: {scores_a.mean():.3f}")
    logger.info(f"Mean {region_b}: {scores_b.mean():.3f}")
    logger.info(f"t-statistic: {t2:.4f}, p-value: {p2:.4f}")

    if p2 < 0.05:
        logger.info(
            f"Interpretation: The happiness gap between {region_a} and {region_b} is "
            f"statistically significant (p < 0.05). This is a large and consistent difference "
            f"that cannot be attributed to random sampling variation."
        )
    else:
        logger.info("Interpretation: The difference is not statistically significant.")

    return p_val


# Task 5: Correlation and Multiple Comparisons
@task
def correlation_analysis(df):
    logger = get_run_logger()

    numeric_df = df[["Happiness score"] + NUMERIC_COLS].dropna()
    n_tests = len(NUMERIC_COLS)
    adjusted_alpha = 0.05 / n_tests

    logger.info(f"Running {n_tests} correlation tests — adjusted alpha (Bonferroni): {adjusted_alpha:.4f}")

    results = {}
    for col in NUMERIC_COLS:
        r, p = pearsonr(numeric_df[col], numeric_df["Happiness score"])
        sig_original  = p < 0.05
        sig_corrected = p < adjusted_alpha
        results[col] = {"r": r, "p": p, "sig_original": sig_original, "sig_corrected": sig_corrected}
        logger.info(
            f"{col}: r={r:.4f}, p={p:.4f} | "
            f"sig at 0.05: {sig_original} | sig after Bonferroni: {sig_corrected}"
        )

    strongest = max(results, key=lambda c: abs(results[c]["r"]))
    logger.info(f"Strongest correlation with happiness: '{strongest}' (r={results[strongest]['r']:.4f})")

    return results


# Task 6: Summary Report
@task
def summary_report(df, by_region, corr_results, p_val_2019_2020):
    logger = get_run_logger()

    n_countries = df["Country"].nunique()
    n_years = df["year"].nunique()
    logger.info(f"Total unique countries: {n_countries} | Years covered: {n_years} (2015-2024)")

    top3 = by_region.head(3)
    bottom3 = by_region.tail(3)
    logger.info("Top 3 regions by mean happiness score:")
    for region, val in top3.items():
        logger.info(f"  {region}: {val:.3f}")
    logger.info("Bottom 3 regions by mean happiness score:")
    for region, val in bottom3.items():
        logger.info(f"  {region}: {val:.3f}")

    if p_val_2019_2020 < 0.05:
        logger.info(
            "Pre/post-2020 test: Global happiness scores shifted significantly between 2019 and 2020 "
            "(p < 0.05), consistent with a pandemic-related impact on well-being."
        )
    else:
        logger.info(
            "Pre/post-2020 test: No statistically significant difference detected between 2019 and 2020 "
            f"(p = {p_val_2019_2020:.4f})."
        )

    # Bonferroni correction
    corrected = {c: v for c, v in corr_results.items() if v["sig_corrected"]}
    if corrected:
        strongest = max(corrected, key=lambda c: abs(corrected[c]["r"]))
        logger.info(
            f"Strongest predictor of happiness (Bonferroni-corrected): '{strongest}' "
            f"with r={corrected[strongest]['r']:.4f}"
        )
    else:
        logger.info("No correlations remained significant after Bonferroni correction.")

@flow
def happiness_pipeline():
    df = load_data()
    by_region = descriptive_stats(df)
    visual_exploration(df)
    p_val = hypothesis_testing(df)
    corr = correlation_analysis(df)
    summary_report(df, by_region, corr, p_val)


if __name__ == "__main__":
    happiness_pipeline()