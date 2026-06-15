import os
import glob
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

DATA_DIR = "../../assignments/resources/happiness_project"
OUTPUTS_DIR = "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Module-level DataFrame used by all tools. Populated before the agent runs.
# The CodeAgent sandbox cannot see this directly, so df is also passed via
# additional_args so the agent can use it in any code it writes.
df = None


def _load_dataframe() -> pd.DataFrame:
    """Load and merge all yearly CSVs into a single cleaned DataFrame."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    yearly_frames = []
    for path in sorted(csv_files):
        frame = pd.read_csv(path, sep=";", decimal=",")
        basename = os.path.basename(path).replace(".csv", "")
        frame["year"] = int(basename[-4:])
        frame.columns = frame.columns.str.strip()
        if "Ladder score" in frame.columns:
            frame.rename(columns={"Ladder score": "Happiness score"}, inplace=True)
        yearly_frames.append(frame)
    merged = pd.concat(yearly_frames, ignore_index=True)
    merged.columns = [c.lower().strip().replace(" ", "_") for c in merged.columns]
    return merged


# --- Task 1: Tool Definitions ---

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory by merging all yearly CSVs.

    Loads and concatenates all CSV files found in the happiness_project directory.
    Each file represents one year of data.

    Returns:
        A dict with keys 'shape' (tuple) and 'columns' (list of str), or an 'error' key.
        Read values directly from the dict, e.g. result['shape']. Do not call .shape
        or any DataFrame method on this return value.
    """
    global df
    df = _load_dataframe()
    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The snake_case column name to summarize (e.g. 'happiness_score').

    Returns:
        A dict of descriptive statistics from pandas describe(), or an 'error' key
        if the data is not loaded or the column is not found.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses scipy.stats.pearsonr. Rows with NaN in either column are dropped before computing.

    Args:
        col1: Name of the first numeric column (snake_case).
        col2: Name of the second numeric column (snake_case).

    Returns:
        A dict with keys 'col1', 'col2', 'pearson_r', and 'p_value' (rounded to 4 decimal
        places), or an 'error' key on bad input.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if col1 not in df.columns:
        return {"error": f"Column '{col1}' not found."}
    if col2 not in df.columns:
        return {"error": f"Column '{col2}' not found."}
    try:
        valid = df[[col1, col2]].dropna()
        r, p = stats.pearsonr(valid[col1], valid[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the dataset to the given year, sorts by the column descending, and returns
    the top N rows.

    Args:
        column: The column to rank countries by (snake_case, must be numeric).
        year: The year to filter on (e.g., 2019, 2020).
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dict with key 'results' containing a list of dicts, each with 'country' and
        the column value. Returns an 'error' key on bad input.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if "year" not in df.columns:
        return {"error": "No 'year' column found in the dataset."}
    if "country" not in df.columns:
        return {"error": "No 'country' column found in the dataset."}

    year_df = df[df["year"] == year]
    if year_df.empty:
        available = sorted(df["year"].dropna().unique().tolist())
        return {"error": f"No data for year {year}. Available years: {available}"}

    top = year_df.sort_values(column, ascending=False).head(n)
    results = [
        {"country": row["country"], column: row[column]}
        for _, row in top.iterrows()
    ]
    return {"results": results}


# --- Task 2: Build the Agent ---

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

Important rules:
- All tools return plain dicts. Do not call .shape, .columns, or any DataFrame method
  on a tool return value. Read keys directly, e.g. result['shape'].
- The full DataFrame is available as the variable `df` in your code environment.
  Use it directly when writing code (e.g. df['happiness_score'], df.groupby(...)).
- When writing any matplotlib code, always call matplotlib.use("Agg") before importing
  matplotlib.pyplot to avoid GUI threading errors.
- Column names are snake_case: happiness_score, gdp_per_capita, regional_indicator,
  country, year, social_support, healthy_life_expectancy, freedom_to_make_life_choices,
  generosity, perceptions_of_corruption.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# --- Task 3: Run Guided Queries ---

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    (
        "Plot happiness_score over the years as a line chart, with one line per region. "
        f"Use the regional_indicator column for regions. "
        f"Save the plot to {OUTPUTS_DIR}/happiness_by_region.png."
    ),
]


def run_guided_queries(df):
    print("=" * 60)
    print("Task 3: Guided Queries")
    print("=" * 60)

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    plot_path = f"{OUTPUTS_DIR}/happiness_by_region.png"
    if os.path.exists(plot_path):
        print(f"\nPlot confirmed saved at: {plot_path}")
    else:
        print(f"\nWarning: plot not found at {plot_path}")


# --- Task 4: Custom Queries ---

def run_custom_queries(df):
    print("\n" + "=" * 60)
    print("Task 4: Custom Queries")
    print("=" * 60)

    # My query 1: requires code generation since no tool computes year-over-year deltas.
    my_query_1 = (
        "Which 5 countries improved their happiness_score the most between 2015 and 2020? "
        "Show the score in each year and the total change."
    )
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print(response_1)
    # Comment: This triggers code generation. No tool computes year-over-year score
    # differences, so the agent writes pandas code to filter by year, pivot or merge,
    # compute the delta, and sort. The tools alone are not sufficient here.

    # My query 2: requires code generation since no tool produces a scatter plot.
    my_query_2 = (
        f"Plot a scatter plot of social_support vs happiness_score. "
        f"Save it to {OUTPUTS_DIR}/social_support_vs_happiness.png."
    )
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)
    # Comment: This triggers code generation. None of the four tools produce plots,
    # so the agent writes matplotlib code directly to create and save the scatter plot.


if __name__ == "__main__":
    df = _load_dataframe()
    run_guided_queries(df)
    run_custom_queries(df)


# --- Task 5: Reflection ---

# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?

#    The agent called compute_correlation and got back pearson_r=0.6313 and p_value=0.0.
#    It returned a final_answer dict with "statistical_significance": "significant",
#    implicitly applying the standard 0.05 threshold. The p_value of 0.0 (rounded to
#    4 decimal places) is far below any reasonable threshold, so the classification is
#    correct. The agent did not explicitly state the 0.05 threshold in its response, but
#    its conclusion was right. The pearson_r of 0.63 indicates a moderate positive
#    correlation, meaning higher GDP per capita is associated with higher happiness scores.

# 2. Did any of the agent's responses surprise you, either by being more capable than
#    you expected, or less? Describe one specific example.

#    Query 1 was more capable than expected. When asked to load data and report shape and
#    column names, the agent hit an AttributeError on the dict return value, then instead
#    of just failing it manually reconstructed the shape from the dict keys and produced
#    the correct answer anyway. That kind of self-correction in a single run without
#    human intervention was impressive for gpt-4o-mini.

#    Less impressive: on the same Query 1, after the error the agent generated 30+ lines
#    of code to fabricate a mock DataFrame from scratch before realizing it could just
#    read the dict keys directly. It reached the right answer through an unnecessarily
#    complicated path. The agent solved the problem but wasted two extra steps and tokens
#    on an approach that had nothing to do with the actual data.

# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help answer.

#    A compare_countries tool that takes a list of country names and a column name, then
#    returns each country's value for every year as a structured table. Right now, a
#    question like "how did Finland and Denmark compare on happiness_score over the years"
#    requires the agent to write pandas filtering and merge code, as seen in Custom Query 1.
#    That code worked, but it required two steps and could break silently on a typo in a
#    column name or a country that appears in one year but not another. A dedicated tool
#    would handle those edge cases reliably and return a clean structured result in a
#    single call. Country-level time series comparisons are a natural and frequent question
#    for this dataset.