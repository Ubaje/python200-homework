import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from scipy import stats
import pandas as pd

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')


# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in Celsius to convert."
                }
            },
            "required": ["celsius"]
        }
    }
}

print("# Q1 - celsius_to_fahrenheit direct calls")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2
from datetime import datetime

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_current_time()


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current UTC date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


tools = [get_current_time_schema]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time."""

    # Step 1: start the conversation with system and user messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Error: unknown tool {function_name}."

            # Print for debugging so we can see what happened
            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ""


# Q2 - Prediction before calling:

# Will run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
# No. The only available tool is get_current_time, which is irrelevant to temperature
# conversion. The model has no reason to call it, so it will answer from its own
# knowledge and skip the tool step entirely saying "No tools needed....".

# How many API calls will be made?
# Exactly 1. This version of run_agent always makes a first API call. If no tool is
# called, it returns first_message.content immediately without making a second call.
# The second API call only happens when a tool is actually invoked (the if branch).

print("\n# Q2 - run_agent with only get_current_time tool")
result_q2 = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Result:", result_q2)

# My prediction was correct. The output printed "No tools needed...." confirming
# no tool call was made, and only one API call was issued. The model answered
# from memory using the formula directly.



# Q3 - Extend run_agent to support both get_current_time and celsius_to_fahrenheit.
# The tools list is updated to include the celsius_to_fahrenheit schema, and the
# dispatcher inside run_agent_extended handles both tool names.

extended_tools = [get_current_time_schema, celsius_to_fahrenheit_schema]


def run_agent_extended(user_prompt: str) -> str:
    """
    Same fixed 2-call structure as run_agent from the lesson, but with two tools.
    Dispatches get_current_time and celsius_to_fahrenheit by name.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that can tell the current time
                     and convert temperatures from Celsius to Fahrenheit."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=extended_tools,
        tool_choice="auto",
    )

    first_message = first_response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            if function_name == "get_current_time":
                tool_result = get_current_time()
            elif function_name == "celsius_to_fahrenheit":
                tool_result = celsius_to_fahrenheit(**args)
            else:
                tool_result = f"Error: unknown tool {function_name}."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        return second_response.choices[0].message.content or ""
    else:
        return first_message.content or ""


print("\n# Q3 - Extended agent with both tools")

response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Tool called: YES. The model recognizes this is a temperature conversion task and
# calls celsius_to_fahrenheit with celsius=37. It needs the tool to produce the answer.

response_b = run_agent_extended("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Tool called: NO. The boiling point of water (100C / 212F) is general
# knowledge. The model doesn't need to call a tool to answer "in plain English."
# It may or may not call celsius_to_fahrenheit depending on how it interprets the
# prompt, but the question doesn't demand a numeric conversion, so it usually answers
# from its training data directly.


# --- Lesson 03 ---
 
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
 
RESOURCES_DIR = Path("../../lessons/07_AI_agents/resources")
#RESOURCES_DIR = Path("resources")
 
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None
 
    # --- Small internal helpers --------------------------------------
 
    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename
 
    def _available_csv_files(self) -> list:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )
 
    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None
 
    # --- Tools (public methods) --------------------------------------
 
    def list_csv_files(self) -> dict:
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}
 
    def load_csv(self, filename: str) -> dict:
        """
        Load a CSV file from resources/ and make it the active dataset.
 
        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
 
        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }
 
        self.df = pd.read_csv(path)
        self.csv_name = filename
 
        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }
 
    def get_columns(self) -> list:
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()
 
    def summarize_columns(self, columns: list = None) -> dict:
        """
        Return basic summary stats for one or more columns.
 
        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error
 
        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]
 
        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()
 
    def describe_column(self, column: str) -> dict:
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error
 
        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}
 
        s = self.df[column]
        summary = s.describe().to_dict()
 
        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value
 
        return cleaned
 
    def plot_data(self, y: str, x: str = None, plot_type: str = "line") -> str:
        """
        Plot from the active CSV.
 
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
 
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
 
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
 
        if x == y:
            x = None
 
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
 
        title_csv = self.csv_name or "current CSV"
 
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
 
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
 
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
 
        return f"Plotted {y} vs {x} as a {plot_type}."
 
    # The lesson hit a tool-round limit here because no tool existed for correlation.
    # Adding it as a method keeps all CSV operations in one place and follows the
    # same pattern the lesson uses for describe_column.
 
    def compute_correlation(self, col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
 
        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found. Options: {self.df.columns.tolist()}"}
        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found. Options: {self.df.columns.tolist()}"}
 
        try:
            r, p = stats.pearsonr(
                self.df[col1].dropna(),
                self.df[col2].dropna()
            )
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4)
            }
        except Exception as e:
            return {"error": str(e)}
 
 
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)
 
#%%


# Tool schemas for the agent - one entry per public CsvManager method.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": (
                "Load a CSV file from resources/ and make it the active dataset. "
                "Pass just the filename, e.g. 'bike_commute' or 'bike_commute.csv'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the CSV file to load from resources/."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return the column names of the currently loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Return descriptive statistics for a single column in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column name to describe."
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": (
                "Return summary statistics for one or more columns. "
                "Pass a list of column names, or omit to summarize all columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of column names to summarize. Omit to summarize all."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": (
                "Plot data from the loaded CSV. Supports 'line' and 'scatter' plot types. "
                "x is optional; if omitted, plots y vs row index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": "The column to plot on the y-axis."
                    },
                    "x": {
                        "type": "string",
                        "description": "The column to plot on the x-axis. Optional."
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter"],
                        "description": "Type of plot. Either 'line' or 'scatter'."
                    }
                },
                "required": ["y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation coefficient and p-value between "
                "two numeric columns in the loaded CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Name of the first column."
                    },
                    "col2": {
                        "type": "string",
                        "description": "Name of the second column."
                    }
                },
                "required": ["col1", "col2"]
            }
        }
    }
]
 
# Dispatch table mapping tool names to CsvManager methods.
node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "describe_column": csv_manager.describe_column,
    "summarize_columns": csv_manager.summarize_columns,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}
 
SYSTEM_PROMPT = """You are a helpful CSV data analysis assistant.
You have access to tools that let you load CSV files, inspect their columns,
summarize statistics, filter rows, and create plots.
Always use the available tools to answer questions about the data.
Never guess or make up data values.
"""
 
 
def run_agent():
    """
    Simple command-line chat loop so it feels like a chatbot.
    We keep a single 'messages' list for the whole session so the model
    sees the conversation history each turn.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]
    print("CSV data agent at your service. Here to help look at your CSV data!")
    print("Type a question. Type 'exit' to quit.\n")
    print("To start, try 'list csv files' or 'load bike_commute.csv'\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break
 
        print(f"User query: {user_text}")
        assistant_text = run_agent_cycle(messages, user_text)
        print(f"\nAssistant: {assistant_text}\n")
 
#%%
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.
    The loop has three main steps:
    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.
    ACT:
      - If tools are requested, run the Python functions
    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.
    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})
 
    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        return tool_message
 
    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )
        msg = response.choices[0].message
 
        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)
 
        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content
 
        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            print(f"ACT: {name}({tool_args})")
            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
 
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
 
            # After appending information about all tool outputs, we loop back and REASON again.
 
    return "I hit the tool-round limit. Try a simpler request."
 
 
# Q5 - Recreate the lesson scenario that used to hit the tool round limit.
# Now that compute_correlation is defined, the agent should complete successfully.
 
print("\n# Q5 - run_agent_cycle with compute_correlation")
 
# load_csv takes just the filename; CsvManager resolves the path against resources_dir.
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
try:
    result = run_agent_cycle(
        messages,
        "Load bike_commute.csv and compute the correlation between "
        "avg_traffic_density and avg_speed_kmh."
    )
    print(result)
except Exception as e:
    print(f"Could not run Q5: {e}")
 
 
# Q6 - Print the full messages list to show the ReAct loop in action.
#
# Role breakdown:
#   "system"- The initial instruction that sets the agent's behavior and persona.
#       This is the "R" setup in ReAct, establishing what the agent should do.
#   "user"- The human's question or task. This is the trigger for the ReAct cycle.
#   "assistant" - The model's response. If it contains tool_calls, this is the "Act"
#       step (deciding which tool to invoke and with what arguments).
#       If it contains plain text, this is the final "Respond" step.
#   "tool"- The result returned by executing the tool. This is the "Observe" step.
#       The model reads this result and decides what to do next.
 
print("\n# Q6 - Full messages list (ReAct loop trace)")
print(json.dumps(messages, indent=2, default=str))
 
 
# --- Lesson 04: smolagents ---
 
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool
 
# Wrap each CsvManager method as a standalone smolagents tool, following the exact
# same pattern used in the lesson. The @tool decorator reads the type annotations
# and Google-style docstring to build the tool description automatically.
 
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.
 
    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()
 
 
@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.
 
    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".
 
    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)
 
 
@tool
def get_columns() -> dict:
    """Return column names for the currently loaded CSV.
 
    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()
 
 
@tool
def summarize_columns(columns: list = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.
 
    Args:
        columns: Column names to summarize. If None, summarizes all columns.
 
    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)
 
 
@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.
 
    Args:
        column: The name of the column to describe.
 
    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)
 
 
@tool
def plot_data(y: str, x: str = None, plot_type: str = "line") -> str:
    """Plot from the active CSV.
 
    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.
 
    Returns:
        A short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)
 
 
# Q7 - Re-wrap compute_correlation as a smolagents tool using @tool decorator.
 
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns in the loaded CSV.
 
    Args:
        col1: Name of the first column.
        col2: Name of the second column.
 
    Returns:
        A dict with keys col1, col2, pearson_r, and p_value, or an error key if
        the columns are not found or no data is loaded.
    """
    return csv_manager.compute_correlation(col1, col2)
 
 
print("\n# Q7 - smolagents auto-generated tool description")
print(compute_correlation.description)
 
# Comment comparing smolagents' auto-generated description to the manual JSON schema:
#
# The manual JSON schema in Q4 required writing out a nested dict with "type",
# "function", "name", "description", and a "parameters" object containing each
# argument's type and description. One typo in a key name breaks everything.
#
# smolagents reads the function's type annotations and Google-style docstring to
# produce the description automatically. The Args section maps directly to parameter
# descriptions, and the return type annotation sets the output type.
#
# What smolagents needs from the developer to produce a good description:
#   1. Accurate type annotations on every argument and the return value.
#   2. A clear first-line docstring explaining what the tool does.
#   3. A proper Args section describing each parameter in plain English.
#
# If any of those are vague or missing, the generated description is weaker and the
# agent is more likely to misuse or skip the tool entirely.
 
 
# Q8 - ToolCallingAgent vs CodeAgent on a scatter plot prompt.
# Set up both agents following the lesson pattern exactly.

smol_model = OpenAIServerModel(
    api_key=os.environ["OPENAI_API_KEY"],
    model_id="gpt-4o",
)
 
SMOL_SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)
 
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.
 
You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.
 
Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color).
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""
 
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]
 
tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=smol_model,
    instructions=SMOL_SYSTEM_PROMPT,
)
 
code_agent = CodeAgent(
    tools=TOOLS,
    model=smol_model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)
 
prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
 
print("\n# Q8 - ToolCallingAgent vs CodeAgent on scatter plot prompt")
 
try:
    response_tool = tool_agent.run(prompt)
    print("ToolCallingAgent response:", response_tool)
except Exception as e:
    print(f"ToolCallingAgent could not complete: {e}")
 
try:
    response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
    print("CodeAgent response:", response_code)
except Exception as e:
    print(f"CodeAgent could not complete: {e}")

# ToolCallingAgent:
#   It called load_csv (step 1), then called plot_data with the correct x and y columns
#   (step 2). The scatter plot was produced, but with the default blue dot color since
#   plot_data has no color parameter. In step 3 it called final_answer and acknowledged
#   it could not set the color. This is more honest than pure hallucination, but the
#   task still wasn't fully completed because the tool doesn't support color styling.
#   The ToolCallingAgent is bounded strictly by the tools it has.

# CodeAgent:
#   It wrote matplotlib code in step 1 with color='green', loaded the CSV via
#   load_csv(), and accessed csv_manager.df directly. The plot was produced correctly
#   with green dots. However, after the successful step 1, the agent tried to respond
#   in plain text instead of calling final_answer, triggering repeated code-parsing
#   errors ("regex pattern not found") for steps 2-8. It burned all 8 steps on
#   formatting errors after the actual work was already done. The core task succeeded,
#   but the agent wasted tokens and steps because it couldn't cleanly wrap up.

# Did the ToolCallingAgent change the dot color? No. It used plot_data which has no
# color argument, then admitted it couldn't apply the color.
# Did the CodeAgent change the dot color? Yes. It wrote matplotlib with color='green'
# in step 1 and the plot was rendered correctly.

# What this reveals about when each type is better:
#   ToolCallingAgent is better when all required operations map to existing tools and
#   you want controlled, auditable behavior. It fails predictably and honestly when a
#   tool can't do something.
#
#   CodeAgent is more powerful for tasks that require custom styling or logic beyond
#   the tool surface. But it can waste steps on formatting errors after completing the
#   real work, and it needs careful prompting to call final_answer cleanly when done.
#   The tradeoff is capability vs. reliability.
 
 
# Q9 - Final reflection on ToolCallingAgent vs CodeAgent tradeoffs.

# 1. A task where ToolCallingAgent is better than CodeAgent:
#    An agent that monitors a data pipeline and pages an on-call engineer when a
#    quality check fails. Each action maps to a fixed function: run_quality_check(),
#    send_alert(), log_incident(). You never want the agent improvising code that
#    touches production tables or sends messages to the wrong channel. The bounded
#    tool surface is a safety guarantee, not a limitation. ToolCallingAgent is the
#    right fit here because correctness and predictability matter more than flexibility.

# 2. One meaningful risk of CodeAgent that does not apply to ToolCallingAgent:
#    Arbitrary code execution against the host environment. The CodeAgent generates
#    Python and runs it in a local interpreter. Even with additional_authorized_imports
#    limiting what can be imported, the agent can still do things like read local files,
#    write to disk, or consume CPU/memory in unexpected ways using only the allowed
#    libraries. ToolCallingAgent never generates or runs code; it only dispatches to
#    functions you explicitly registered. The smolagents sandbox restricts the most
#    dangerous built-ins, but the attack surface is still meaningfully larger than
#    a tool-calling setup.