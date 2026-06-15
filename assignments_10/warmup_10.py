# Week 10 Warmup

# --- LLMs as Transform ---

# Q1
# Parse "Jan 5th, 2024" into "2024-01-05":
#   Deterministic code. Date parsing has one correct answer and libraries like
#   dateutil handle it reliably without the cost or variability of a model call.
# Classify a support ticket ("my card was charged twice") as billing/technical/general:
#   LLM. The intent has to be inferred from free text, which needs reading
#   comprehension that rules cannot cover for every phrasing a customer might use.
# Calculate the average of a list of numbers:
#   Deterministic code. Arithmetic is exact and should never be handed to a model.
# Extract the company name from "Sr. Data Eng @ Acme Corp (contract)":
#   LLM. The input format is irregular, so a model is more robust than trying to
#   anticipate every separator, suffix, and noise pattern with regex.
# Determine whether a review is more than 100 words long:
#   Deterministic code. Splitting on whitespace and counting the tokens is exact
#   and trivial, so there is no reason to involve a model.

# Q2
# The prompt "Summarize this product review in a few sentences." is a problem
# downstream because "a few sentences" is unconstrained. Each record comes back
# with a different length, tone, and shape, and sometimes with a preamble like
# "Sure, here is a summary." That inconsistency is hard to store in a fixed column
# and hard to parse reliably. A pipeline needs predictable output, not prose that
# changes shape from row to row.
#
# Rewritten so the output is easy to parse and store:
SUMMARY_PROMPT = (
    "Summarize this product review in one sentence of 25 words or fewer. "
    "Reply with the summary text only, no preamble or labels."
)

# Q3
# 1. 50,000 records * 1 second each = 50,000 seconds, which is about 13.9 hours
#    if the calls run one after another.
# 2. Run the calls concurrently instead of sequentially, for example with a
#    thread pool or an async client that keeps many requests in flight at once.
#    OpenAI's Batch API is the other standard option. Both cut wall-clock time
#    dramatically without switching to a different model.

# --- Azure OpenAI ---

# Q1
# 1. Data residency and compliance: with Azure OpenAI the requests stay inside
#    the organization's own Azure environment instead of leaving to OpenAI's
#    servers, which is often required in regulated industries like healthcare,
#    finance, and government.
# 2. Unified billing and support: Azure OpenAI usage appears on the same Azure
#    bill as the rest of the infrastructure and support runs through Microsoft,
#    which simplifies procurement and reduces vendor relationships for large
#    organizations already standardized on Azure.

# Q2
# The Azure-specific parameters on the AzureOpenAI client:
#   azure_endpoint: the URL of your Azure OpenAI resource, for example
#     "https://<resource-name>.openai.azure.com". It points the client at your
#     organization's resource instead of OpenAI's public endpoint.
#   api_version: the dated Azure OpenAI REST API version string, for example
#     "2024-02-01". Azure pins request and response behavior to a specific
#     version, so you have to state which one you are calling.
#   azure_deployment: the name of the model deployment to use. The SDK accepts
#     this on the client itself, though you can also pass the deployment name as
#     the model argument on each request, which is what the lesson example does.

# Q3
# It takes the deployment name, not the model name. In Azure OpenAI you do not
# call a model directly. You call a named deployment that an admin created and
# configured, which might be something like "gpt4o-mini-prod" rather than
# "gpt-4o-mini". You find the deployment name in Azure AI Foundry, under the
# Deployments section of your Azure OpenAI resource,and your platform or 
# infrastructure team can also provide it directly.

