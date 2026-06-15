# --- Azure Authentication ---

# Q1
# When a Python script runs locally and uses DefaultAzureCredential, it relies on
# the active CLI session established by running "az login" in your terminal.
# Running "az login" opens a browser, authenticates your Microsoft account, and
# stores a token in a local credential cache on your machine. When DefaultAzureCredential
# is instantiated, it tries a sequence of authentication methods in order. One of those
# methods is EnvironmentCredential (checks env vars), another is ManagedIdentityCredential
# (for Azure-hosted compute), and further down the chain is AzureCliCredential, which
# reads from that cached az login token. It knows to use it because the azure-identity
# SDK probes each method sequentially and the AzureCliCredential check succeeds as long
# as the token is valid and not expired.


# Q2
# A deployed pipeline running on a VM or container has no interactive session.
# There is no user present to open a browser and authenticate via "az login", and
# baking credentials into source code or environment variables is a security liability.
# Instead, deployed resources use a Managed Identity, which is an identity Azure
# automatically provisions and attaches to the compute resource (VM, container, etc.).
# Azure handles the credential lifecycle internally, so there is no secret to manage.
# When DefaultAzureCredential runs on that VM, it detects the managed identity via the
# instance metadata endpoint and authenticates through it. The Python code does not
# need to change at all because DefaultAzureCredential abstracts away which method
# is used. Locally it uses AzureCliCredential, in the cloud it uses ManagedIdentityCredential,
# and the caller never knows the difference.


# Q3
# Two most likely causes of an AuthenticationError immediately after instantiating
# DefaultAzureCredential:
#
# Cause 1 - No valid az login session.
# Either "az login" was never run, or the existing token expired.
# To diagnose: run "az account show" in the terminal. If it prints your subscription
# info, the session is valid. If it errors or says no subscription is found, run
# "az login" again to refresh the token.
#
# Cause 2 - The authenticated identity lacks the required RBAC permissions.
# The account is recognized by Azure but does not have the role needed (e.g.,
# "Storage Blob Data Contributor") on the target resource.
# To diagnose: go to the Azure Portal, navigate to the storage account (or other
# resource), open "Access Control (IAM)", and confirm the logged-in user has the
# correct role assignment. If not, add the role or ask an admin to do so.


# --- Blob Storage ---

# Q1
# Three-level hierarchy of Azure Blob Storage:
#
# Level 1 - Storage Account: the top-level resource tied to a unique Azure URL
#   (https://<account-name>.blob.core.windows.net). Think of this as the hard drive.
#   Every file you store ultimately lives under one account.
#
# Level 2 - Container: a logical bucket inside the storage account, analogous to
#   a top-level folder on that hard drive. You might have separate containers for
#   different projects or pipeline stages (e.g., "raw-data", "processed-data").
#
# Level 3 - Blob: an individual file stored inside a container. Blob names can
#   contain forward slashes to simulate subdirectories (e.g., "raw/2024-01-15/weather.json"),
#   but this is purely a naming convention, not a real directory hierarchy.
#
# Analogy using a filing cabinet:
#   Storage Account = the entire filing cabinet in the office.
#   Container       = one of the drawers in that cabinet.
#   Blob            = a document (or folder of documents) inside that drawer.


# Q2
# Scenario 1 - A REST API returns a JSON payload each hour, raw responses stored for reprocessing:
# Use Blob Storage. Raw API responses are unstructured files, and the access pattern is
# write-once, read-later by file, not query-by-value. Blob Storage is the right fit.
#
# Scenario 2 - 50 million customer transactions queried by date range and customer ID daily:
# Use a relational database (e.g., Azure SQL). The access pattern is row-level filtering,
# joins, and aggregations, which require indexed query execution, not file retrieval.
#
# Scenario 3 - NumPy image embeddings saved between pipeline runs:
# Use Blob Storage. Embeddings are binary array files. They are stored and retrieved
# as-is between runs, not queried by value, so Blob Storage is the appropriate choice.


# Q3
def list_container(container_client):
    """
    Prints the name and size (in bytes) of every blob in the container.

    Args:
        container_client: An azure.storage.blob.ContainerClient instance.
    """
    for blob in container_client.list_blobs():
        print(f"{blob.name}  ({blob.size} bytes)")


# Q4
def upload_text(container_client, blob_name, text):
    """
    Encodes a string as UTF-8 and uploads it to Blob Storage, overwriting
    any existing blob with the same name.

    Args:
        container_client: An azure.storage.blob.ContainerClient instance.
        blob_name: The target blob path within the container (str).
        text: The content to upload (str).
    """
    data = text.encode("utf-8")
    container_client.upload_blob(blob_name, data, overwrite=True)