# Video: https://youtu.be/IGGK4JXzhxY

import json
import os
import requests
import pandas as pd
from datetime import date
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

ACCOUNT_URL = "https://chikeziectd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Houston, TX coordinates
LATITUDE = 29.7604
LONGITUDE = -95.3698


def extract_weather():
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def serialize(data):
    return json.dumps(data).encode("utf-8")


def load_to_blob(container_client, blob_path, payload):
    container_client.upload_blob(blob_path, payload, overwrite=True)
    print(f"Uploaded {len(payload)} bytes to {blob_path}")


def list_blobs(container_client):
    print("\nBlobs in container:")
    for blob in container_client.list_blobs():
        print(f"  {blob.name}  ({blob.size} bytes)")


def read_back(container_client, blob_path):
    raw = container_client.download_blob(blob_path).readall()
    data = json.loads(raw.decode("utf-8"))
    df = pd.DataFrame(data["hourly"])
    return raw, df


def save_local(raw_bytes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(raw_bytes)
    print(f"\nSaved raw JSON to {output_path}")


def main():
    # Step 1 - Extract
    print("Extracting weather data from Open-Meteo...")
    data = extract_weather()

    # Step 2 - Serialize
    payload = serialize(data)

    # Step 3 - Load
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"

    credential = DefaultAzureCredential()
    container_client = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

    load_to_blob(container_client, blob_path, payload)

    # Step 4 - Verify
    list_blobs(container_client)

    # Step 5 - Read Back
    raw, df = read_back(container_client, blob_path)

    print("\nFirst 5 rows of hourly DataFrame:")
    print(df.head())

    # Save locally
    save_local(raw, "outputs/weather_raw.json")


if __name__ == "__main__":
    main()