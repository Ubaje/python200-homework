# Video: https://youtu.be/bKQrR96EsuQ

import os
from dotenv import load_dotenv
import requests
from prefect import task, flow
from datetime import date
from openai import OpenAI
import json
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

ACCOUNT_URL = "https://chikeziectd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24  
LATITUDE = 29.7604 # Houston
LONGITUDE = -95.3698

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted 7 days of hourly data for ({latitude}, {longitude})")
    return response.json()

@task
def transform(data: dict, max_records: int) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
 
    hourly = data["hourly"]
    count = min(max_records, len(hourly["time"]))
 
    records = []
    for i in range(count):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })
 
    enriched = []
    for i, record in enumerate(records):
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
 
        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"
 
        enriched.append({**record, "conditions": label})
 
        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")
 
    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched

@task
def load(records: list, blob_path: str) -> None:
    credential = DefaultAzureCredential()
    container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)
 
    payload = json.dumps(records).encode("utf-8")
    container.upload_blob(blob_path, payload, overwrite=True)
 
    print(f"Loaded {len(payload)} bytes to {blob_path}")

@flow(log_prints=True)
def etl_pipeline(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE

):
    today = date.today().isoformat()

    blob_path = f"final/{today}/weather_etl.json"

    data = extract(latitude, longitude)

    # os.makedirs("outputs", exist_ok=True)
    # with open("outputs/data.json", "w") as f:
    #     json.dump(data, f, indent=2)

    enriched = transform(
        data,
        max_records=MAX_RECORDS
    )

    # os.makedirs("outputs", exist_ok=True)
    # with open("outputs/enriched.json", "w") as f:
    #     json.dump(enriched, f, indent=2)

    load(enriched, blob_path)
    print(f"Pipeline complete. Results at {blob_path}")

if __name__ == "__main__":
    etl_pipeline()

