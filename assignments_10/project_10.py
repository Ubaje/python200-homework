"""
Week 10 Project: LLM Transform Pipeline

Reflection:
    Classifying running conditions from temperature and precipitation is a weak use
    case for an LLM. Both inputs are clean numeric values with intuitive cutoffs, so
    a deterministic rule like "temperature between 5 and 25 and precipitation under
    1 mm equals good" would produce the same answers with zero cost, no latency, and
    full reproducibility. Switching to rules would gain speed, testability, and
    consistent output, and would lose only the fuzzy human judgment a model brings,
    which barely matters when the decision reduces to two numbers. The LLM is
    justified here mainly as a teaching exercise for the Transform pattern, not
    because the task genuinely needs language understanding. In a real pipeline I
    would reach for code first and reserve the model for inputs that are actually
    ambiguous, like free-text weather descriptions.

Video: https://youtu.be/ikjMfKbD4Zc
"""

import json
import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

ACCOUNT_URL = "https://chikeziectd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
FALLBACK_PATH = "../../assignments/resources/weather_raw.json"
RECORD_LIMIT = 24
VALID_LABELS = {"good", "marginal", "bad"}

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)


def reshape_hourly(data):
    hourly = data["hourly"]
    records = []
    for i in range(len(hourly["time"])):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })
    return records


def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )


def read_records(container, today):
    blob_path = f"raw/{today}/weather.json"
    #blob_path = f"fail"
    try:
        raw = container.download_blob(blob_path).readall()
        data = json.loads(raw.decode("utf-8"))
        print(f"Loaded raw data from {blob_path}")
    except Exception as err:
        print(f"Could not read {blob_path} ({err}). Using fallback dataset.")
        with open(FALLBACK_PATH) as f: #closes file
            data = json.load(f)
    return reshape_hourly(data)


def classify_records(client, records):
    enriched = []
    for i, record in enumerate(records):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(record)},
            ],
        )
        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"
        enriched.append({**record, "conditions": label})
        if i % 6 == 5:
            print(f"  Processed {i + 1} records...")
    return enriched


def main():
    today = date.today().isoformat()
    credential = DefaultAzureCredential()
    container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

    # Step 1: Read
    records = read_records(container, today)
    records = records[:RECORD_LIMIT]
    print(f"Classifying {len(records)} records")

    # Step 2: Transform
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    enriched = classify_records(client, records)

    # Step 3: Write
    processed_path = f"processed/{today}/weather_classified.json"
    payload = json.dumps(enriched).encode("utf-8")
    container.upload_blob(processed_path, payload, overwrite=True)
    print(f"Uploaded {len(payload)} bytes to {processed_path}")

    # Step 4: Spot-check
    raw = container.download_blob(processed_path).readall()
    df = pd.DataFrame(json.loads(raw.decode("utf-8")))
    print("\nLabel distribution:")
    print(df["conditions"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head(5))

    # Step 5: Save output
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/first_10_records.json", "w") as f:
        json.dump(enriched[:10], f, indent=2)
    print("\nSaved first 10 records to outputs/first_10_records.json")


if __name__ == "__main__":
    main()