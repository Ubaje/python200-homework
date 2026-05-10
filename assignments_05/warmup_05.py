from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("API Q1 - Response text:", response.choices[0].message.content)
print("API Q1 - Model name:", response.model)
print("API Q1 - Total tokens used:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

print("\nAPI Q2 - Temperature experiment:")
for temp in temperatures:
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"  Temperature {temp}: {result.choices[0].message.content}")

# At temp=0 and 0.7 the output was identical for this prompt, which suggests 
# the token probabilities were heavily skewed toward one answer regardless 
# of the sampling setting. At 1.5 you finally get variation.

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

print("\nAPI Q3 - Three completions:")
for i, choice in enumerate(response.choices):
    print(f"  Choice {i + 1}: {choice.message.content}")

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print("\nAPI Q4 - max_tokens=15 result:", response.choices[0].message.content)
# The response gets cut off mid-sentence because we capped the output at 15 tokens.
# In a real application you'd use max_tokens to control cost and prevent runaway responses,
# for example in a production chatbot where you bill per token or need a predictable response length.

# --- System Messages and Personas ---

# System Q1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

result = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nSystem Q1 - Encouraging tutor persona:")
print(result.choices[0].message.content)

messages_grumpy = [
    {"role": "system", "content": "You are a grumpy senior developer who has seen too many bad codebases. You answer questions correctly but always complain that the person should have Googled it first."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

result_grumpy = client.chat.completions.create(model="gpt-4o-mini", messages=messages_grumpy)
print("\nSystem Q1 - Grumpy dev persona:")
print(result_grumpy.message.content if hasattr(result_grumpy, 'message') else result_grumpy.choices[0].message.content)
# The factual content is similar but the tone is completely different. The second persona adds
# complaints and sarcasm. This shows how much the system message shapes voice and behavior.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

result = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nSystem Q2 - Stateless memory via message history:")
print(result.choices[0].message.content)
# The model knows Jordan's name because we passed the full conversation history in the messages list.
# The API itself has no memory between calls. The "memory" is just context we supply each time.
# If we removed the first user message from the list, the model would have no idea who Jordan is.

# --- Prompt Engineering ---

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

# Prompt Q1 - Zero-Shot
print("\nPrompt Q1 - Zero-shot sentiment classification:")
for i, review in enumerate(reviews):
    prompt = f"Classify the sentiment of this review as positive, negative, or mixed.\n\nReview: {review}\nSentiment:"
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"  Review {i + 1}: {result.choices[0].message.content.strip()}")

# Prompt Q2 - One-Shot
print("\nPrompt Q2 - One-shot sentiment classification:")
one_shot_prefix = """Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

"""

for i, review in enumerate(reviews):
    prompt = one_shot_prefix + f'Review: "{review}"\nSentiment:'
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"  Review {i + 1}: {result.choices[0].message.content.strip()}")

# Adding one example made the output format more consistent. Zero-shot sometimes returns a full
# sentence like "The sentiment is positive." whereas one-shot tends to return just the label word.

# Prompt Q3 - Few-Shot
print("\nPrompt Q3 - Few-shot sentiment classification:")
few_shot_prefix = """Classify the sentiment of each review as positive, negative, or mixed.

Example 1:
Review: "Incredibly easy setup and the customer service was outstanding."
Sentiment: positive

Example 2:
Review: "The product broke after two days and there was no warranty support."
Sentiment: negative

Example 3:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

"""

for i, review in enumerate(reviews):
    prompt = few_shot_prefix + f'Review: "{review}"\nSentiment:'
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"  Review {i + 1}: {result.choices[0].message.content.strip()}")

# Zero-shot works fine for simple tasks but output format can vary a lot.
# One-shot locks in the format quickly and is a good default for most classification tasks.
# Few-shot is best when you have edge cases you want to explicitly cover, like making sure
# the model knows the difference between "mixed" and "negative." I'd use zero-shot for quick
# prototyping, one-shot when I need format consistency, and few-shot when accuracy on edge cases matters.

# Prompt Q4 - Chain of Thought
print("\nPrompt Q4 - Chain of thought reasoning:")
cot_prompt = """Solve the problem below. Think through it step by step before giving a final answer.
Label your final answer clearly as "Final Answer:".

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}]
)
print(result.choices[0].message.content)

# Asking the model to reason step by step forces it to break a multi-step problem into smaller
# pieces before committing to an answer. LLMs are essentially next-token predictors, so if you
# ask for a direct answer the model might latch onto a plausible-sounding number without doing
# the actual arithmetic. Requiring intermediate steps gives each calculation its own "reasoning budget"
# and makes errors much easier to spot.

# Prompt Q5 - Structured Output
print("\nPrompt Q5 - Structured JSON output:")
import json

review = ("I've been using this tool for three months. It handles large datasets well, "
          "but the UI is clunky and the export options are limited.")

structured_prompt = f"""Analyze the review below and return ONLY valid JSON with these exact keys:
- "sentiment": one of positive, negative, or mixed
- "confidence": a float from 0 to 1
- "reason": one sentence explaining the sentiment

Do not include any text before or after the JSON object.

Review: {review}
"""

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": structured_prompt}]
)

raw = result.choices[0].message.content.strip()
print("  Raw response:", raw)

try:
    parsed = json.loads(raw)
    print("  Sentiment:", parsed["sentiment"])
    print("  Confidence:", parsed["confidence"])
    print("  Reason:", parsed["reason"])
except json.JSONDecodeError:
    print("  JSON parse failed. Raw response for debugging:", raw)

# Prompt Q6 - Delimiters
print("\nPrompt Q6 - Delimiters:")
user_text = ("First boil a pot of water. Once boiling, add a handful of salt and the "
             "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice.")

prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("  Instruction text result:")
print(" ", result.choices[0].message.content)

non_instruction_text = "The weather in Houston in July is brutally hot and humid."
prompt_no_steps = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

result2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_no_steps}]
)
print("  Non-instruction text result:", result2.choices[0].message.content)

# Delimiters prevent "prompt injection" where user-supplied text accidentally bleeds into
# the instruction part of the prompt and confuses the model. Without them, a malicious or
# ambiguous user input could manipulate the model's behavior by looking like part of the instructions.

# --- Local Models with Ollama ---

# Ollama Q1

# Ollama CLI output (run: ollama run qwen3:0.6b "Explain what a large language model is in two sentences.")
"""
A large language model (LLM) is an artificial intelligence system designed to understand and generate human-like text by analyzing vast amounts of language data. It uses deep learning techniques, particularly neural networks, to process and predict the next word in a sentence, enabling it to perform various language-related tasks such as translation, summarization, and conversation.
"""

result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)
print("\nOllama Q1 - OpenAI gpt-4o-mini response:")
print(result.choices[0].message.content)

# Differences: gpt-4o-mini tends to give cleaner, more polished phrasing. qwen3:0.6b is more
# compact in its wording, sometimes a bit rougher, but still coherent for a 0.6B model.
# Advantage of local: no API cost, no data leaving your machine, works offline.
# Disadvantage of local: smaller models produce noticeably lower quality output, and you need
# enough RAM/GPU to run the model at all.