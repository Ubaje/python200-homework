from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# Task 1: System Prompt
# Deliberate choice: I gave the assistant a specific role ("job application coach") and
# a specific user type ("career changer"). This keeps responses focused and avoids generic
# life-coach tangents. I also added an explicit reminder to review output before submitting,
# because LLMs hallucinate and someone taking AI output at face value in a job search could
# hurt their candidacy. The "I may not know your industry norms" constraint prevents the
# model from sounding overconfident in domains like law, medicine, or government contracting
# where cover letter norms are very different from a typical tech job.
SYSTEM_PROMPT = """You are a job application coach specializing in helping career changers.
Your user is transitioning from a previous field into a new one and needs help with resume bullet points and cover letters.

Your rules:
- Stay focused on job application materials only. Politely redirect off-topic questions.
- Always remind the user to review and personalize any output before submitting it to a real employer.
- Acknowledge that you may not know the specific norms of the user's target industry, and encourage them to use their own judgment.
- Do not invent credentials, job titles, or accomplishments the user did not mention.
- Keep your responses concise and actionable.
"""


# Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item must have exactly two keys:
    "original" (the original bullet text) and "improved" (your rewritten version).
    Do not include any text before or after the JSON array.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw = get_completion(messages, temperature=0.5)

    # Strip markdown code fences if the model wraps the JSON anyway
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(clean)
        print("\nRewritten Bullet Points:")
        for item in result:
            print(f"  Original:  {item['original']}")
            print(f"  Improved:  {item['improved']}")
            print()
        return result
    except json.JSONDecodeError:
        print("  JSON parse failed. Raw response:", raw)
        return []


# What makes the test bullets weak: they use vague, passive-sounding verbs ("helped", "made", "worked"),
# give no numbers or context, and describe activities rather than results.
# The model tends to add quantifiers where implied, swap weak verbs for specific ones (e.g.,
# "Collaborated" or "Delivered"), and reframe tasks as outcomes.


# Task 3: Cover Letter Generator
def generate_cover_letter(job_title: str, background: str) -> str:
    # I chose examples that both feature career changers with real domain knowledge transitioning
    # into technical roles. The few-shot pattern here controls tone (confident, not generic),
    # structure (3-5 sentences, ends with a clear "why this company" line), and what to avoid
    # (cliches like "I am passionate about" or inventing credentials).
    prompt = f"""You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of cliches.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information, which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)


# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("  Your message was flagged by our content filter. Please rephrase and try again.")
        return False
    return True


# Moderation tests
print("=== Moderation Tests ===")
safe_input = "Can you help me rewrite my resume for a software engineering role?"
unsafe_input = "I want to threaten my interviewer if they don't hire me."

print(f"  Safe input ({'PASS' if is_safe(safe_input) else 'FAIL - unexpectedly flagged'}): {safe_input}")
print(f"  Unsafe input ({'FAIL - not caught' if is_safe(unsafe_input) else 'CAUGHT as expected'}): {unsafe_input}")


# Bullet rewriter test
print("\n=== Bullet Rewriter Test ===")
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(bullets)


# Cover letter test
print("=== Cover Letter Test ===")
job_title = "Junior Data Engineer"
background = ("Five years of experience as a middle school math teacher; recently completed "
              "a Python course and built data pipelines using Prefect and Pandas.")

cover_letter = generate_cover_letter(job_title, background)
print("Generated cover letter opening:")
print(cover_letter)
print()


# Task 5: Chatbot Loop
def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets:
                rewrite_bullets(raw_bullets)
            else:
                print("Job Application Helper: No bullets entered.")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if job_title and background:
                result = generate_cover_letter(job_title, background)
                print(f"\nJob Application Helper: {result}")
                print("\n(Remember to review and personalize this before submitting anywhere.)\n")
            else:
                print("Job Application Helper: I need both a job title and your background to write that.")

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# Task 6: Ethics Reflection (Option A)
# Addressing questions 1 and 3.
#
# Question 1 - Bias in the training data:
# This bot was trained on text that skews heavily toward white-collar, English-speaking,
# Western professional culture. That shows up in subtle ways: the cover letter style it
# defaults to (confident, individualistic, results-focused) is very American tech-industry.
# Someone from a cultural background where self-promotion is considered rude, or someone
# targeting industries like government or academia where norms are completely different,
# could get advice that actually hurts them. The model also likely over-represents certain
# communication styles, vocabulary, and framing that correlates with educational privilege.
# A resume bullet rewritten to sound like a senior tech worker at a FAANG company might
# read as inauthentic or over-the-top for a trade job or a nonprofit role.
#
# Question 3 - Guardrails for a production deployment:
# The most important guardrail I would add is a clear disclaimer in the UI, not buried in
# a system prompt, that explicitly tells the user: "This output was generated by AI and
# may contain errors, fabricated details, or advice that does not match your industry norms.
# Do not submit this content without reviewing and editing it yourself." The current moderation
# check only catches harmful content, not the more common failure mode of plausible-sounding
# but subtly wrong advice. A second guardrail would be a "confidence" flag on the cover letter
# output, prompting the model to explicitly state what it does not know about the user's
# situation before producing the draft. That shifts the user from passive consumer to active
# editor, which is the right mental model for using AI in something as high-stakes as a job search.