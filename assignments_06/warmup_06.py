from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

def headliner(string:str):
    print("\n" + "="*40)
    print(string)
    print("="*40)


# --- RAG Concepts ---

# Concepts Q1
# Situation A: RAG
# Every quarter, hundreds of PDFs are updated by the legal team. Every time 
# papers are updated, fine-tuning would necessitate retraining, which is 
# costly and sluggish. Prompt engineering cannot manage hundreds of documents 
# by itself. RAG enables the model to retrieve relevant policy text at query 
# time and modify theTo utilize the document storage, you only need to add or 
# modify files.

# Scenario B: Fine-tuning
# Here, style is more important than knowledge. The firm wants the model to 
# write in a very particular brand voice, which is rare among training data 
# that is available to the general public. Because you are incorporating a 
# behavioral pattern into the model's weights rather than obtaining external 
# information, fine-tuning on 3,000 internal samples is the appropriate method.

# Scenario C: Context injection through prompt engineering
# The analyst has a one-time requirement and a single two-page report. The 
# easiest, least expensive, and quickest option is to paste the entire content 
# into the prompt context. For a two-page document that is only used once, RAG 
# and fine-tuning would be quite excessive.


# Concepts Q2
# "I'm not sure" is less risky than a confidently incorrect response since it
# eliminates the reader's natural tendency to double-check. The reader is aware
# that a model should be double-checked when it hedges. When it speaks with 
# authority, people act without examining the output, treating it as fact.

# An actual example would be a doctor looking up information on drug interactions 
# using an AI assistant. When two medications are not safe to combine, the model 
# boldly claims that they are. The patient is given the incorrect treatment since 
# the doctor fails to cross-check a reference database due to the authoritative tone.

# Because people are programmed to see confident speaking as a sign of expertise, 
# tone is important. When a model states, "the patient should take 500mg of X," it 
# sounds like a professional delivering instructions. "I believe it might be around
# "500mg but I'm not certain" is a model that expresses doubt and asks the reader 
# to confirm. Very varied behavioral results with the same information.


# Concepts Q3
# Proper sequence with descriptions:

# 1. Take text out of the original papers
# To give the system something to work with, extract raw text from Word documents, PDFs, and other source files.

# 2. Divide the text into sections
# Divide the extracted text into manageable chunks so that retrieval can be accurate and the model is not compelled to analyze complete texts at once.

# 3. Create embeddings from text segments
# To obtain a vector that quantitatively represents each chunk's meaning, run it through an embedding model.

# 4. Get the user's inquiry
# The pipeline awaits a query from the user.

# 5. Include the user's inquiry
# Utilizing the same embedding approach, transform the user's query into a vector for comparison with the chunk vectors that have been saved.

# 6. Get the most pertinent sections
# To identify the chunks whose vectors are closest to the query vector, use cosine similarity (or another metric).

# 7. Add the obtained portions to the prompt
# To allow the model to reason over the top retrieved chunks, paste them alongside the original query in the LLM's context box.

# 8. Produce an answer from the LLM
# The LLM generates an answer based on the collected content after reading the question and the injected context.


# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Keyword Q1
headliner("Keyword Q1")
query_1 = "What are your hours on the weekend?"
result_1 = simple_keyword_retrieval(query_1, documents, verbose=True)
print(f"Selected document: {result_1[0][0]}")

# The incorrect response, loyalty.txt, was chosen. Hours were the anticipated 
# outcome.text. Common question words like "what," "your," "do," and "how" are 
# not included in the assignment's stopword list, which causes them to pass 
# through the filter as query tokens.

# The tokens that were really filtered were: ['hours', 'weekend', 'what', 'your']
# "weekend" and "hours" do not appear literally in hours.text. "What" doesn't occur 
# in any text, and the document says "weekends" (plural), therefore "weekend" doesn't 
# match because of the single vs. plural mismatch. However, "your" is present in both 
# hiring.txt files. Giving each a score of 1, loyalty.txt ("Send your resume") and 
# loyalty.txt ("a free drink of your choice") are ultimately chosen first by dict ordering.

# This illustrates two issues at once: keyword matching lacks stemming, treating 
# "weekend" and "weekends" as distinct strings, and the stopword list is too 
# limited (leaking function words like "your" that match at random).


# Keyword Q2
headliner("Keyword Q2")
query_2 = "Do you have anything without caffeine?"
result_2 = simple_keyword_retrieval(query_2, documents, verbose=True)
print(f"Selected document: {result_2[0][0]}")

# The right thing to note in this case is that no document was chosen. The 
# actual filtered tokens were: ['anything', 'caffeine', 'do', 'have', 'without']; 
# since "do," "have," and "without" aren't on the assignment's stopword list, 
# they seeped through. Every document received a zero because none of those 
# terms and "caffeine" are found in any of the documents.

# The document that is most likely to be helpful is menu.txt 
# (oat milk, almond milk, cold brew), but the query doesn't share a single 
# token with it, which is why keyword RAG fails in this case. For keyword search 
# to be effective, "caffeine" would have to appear in the page.

# This would be appropriately handled by semantic RAG. Even in the absence of 
# precise word overlap, an embedding model recognizes that "anything without 
# caffeine" is semantically similar to menu and beverage content. Even when 
# the vocabulary doesn't match, the query's meaning corresponds to the 
# document's meaning.



# Keyword Q3
headliner("Keyword Q3")
query_3 = "How do I sign up for rewards?"

# Prediction: loyalty.txt
# I anticipate that the query's mention of "rewards" will overlap with 
# loyalty.txt. because points, redemption, and the loyalty program are covered 
# in that booklet.Although "rewards" isn't a keyword synonym for "loyalty," I'm 
# assuming the phrase appears in the page.

result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)
print(f"Selected document: {result_3[0][0]}")

# The prediction was incorrect: no document was chosen.
# The tokens that were really filtered were: ['do', 'how', 'i', 'rewards', 'sign', 'up']
# There is not a single literal instance of the word "rewards" in loyalty.text. The terms 
# "loyalty," "program," "points," and "redeem" are used in that text, but "rewards" are 
# never mentioned. The synonym link between "rewards" and "loyalty program" is completely 
# useless because keyword search requires an exact string match.

# This is a clear illustration of the vocabulary mismatch issue. Although they utilize 
# different terminology, the user and the document are discussing the same topic. Because 
# the embeddings for the "rewards program" and "loyalty program" would fall near to one 
# another in vector space, semantic RAG would resolve this.


# --- Semantic RAG Concepts ---

# Semantic Q1
#
# A vector embedding: what is it?
# A set of numbers that expresses a text's meaning is called a vector embedding. A neural network that has been trained to group texts with similar meanings together in that numerical space generates the numbers.

# 0.85 vs. 0.30 cosine similarity:
# A score of 0.85 indicates that the piece is more pertinent. In high-dimensional space, cosine similarity quantifies the angle formed by two vectors. A score approaching 1 indicates that the two vectors are pointing almost in the same direction, indicating a similar meaning between the texts. A high semantic match is indicated with a score of 0.85. A score of 0.30 indicates that the texts' meanings are largely unrelated.

# Why exact word matches are not necessary for semantic search to function:
# Embeddings understand that words like "car" and "automobile" occur in comparable situations after being educated on vast volumes of text. In embedding space, the resulting vectors for those words end up near each other. Therefore, even though the surface-level strings are different when you search for "car" and the text says "automobile," the embedding similarity is still high. The model does not only learn spelling; it also learns meaning from context.

# Semantic Q2
# | Feature                 | Keyword RAG                   | Semantic RAG                         |
# |-------------------------|-------------------------------|--------------------------------------|
# | What is compared?       | Exact word overlap            | Vector similarity (meaning/semantics)|
# | What is retrieved?      | Full document                 | Specific chunks (sub-document pieces)|
# | Can it handle synonyms? | No                            | Yes                                  |
# | Storage format          | Plain text dictionary         | Vector database / embedding index    |
# | Relevance score         | Number of overlapping keywords| Cosine similarity score (0 to 1)     |


# --- LlamaIndex ---

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
import asyncio

BRIGHTLEAF_PATH = "../../lessons/06_AI_augmentation/resources/brightleaf_pdfs"

headliner("LlamaIndex Q1")

brightleaf_docs = SimpleDirectoryReader(BRIGHTLEAF_PATH).load_data()
brightleaf_index = VectorStoreIndex.from_documents(brightleaf_docs)
brightleaf_engine = brightleaf_index.as_query_engine(similarity_top_k=3)

#%%

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQuestion: {q}")
    response = brightleaf_engine.query(q)
    print(f"Answer: {response}")
    print("\nSource nodes:")
    for i, node in enumerate(response.source_nodes):
        score = round(node.score, 4) if node.score else "N/A"
        preview = node.text[:150].replace("\n", " ")
        print(f"  Node {i+1} | Score: {score} | Text: {preview}")

# Q1 results:
#
# Query 1 (employee benefits):
# Retrieval worked well. Node 1 scored 0.91 and came from the benefits
# introduction section, which is directly on topic. Nodes 2 and 3 (company
# overview and security intro) are less relevant but still BrightLeaf content.
# The model's response was confident and specific, listing actual benefit names
# like the Wellness Reimbursement Plan, 401(k) match, and Learning Hub. No
# hedging language. The high top-node score (0.91) and specific response are
# both signs that retrieval and generation worked as intended here.
#
# Query 2 (security policies):
# Also worked well. Node 1 (0.88) came from the network and data security
# section, which is the right document. The answer was detailed and specific,
# covering MFA, VPN, credential rotation, encryption, and ISO 27001 alignment.
# Interestingly, Node 2 was the benefits intro chunk again (0.84), which is
# not relevant to security. The model correctly ignored it and focused on the
# security chunk. This is a good sign that the LLM can filter out noise from
# the retrieved context when the top result is strong enough.


# LlamaIndex Q2
headliner("LlamaIndex Q2")

test_query = "What employee benefits does BrightLeaf offer?"

for k in [1, 5]:
    engine_k = brightleaf_index.as_query_engine(similarity_top_k=k)
    resp = engine_k.query(test_query)
    print(f"\n--- top_k={k} ---")
    print(f"Answer: {resp}")
    print("Source node scores:")
    for node in resp.source_nodes:
        score = round(node.score, 4) if node.score else "N/A"
        print(f"  Score: {score}")

# top_k=1 vs top_k=5 results:
#
# Both runs returned the same top node (score 0.91, benefits intro chunk),
# so the core answer was identical in both cases. The difference was in
# detail level. top_k=1 gave a slightly more focused list. top_k=5 gave a
# longer answer with specifics like the $600 Wellness Reimbursement limit,
# telemedicine, and nutrition counseling, because those details came from
# additional chunks that weren't included in the top_k=1 run.
#
# In this case more context genuinely helped: the extra chunks added real
# detail without introducing noise, because all five chunks were still
# relevant BrightLeaf content with scores between 0.79 and 0.91.
# The diminishing returns concern applies more when lower-ranked chunks are
# from unrelated documents. Here the document set is small and focused, so
# all five chunks stayed on topic.


# LlamaIndex Q3
headliner("LlamaIndex Q3")

# Trying something vague that spans multiple documents or might not be in
# the docs at all. Asking about the company's future product roadmap is a
# reasonable test since internal policy docs usually don't cover that.
hard_query = "What new products or services is BrightLeaf planning to launch next year?"
engine_3 = brightleaf_index.as_query_engine(similarity_top_k=3)
hard_response = engine_3.query(hard_query)
print(f"Question: {hard_query}")
print(f"Answer: {hard_response}")
print("\nSource nodes:")
for node in hard_response.source_nodes:
    score = round(node.score, 4) if node.score else "N/A"
    preview = node.text[:150].replace("\n", " ")
    print(f"  Score: {score} | Text: {preview}")

# Expected: The model would hedge or admit it doesn't know, since product
# roadmaps aren't typically in internal HR or policy documents.
#
# What actually happened: The model hallucinated a confident forward-looking
# answer, stating BrightLeaf is planning to expand into Latin America through
# NGO collaborations. That information came from a partnerships chunk about
# past activity, not a roadmap. The model extrapolated future plans from
# existing context and presented it as fact without hedging at all.
#
# The chunk scores were 0.83, 0.83, and 0.82. Those are reasonably high
# scores, which means the retriever found related content, but "related to
# BrightLeaf's work" is not the same as "answers a question about next year's
# product launches." High retrieval scores don't guarantee the question can
# actually be answered from those chunks.
#
# To improve this: set a minimum relevance threshold for the specific question
# type, or use a system prompt that instructs the model to say "I don't have
# information about future plans" when the retrieved content doesn't directly
# address the query. A confidence gate before generation would prevent this
# kind of fluent but fabricated response.


# LlamaIndex Q4
headliner("LlamaIndex Q4")

judge_llm = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

eval_query_good = "What employee benefits does BrightLeaf offer?"
eval_response_good = brightleaf_engine.query(eval_query_good)

faith_result_good = faithfulness_evaluator.evaluate_response(response=eval_response_good)
rel_result_good = relevancy_evaluator.evaluate_response(
    query=eval_query_good, response=eval_response_good
)

print(f"\nGood query: {eval_query_good}")
print(f"Faithfulness score: {faith_result_good.score}")
print(f"Relevancy score: {rel_result_good.score}")

# Now test with a query that the docs probably can't answer well
eval_query_bad = "What is BrightLeaf's stock price and market cap?"
eval_response_bad = brightleaf_engine.query(eval_query_bad)

faith_result_bad = faithfulness_evaluator.evaluate_response(response=eval_response_bad)
rel_result_bad = relevancy_evaluator.evaluate_response(
    query=eval_query_bad, response=eval_response_bad
)

print(f"\nPoor query: {eval_query_bad}")
print(f"Faithfulness score: {faith_result_bad.score}")
print(f"Relevancy score: {rel_result_bad.score}")

# --- Evaluation Q&A ---
#
# Faithfulness score of 1.0 means every claim in the model's response is
# supported by the retrieved source chunks. Nothing was made up. A score of
# 0.0 would mean the response contains statements that contradict or have no
# basis in the retrieved material, which is essentially the model hallucinating
# while ignoring what was given to it.
#
# Relevancy measures whether the response actually answers the question that
# was asked. A response can be perfectly faithful (everything it says is in
# the source) but still irrelevant (it answered a different question). The
# two scores capture different failure modes: faithfulness catches fabrication,
# relevancy catches topic drift.
#
# The scores likely changed between the two queries. The first query (benefits)
# is probably well-supported by the documents, so both scores should be close
# to 1.0. The stock price query is not in internal HR docs, so the model might
# retrieve loosely related text and produce a response that is either
# unfaithful (makes something up) or irrelevant (answers about something else
# entirely). Lower scores on the second query reflect both problems.
#
# The "LLM-as-a-judge" approach uses a language model (here gpt-4o-mini) to
# evaluate another language model's output. It's used instead of a simple
# accuracy metric because RAG responses are free-form text, not multiple
# choice answers. You can't compute "correct vs incorrect" with a string
# comparison. An LLM judge can read both the response and the source material
# and make a nuanced judgment about whether the claims are supported. The
# tradeoff is that the judge model can also make mistakes, so this is not a
# perfect ground truth.