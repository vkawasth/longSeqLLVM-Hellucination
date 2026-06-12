import json
import ollama
import time
from FactScoreLite import FactScorer

# -------------------------------------------------------------
# 1. CORE AI ENGINES (LOCAL OLLAMA BRIDGES)
# -------------------------------------------------------------

def extract_atomic_facts_local(text):
    """
    Uses Llama 3.2 1B with custom parameters to extract atomic facts.
    Emulates legacy GPT-2 baseline generation parameters and context limits.
    """
    prompt = (
        f"System: You are a strict factual extraction engine. Do not chat or write an introduction.\n"
        f"User: Deconstruct the following text into distinct, simple atomic facts.\n"
        f"Output exactly ONE basic sentence per line. Do not use bullet points or numbering.\n\n"
        f"Text: {text}"
    )
    
    response = ollama.generate(
        model='llama3.2:1b', # Verfied Local Tag
        prompt=prompt,
        options={
            'num_ctx': 1024,      # Critical: Lock window to GPT-2 bounds
            'temperature': 0.2,   # Low variance for structural breakdown
            'top_k': 40,          # GPT-2 original default filtering
            'top_p': 0.9          # GPT-2 original default nucleus sampling
        }
    )
    
    return [line.strip() for line in response['response'].split('\n') if line.strip()]


def validate_fact_local(fact, context_knowledge):
    """
    Upgraded validator: Tolerates 1B conversational drift by scanning for explicit negative cues.
    """
    prompt = (
        f"Context: {context_knowledge}\n\n"
        f"Fact to check: {fact}\n\n"
        f"Is this fact supported by the context? Answer yes or no."
    )
    
    response = ollama.generate(
        model='llama3.2:1b', 
        prompt=prompt, 
        options={'temperature': 0.0} # Absolute determinism
    )
    
    # Clean the string for broad matching
    verdict = response['response'].strip().lower()
    
    # 1B models often output 'not applicable' or change formatting if confused.
    # If the response explicitly states 'no', 'false', or 'not', consider it rejected.
    if any(neg in verdict for neg in ["no", "false", "not supported", "incorrect"]):
        return False
        
    # If it leans positive ('yes', 'true', 'supported'), confirm it
    if any(pos in verdict for pos in ["yes", "true", "supported"]):
        return True
        
    return False # Fallback safety default

# -------------------------------------------------------------
# 2. RUNNING THE FACTSCORE LITE PIPELINE
# -------------------------------------------------------------

knowledge_base = {}

print("📂 Loading local knowledge source...")
try:
    with open("my_knowledge_source.jsonl", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            clean_line = line.strip()
            if not clean_line:
                continue
            try:
                data = json.loads(clean_line)
                knowledge_base[data["title"]] = data["text"]
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Skipping malformed JSON on line {line_num}: (Error: {e})")
except FileNotFoundError:
    print("❌ Error: Could not find 'my_knowledge_source.jsonl' in this directory.")
    exit(1)

# Input parameters for your local experiment run
topic_to_test = "Albert Einstein"
generated_output_to_score = "Albert Einstein was born in Germany. He won a Nobel Prize in Chemistry."

print(f"\n📊 Analyzing generation for topic: '{topic_to_test}'...")

# Phase A: Atomic Fact Breakdown
extracted_facts = extract_atomic_facts_local(generated_output_to_score)
print(f"➡️ Extracted {len(extracted_facts)} Atomic Facts.")

# Phase B: Match against Local Knowledge Map
reference_text = knowledge_base.get(topic_to_test, "")

if not reference_text:
    print(f"❌ Error: Topic '{topic_to_test}' was not found in your local JSONL reference data.")
else:
    supported_count = 0
    detailed_results = []
    
    # Phase C: Evaluate each fact sequentially
    for fact in extracted_facts:
        is_supported = validate_fact_local(fact, reference_text)
        if is_supported:
            supported_count += 1
        detailed_results.append({
            "fact": fact, 
            "status": "SUPPORTED" if is_supported else "HALLUCINATION"
        })
    
    # Phase D: Compute Final Metric Scores
    fact_score = supported_count / len(extracted_facts) if extracted_facts else 0
    
    # -------------------------------------------------------------
    # 3. EXPERIMENT METRICS PRINT OUT
    # -------------------------------------------------------------
    print("\n" + "="*20 + " EXPERIMENT METRICS " + "="*20)
    print(f"Target Topic      : {topic_to_test}")
    print(f"Local FActScore   : {fact_score * 100:.1f}%")
    print(f"Total Atomic Facts: {len(extracted_facts)}")
    print(f"Verified True     : {supported_count}")
    print(f"Hallucinations    : {len(extracted_facts) - supported_count}")
    print("-"*49)
    for res in detailed_results:
        marker = "✅" if res["status"] == "SUPPORTED" else "❌"
        print(f"{marker} [{res['status']}]: {res['fact']}")
    print("="*49)

