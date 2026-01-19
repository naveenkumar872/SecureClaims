import pandas as pd
import joblib
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ==================== CONFIGURATION ====================
CONFIDENCE_THRESHOLD = 0.70  # Below this, claim needs LLM review
COLLECTION_NAME = "insurance_policy_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ==================== LOAD MODELS & CLIENTS ====================
print("Loading models and clients...")

# ML Model
rf_model = joblib.load("fraud_model.joblib")
le_accident = joblib.load("le_accident.joblib")
le_damage = joblib.load("le_damage.joblib")

# Embedding model for RAG
device = "cpu"
embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Qdrant client with increased timeout
qdrant_client = QdrantClient(
    url=os.getenv("QUADRANT_URL"),
    api_key=os.getenv("QUADRANT_API_KEY"),
    timeout=60,  # Increase timeout to 60 seconds
)

# Groq client for LLM
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("All models loaded successfully!\n")


# ==================== STEP 1: FEATURE EXTRACTION ====================
def extract_features(claim: dict) -> pd.DataFrame:
    """Convert raw claim into structured features for ML model."""
    features = {
        "customer_age": claim["customer_age"],
        "customer_tenure_months": claim["customer_tenure_months"],
        "vehicle_age_years": claim["vehicle_age_years"],
        "idv": claim["idv"],
        "claim_amount": claim["claim_amount"],
        "repair_estimate": claim["repair_estimate"],
        "police_report_filed": claim["police_report_filed"],
        "previous_claims_count": claim["previous_claims_count"],
        "engine_guard_addon": claim["engine_guard_addon"],
        "tyre_rim_addon": claim["tyre_rim_addon"],
        "vehicle_fire": claim["accident_type"] == "Fire",
        "theft": claim["accident_type"] == "Theft",
        "accident_type_encoded": le_accident.transform([claim["accident_type"]])[0],
        "damage_type_encoded": le_damage.transform([claim["damage_type"]])[0],
    }
    return pd.DataFrame([features])


# ==================== STEP 2: ML FRAUD PREDICTION ====================
def predict_fraud(claim: dict) -> dict:
    """Run ML model to predict fraud probability."""
    features_df = extract_features(claim)
    
    prediction = rf_model.predict(features_df)[0]
    probabilities = rf_model.predict_proba(features_df)[0]
    
    fraud_probability = probabilities[1]
    confidence = max(probabilities)
    
    return {
        "prediction": "FRAUD" if prediction else "NOT FRAUD",
        "fraud_probability": fraud_probability,
        "confidence": confidence,
        "needs_llm_review": confidence < CONFIDENCE_THRESHOLD
    }


# ==================== STEP 3: RAG - GENERATE SEARCH QUERY ====================
def generate_search_query(claim: dict, ml_result: dict) -> str:
    """Ask LLM to generate a targeted search query for policy documents."""
    
    claim_summary = f"""
Claim Details:
- Accident Type: {claim['accident_type']}
- Damage Type: {claim['damage_type']}
- Claim Amount: Rs. {claim['claim_amount']:,} (IDV: Rs. {claim['idv']:,})
- Claim to IDV Ratio: {(claim['claim_amount'] / claim['idv'] * 100):.1f}%
- Vehicle Age: {claim['vehicle_age_years']} years
- Police Report Filed: {'Yes' if claim['police_report_filed'] else 'No'}
- Previous Claims: {claim['previous_claims_count']}
- ML Prediction: {ml_result['prediction']} ({ml_result['fraud_probability'] * 100:.1f}% fraud probability)
"""

    prompt = f"""You are an insurance fraud analyst. Based on the following claim, generate a simple search query or a follow up question to find the most relevant policy clauses and rules in a document using similarity search.

{claim_summary}

Generate a focused search query (2-3 sentences) that will help retrieve:
1. Coverage rules for this type of accident/damage
2. Depreciation rules if applicable
3. Claim limits and exclusions
4. Any fraud-related policy clauses

Return ONLY the search query, nothing else."""

    full_prompt = "You are an insurance policy search expert. Generate precise search queries.\n\n" + prompt
    
    response = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
        max_completion_tokens=150
    )
    
    return response.choices[0].message.content.strip()


# ==================== STEP 4: RAG - RETRIEVE POLICY DOCS ====================
def retrieve_policy_docs(search_query: str, top_k: int = 3) -> list:
    """Retrieve relevant policy documents from Qdrant based on LLM-generated query."""
    
    # Generate embedding for query
    query_embedding = embed_model.encode(search_query).tolist()
    
    # Search Qdrant
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    ).points
    
    # Extract text from results
    policy_docs = [hit.payload["text"] for hit in results]
    return policy_docs


# ==================== STEP 5: LLM REASONING (HIGH CONFIDENCE) ====================
def llm_reasoning_high_confidence(claim: dict, ml_result: dict, policy_docs: list) -> dict:
    """LLM reasoning for HIGH confidence ML predictions - focus on explanation."""
    
    claim_summary = f"""
CLAIM DETAILS:
- Customer Age: {claim['customer_age']} years
- Customer Tenure: {claim['customer_tenure_months']} months
- Vehicle Age: {claim['vehicle_age_years']} years
- Insured Declared Value (IDV): Rs. {claim['idv']:,}
- Claim Amount: Rs. {claim['claim_amount']:,}
- Repair Estimate: Rs. {claim['repair_estimate']:,}
- Claim to IDV Ratio: {(claim['claim_amount'] / claim['idv'] * 100):.1f}%
- Accident Type: {claim['accident_type']}
- Damage Type: {claim['damage_type']}
- Police Report Filed: {'Yes' if claim['police_report_filed'] else 'No'}
- Previous Claims Count: {claim['previous_claims_count']}
- Engine Guard Add-on: {'Yes' if claim['engine_guard_addon'] else 'No'}
- Tyre & Rim Add-on: {'Yes' if claim['tyre_rim_addon'] else 'No'}
"""

    policy_context = "\n\n---\n\n".join(policy_docs) if policy_docs else "No policy documents available."

    prompt = f"""You are an insurance fraud analyst. The ML model has made a HIGH CONFIDENCE prediction. Your job is to EXPLAIN and VALIDATE this decision.

{claim_summary}

ML MODEL PREDICTION (HIGH CONFIDENCE):
- Decision: {ml_result['prediction']}
- Fraud Probability: {ml_result['fraud_probability'] * 100:.1f}%
- Confidence Score: {ml_result['confidence'] * 100:.1f}%

RELEVANT POLICY DOCUMENTS:
{policy_context}

Since the ML model is confident, provide:
1. FINAL DECISION: Confirm or adjust the ML prediction (FRAUD / NOT FRAUD)
Do not make final decision other fraud/ not fraud
2. EXPLANATION: Explain why this claim is {ml_result['prediction'].lower()} based on:
   - Key factors from the claim data
   - Relevant policy clauses that support this decision
3. RISK FACTORS: List any red flags or positive indicators
4. RECOMMENDATION: Approve, Reject, or any special handling needed

Keep the response concise and actionable."""

    full_prompt = "You are an expert insurance fraud analyst. Provide clear, concise explanations.\n\n" + prompt
    
    response = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3,
      
    )
    
    return {
        "reasoning": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens if response.usage else 0
    }


# ==================== STEP 6: LLM REASONING (LOW CONFIDENCE) ====================
def llm_reasoning_low_confidence(claim: dict, ml_result: dict, policy_docs: list) -> dict:
    """LLM reasoning for LOW confidence ML predictions - thorough independent analysis."""
    
    claim_summary = f"""
CLAIM DETAILS:
- Customer Age: {claim['customer_age']} years
- Customer Tenure: {claim['customer_tenure_months']} months
- Vehicle Age: {claim['vehicle_age_years']} years
- Insured Declared Value (IDV): Rs. {claim['idv']:,}
- Claim Amount: Rs. {claim['claim_amount']:,}
- Repair Estimate: Rs. {claim['repair_estimate']:,}
- Claim to IDV Ratio: {(claim['claim_amount'] / claim['idv'] * 100):.1f}%
- Accident Type: {claim['accident_type']}
- Damage Type: {claim['damage_type']}
- Police Report Filed: {'Yes' if claim['police_report_filed'] else 'No'}
- Previous Claims Count: {claim['previous_claims_count']}
- Engine Guard Add-on: {'Yes' if claim['engine_guard_addon'] else 'No'}
- Tyre & Rim Add-on: {'Yes' if claim['tyre_rim_addon'] else 'No'}
"""

    policy_context = "\n\n---\n\n".join(policy_docs) if policy_docs else "No policy documents available."

    prompt = f"""You are a SENIOR insurance fraud analyst. The ML model has LOW CONFIDENCE on this claim, so YOU must make the final decision independently.

{claim_summary}

ML MODEL PREDICTION (LOW CONFIDENCE - UNRELIABLE):
- Decision: {ml_result['prediction']} (DO NOT rely on this)
- Fraud Probability: {ml_result['fraud_probability'] * 100:.1f}%
- Confidence Score: {ml_result['confidence'] * 100:.1f}% (Below threshold!)

RELEVANT POLICY DOCUMENTS:
{policy_context}

⚠️ The ML model is UNCERTAIN. You must perform INDEPENDENT analysis:

1. FRAUD INDICATORS ANALYSIS:
   - Is the claim amount unusually high relative to IDV?
   - Is there a police report for theft/major accidents?
   - How many previous claims does this customer have?
   - Are there any suspicious patterns?

2. POLICY COMPLIANCE CHECK:
   - Does the claim fall within coverage limits?
   - Are there any exclusions that apply?
   - What depreciation rules apply to this vehicle age?

3. YOUR INDEPENDENT DECISION: FRAUD or NOT FRAUD
   - Provide your confidence level: HIGH, MEDIUM, or LOW

4. DETAILED REASONING:
   - Explain step-by-step why you reached this conclusion
   - Reference specific policy clauses

5. RECOMMENDATION:
   - Should this claim be approved, rejected, or escalated for investigation?
   - What additional verification is needed?

Be thorough - this is a borderline case that needs careful analysis."""

    full_prompt = "You are a senior insurance fraud analyst handling complex borderline cases. Be thorough and make independent decisions.\n\n" + prompt
    
    response = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.4,
        
    )
    
    return {
        "reasoning": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens if response.usage else 0
    }


# ==================== STEP 7: FULL PIPELINE ====================
def process_claim(claim: dict) -> dict:
    """Complete fraud detection pipeline."""
    print("=" * 60)
    print("PROCESSING NEW CLAIM")
    print("=" * 60)
    
    # Step 1: ML Prediction
    print("\n[Step 1] Running ML Model...")
    ml_result = predict_fraud(claim)
    print(f"  Prediction: {ml_result['prediction']}")
    print(f"  Fraud Probability: {ml_result['fraud_probability'] * 100:.1f}%")
    print(f"  Confidence: {ml_result['confidence'] * 100:.1f}%")
    
    # Step 2: Check confidence level
    if ml_result['needs_llm_review']:
        print(f"\n[Step 2] ⚠️ LOW CONFIDENCE ({ml_result['confidence'] * 100:.1f}%) - LLM will make independent decision")
    else:
        print(f"\n[Step 2] ✓ HIGH CONFIDENCE ({ml_result['confidence'] * 100:.1f}%) - LLM will explain the decision")
    
    # Step 3: LLM generates search query
    print("\n[Step 3] LLM generating search query for policy documents...")
    try:
        search_query = generate_search_query(claim, ml_result)
        print(f"  Generated query:\n  {search_query}")
    except Exception as e:
        print(f"  Warning: Could not generate search query - {e}")
        search_query = f"{claim['accident_type']} {claim['damage_type']} insurance policy coverage"
    
    # Step 4: Retrieve policy documents using LLM-generated query
    print("\n[Step 4] Retrieving policy documents from Qdrant...")
    try:
        policy_docs = retrieve_policy_docs(search_query)
        print(f"  Retrieved {len(policy_docs)} relevant policy sections")
        print("\n  --- RETRIEVED DOCUMENTS ---")
        for i, doc in enumerate(policy_docs, 1):
            print(f"\n  [Document {i}]:")
            print(f"  {doc[:500]}..." if len(doc) > 500 else f"  {doc}")
        print("\n  --- END OF DOCUMENTS ---")
    except Exception as e:
        print(f"  Warning: Could not retrieve policy docs - {e}")
        policy_docs = []
    
    # Step 5/6: LLM Reasoning (different prompts based on confidence)
    if ml_result['needs_llm_review']:
        print("\n[Step 5] LLM performing INDEPENDENT analysis (low confidence case)...")
        llm_result = llm_reasoning_low_confidence(claim, ml_result, policy_docs)
    else:
        print("\n[Step 5] LLM generating explanation (high confidence case)...")
        llm_result = llm_reasoning_high_confidence(claim, ml_result, policy_docs)
    
    print(f"  Tokens used: {llm_result['tokens_used']}")
    
    # Final output
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)
    print(llm_result['reasoning'])
    
    return {
        "claim": claim,
        "ml_prediction": ml_result,
        "search_query": search_query,
        "policy_docs_count": len(policy_docs),
        "llm_analysis": llm_result['reasoning'],
        "tokens_used": llm_result['tokens_used']
    }


# ==================== TEST THE PIPELINE ====================
if __name__ == "__main__":
    # Test with a low confidence claim (Case 7 - borderline theft)
    print("\n" + "=" * 60)
    print("TEST CASE: Low Confidence Claim (Theft without police report)")
    print("=" * 60)
    
    test_claim_low_conf = {
        "customer_age": 45,
        "customer_tenure_months": 30,
        "vehicle_age_years": 4,
        "idv": 1000000,
        "claim_amount": 400000,
        "repair_estimate": 450000,
        "police_report_filed": False,
        "previous_claims_count": 3,
        "engine_guard_addon": False,
        "tyre_rim_addon": True,
        "accident_type": "Theft",
        "damage_type": "Engine"
    }
    
    result = process_claim(test_claim_low_conf)
    
    # Save full LLM analysis to file
    with open("analysis_output.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("FULL LLM ANALYSIS OUTPUT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ML Prediction: {result['ml_prediction']['prediction']}\n")
        f.write(f"Fraud Probability: {result['ml_prediction']['fraud_probability'] * 100:.1f}%\n")
        f.write(f"Confidence: {result['ml_prediction']['confidence'] * 100:.1f}%\n")
        f.write(f"Tokens Used: {result['tokens_used']}\n\n")
        f.write("Search Query:\n")
        f.write(result['search_query'] + "\n\n")
        f.write("=" * 60 + "\n")
        f.write("LLM REASONING:\n")
        f.write("=" * 60 + "\n\n")
        f.write(result['llm_analysis'])
    
    print("\n\n✓ Full analysis saved to 'analysis_output.txt'")
    
    # Close Qdrant client to avoid warning
    qdrant_client.close()
