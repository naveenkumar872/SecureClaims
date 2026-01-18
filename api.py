from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ==================== CONFIGURATION ====================
CONFIDENCE_THRESHOLD = 0.70
COLLECTION_NAME = "insurance_policy_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ==================== LOAD MODELS & CLIENTS ====================
print("Loading models and clients...")

# ML Model (bundled with encoders)
model_bundle = joblib.load("models/fraud_model.joblib")
rf_model = model_bundle["model"]
le_accident = model_bundle["le_accident"]
le_damage = model_bundle["le_damage"]

# Embedding model for RAG
device = "cpu"
embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QUADRANT_URL"),
    api_key=os.getenv("QUADRANT_API_KEY"),
    timeout=60,
)

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("All models loaded successfully!")

# ==================== FASTAPI APP ====================
app = FastAPI(title="SecureClaims Fraud Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ==================== PYDANTIC MODELS ====================
class ClaimInput(BaseModel):
    customer_age: int
    customer_tenure_months: int
    vehicle_age_years: int
    idv: int
    claim_amount: int
    repair_estimate: int
    police_report_filed: bool
    previous_claims_count: int
    engine_guard_addon: bool
    tyre_rim_addon: bool
    accident_type: str
    damage_type: str


class FraudResult(BaseModel):
    ml_prediction: str
    fraud_probability: float
    confidence: float
    needs_llm_review: bool
    search_query: str
    policy_docs_count: int
    llm_analysis: str
    final_decision: str


# ==================== HELPER FUNCTIONS ====================
def extract_features(claim: dict) -> pd.DataFrame:
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


def predict_fraud(claim: dict) -> dict:
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


def generate_search_query(claim: dict, ml_result: dict) -> str:
    claim_summary = f"""
Claim Details:
- Accident Type: {claim['accident_type']}
- Damage Type: {claim['damage_type']}
- Claim Amount: Rs. {claim['claim_amount']:,} (IDV: Rs. {claim['idv']:,})
- Claim to IDV Ratio: {(claim['claim_amount'] / claim['idv'] * 100):.1f}%
- Vehicle Age: {claim['vehicle_age_years']} years
- Police Report Filed: {'Yes' if claim['police_report_filed'] else 'No'}
- Previous Claims: {claim['previous_claims_count']}
"""
    prompt = f"""Generate a search query to find relevant insurance policy clauses for this claim:
{claim_summary}
Return ONLY the search query."""

    response = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
      
    )
    return response.choices[0].message.content.strip()


def retrieve_policy_docs(search_query: str, top_k: int = 3) -> list:
    query_embedding = embed_model.encode(search_query).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k
    ).points
    return [hit.payload["text"] for hit in results]


def llm_reasoning(claim: dict, ml_result: dict, policy_docs: list) -> str:
    claim_summary = f"""
CLAIM DETAILS:
- Customer Age: {claim['customer_age']} years
- Customer Tenure: {claim['customer_tenure_months']} months
- Vehicle Age: {claim['vehicle_age_years']} years
- IDV: Rs. {claim['idv']:,}
- Claim Amount: Rs. {claim['claim_amount']:,}
- Repair Estimate: Rs. {claim['repair_estimate']:,}
- Claim to IDV Ratio: {(claim['claim_amount'] / claim['idv'] * 100):.1f}%
- Accident Type: {claim['accident_type']}
- Damage Type: {claim['damage_type']}
- Police Report Filed: {'Yes' if claim['police_report_filed'] else 'No'}
- Previous Claims: {claim['previous_claims_count']}
- Engine Guard Add-on: {'Yes' if claim['engine_guard_addon'] else 'No'}
- Tyre & Rim Add-on: {'Yes' if claim['tyre_rim_addon'] else 'No'}
"""
    policy_context = "\n\n---\n\n".join(policy_docs) if policy_docs else "No policy documents available."
    
    confidence_type = "LOW" if ml_result['needs_llm_review'] else "HIGH"
    
    prompt = f"""You are an insurance fraud analyst for an Indian insurance company. Analyze this claim.
IMPORTANT: All monetary values are in Indian Rupees (Rs.). Always use Rs. when referring to amounts.

{claim_summary}

ML MODEL PREDICTION ({confidence_type} CONFIDENCE):
- Decision: {ml_result['prediction']}
- Fraud Probability: {ml_result['fraud_probability'] * 100:.1f}%
- Confidence: {ml_result['confidence'] * 100:.1f}%

POLICY DOCUMENTS:
{policy_context}

Provide:
1. FINAL DECISION: FRAUD or NOT FRAUD
2. KEY FINDINGS: Main reasons for your decision
3. RISK FACTORS: Red flags or positive indicators
4. RECOMMENDATION: Approve, Reject, or Investigate

Be concise but thorough. Use Rs. for all currency references."""

    response = groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,

    )
    return response.choices[0].message.content


# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.post("/api/detect-fraud", response_model=FraudResult)
async def detect_fraud(claim: ClaimInput):
    try:
        claim_dict = claim.model_dump()
        
        # Step 1: ML Prediction
        ml_result = predict_fraud(claim_dict)
        
        # Step 2: Generate search query
        try:
            search_query = generate_search_query(claim_dict, ml_result)
        except Exception as e:
            search_query = f"{claim_dict['accident_type']} {claim_dict['damage_type']} insurance coverage"
        
        # Step 3: Retrieve policy docs
        try:
            policy_docs = retrieve_policy_docs(search_query)
        except Exception as e:
            policy_docs = []
        
        # Step 4: LLM Reasoning
        llm_analysis = llm_reasoning(claim_dict, ml_result, policy_docs)
        
        # Extract final decision from LLM response
        if "FINAL DECISION" in llm_analysis.upper():
            decision_part = llm_analysis.upper().split("FINAL DECISION")[1][:50]
            if "NOT FRAUD" in decision_part:
                final_decision = "NOT FRAUD"
            elif "FRAUD" in decision_part:
                final_decision = "FRAUD"
            else:
                final_decision = ml_result['prediction']
        else:
            final_decision = ml_result['prediction']
        
        return FraudResult(
            ml_prediction=ml_result['prediction'],
            fraud_probability=round(ml_result['fraud_probability'] * 100, 2),
            confidence=round(ml_result['confidence'] * 100, 2),
            needs_llm_review=ml_result['needs_llm_review'],
            search_query=search_query,
            policy_docs_count=len(policy_docs),
            llm_analysis=llm_analysis,
            final_decision=final_decision
        )
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
