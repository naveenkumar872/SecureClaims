import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../data/synthetic_fraud_dataset.csv")

# Encode categorical columns
le_accident = LabelEncoder()
le_damage = LabelEncoder()
df["accident_type_encoded"] = le_accident.fit_transform(df["accident_type"])
df["damage_type_encoded"] = le_damage.fit_transform(df["damage_type"])

# Features and target
features = ["customer_age", "customer_tenure_months", "vehicle_age_years", "idv", 
            "claim_amount", "repair_estimate", "police_report_filed", 
            "previous_claims_count", "engine_guard_addon", "tyre_rim_addon",
            "vehicle_fire", "theft", "accident_type_encoded", "damage_type_encoded"]

X = df[features]
y = df["fraud_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Fraud cases in training: {y_train.sum()}")
print()

# Model 1: Random Forest (fewer trees for less certainty)
print("=" * 50)
print("MODEL 1: Random Forest")
print("=" * 50)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(classification_report(y_test, rf_pred, zero_division=0))

# Model 2: Gradient Boosting
print("=" * 50)
print("MODEL 2: Gradient Boosting")
print("=" * 50)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
print(classification_report(y_test, gb_pred))

# Save model and encoders as a single file
model_bundle = {
    "model": rf_model,
    "le_accident": le_accident,
    "le_damage": le_damage
}
joblib.dump(model_bundle, "../models/fraud_model.joblib")
print("\nModel bundle saved to models/fraud_model.joblib")

# Test with a single ambiguous row (borderline case for low confidence)
print("\n" + "=" * 50)
print("TESTING WITH MULTIPLE TEST SAMPLES")
print("=" * 50)

# Try multiple test cases to find one with low confidence
test_cases = [
    # Case 1: Contradictory signals - high claim but police report filed
    {
        "customer_age": 35, "customer_tenure_months": 60, "vehicle_age_years": 3,
        "idv": 800000, "claim_amount": 600000, "repair_estimate": 650000,
        "police_report_filed": True, "previous_claims_count": 3,
        "engine_guard_addon": True, "tyre_rim_addon": True,
        "vehicle_fire": False, "theft": False,
        "accident_type_encoded": le_accident.transform(["Rear-end collision"])[0],
        "damage_type_encoded": le_damage.transform(["Front bumper"])[0]
    },
    # Case 2: Very average/normal case - hard to classify
    {
        "customer_age": 40, "customer_tenure_months": 36, "vehicle_age_years": 4,
        "idv": 500000, "claim_amount": 250000, "repair_estimate": 280000,
        "police_report_filed": True, "previous_claims_count": 1,
        "engine_guard_addon": False, "tyre_rim_addon": True,
        "vehicle_fire": False, "theft": False,
        "accident_type_encoded": le_accident.transform(["Hit and run"])[0],
        "damage_type_encoded": le_damage.transform(["Rear bumper"])[0]
    },
    # Case 3: Mixed signals - no police but low claim
    {
        "customer_age": 55, "customer_tenure_months": 80, "vehicle_age_years": 6,
        "idv": 700000, "claim_amount": 150000, "repair_estimate": 180000,
        "police_report_filed": False, "previous_claims_count": 2,
        "engine_guard_addon": True, "tyre_rim_addon": False,
        "vehicle_fire": False, "theft": False,
        "accident_type_encoded": le_accident.transform(["Side collision"])[0],
        "damage_type_encoded": le_damage.transform(["Windshield"])[0]
    },
    # Case 4: Unusual combination - old vehicle, moderate claim, some red flags
    {
        "customer_age": 28, "customer_tenure_months": 12, "vehicle_age_years": 9,
        "idv": 400000, "claim_amount": 200000, "repair_estimate": 250000,
        "police_report_filed": True, "previous_claims_count": 3,
        "engine_guard_addon": False, "tyre_rim_addon": False,
        "vehicle_fire": False, "theft": False,
        "accident_type_encoded": le_accident.transform(["Rollover"])[0],
        "damage_type_encoded": le_damage.transform(["Engine"])[0]
    },
    # Case 5: Borderline theft case - no police report
    {
        "customer_age": 32, "customer_tenure_months": 18, "vehicle_age_years": 2,
        "idv": 900000, "claim_amount": 450000, "repair_estimate": 500000,
        "police_report_filed": False, "previous_claims_count": 1,
        "engine_guard_addon": False, "tyre_rim_addon": False,
        "vehicle_fire": False, "theft": True,
        "accident_type_encoded": le_accident.transform(["Theft"])[0],
        "damage_type_encoded": le_damage.transform(["Total loss"])[0]
    },
    # Case 6: High previous claims but everything else looks normal
    {
        "customer_age": 50, "customer_tenure_months": 48, "vehicle_age_years": 5,
        "idv": 600000, "claim_amount": 200000, "repair_estimate": 230000,
        "police_report_filed": False, "previous_claims_count": 4,
        "engine_guard_addon": True, "tyre_rim_addon": True,
        "vehicle_fire": False, "theft": False,
        "accident_type_encoded": le_accident.transform(["Side collision"])[0],
        "damage_type_encoded": le_damage.transform(["Side panel"])[0]
    },
    # Case 7: Extreme borderline - theft without police but low claim ratio
    {
        "customer_age": 45, "customer_tenure_months": 30, "vehicle_age_years": 4,
        "idv": 1000000, "claim_amount": 400000, "repair_estimate": 450000,
        "police_report_filed": False, "previous_claims_count": 3,
        "engine_guard_addon": False, "tyre_rim_addon": True,
        "vehicle_fire": False, "theft": True,
        "accident_type_encoded": le_accident.transform(["Theft"])[0],
        "damage_type_encoded": le_damage.transform(["Engine"])[0]
    },
]

for i, test_row in enumerate(test_cases):
    test_df = pd.DataFrame([test_row])
    prediction = rf_model.predict(test_df)[0]
    probabilities = rf_model.predict_proba(test_df)[0]
    max_conf = max(probabilities) * 100
    
    print(f"\nTest Case {i+1}: Prediction={'FRAUD' if prediction else 'NOT FRAUD'}, "
          f"Confidence={max_conf:.2f}% (Fraud: {probabilities[1]*100:.2f}%, Not Fraud: {probabilities[0]*100:.2f}%)")
    
    if max_conf < 70:
        print(f"  ⚠️ LOW CONFIDENCE FOUND!")
        if max_conf < 50:
            print(f"  🔴 SENDING TO LLM for policy document review...")
else:
    print(f"\n✓ Confidence is sufficient ({max(probabilities) * 100:.2f}%)")

