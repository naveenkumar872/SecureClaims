from faker import Faker
import random
import pandas as pd

fake = Faker("en_IN")

ACCIDENT_TYPES = ["Rear-end collision", "Side collision", "Fire", "Theft", "Rollover", "Hit and run"]
DAMAGE_TYPES = ["Front bumper", "Rear bumper", "Side panel", "Engine", "Windshield", "Total loss"]

def generate_claim():
    customer_age = random.randint(21, 70)
    customer_tenure_months = random.randint(1, 120)
    vehicle_age_years = random.randint(0, 12)
    idv = random.randint(300000, 1200000)

    accident_type = random.choice(ACCIDENT_TYPES)
    damage_type = random.choice(DAMAGE_TYPES)

    police_report_filed = random.choice([True, True, True, False])
    previous_claims_count = random.randint(0, 6)
    engine_guard_addon = random.choice([True, False])
    tyre_rim_addon = random.choice([True, False])

    vehicle_fire = accident_type == "Fire"
    theft = accident_type == "Theft"

    base_claim = random.randint(10000, int(idv * 0.8))
    repair_estimate = base_claim + random.randint(2000, 30000)

    # --- Fraud Logic ---
    fraud_risk = 0

    if not police_report_filed:
        fraud_risk += 30
    if previous_claims_count >= 4:
        fraud_risk += 25
    if base_claim > idv * 0.7:
        fraud_risk += 20
    if theft and not police_report_filed:
        fraud_risk += 40
    if vehicle_age_years > 8 and base_claim > idv * 0.6:
        fraud_risk += 15

    fraud_label = fraud_risk >= 50

    # --- ADD NOISE: Randomly flip labels to prevent overfitting ---
    # 25% chance to flip the fraud label (introduces more uncertainty)
    
    
    # Add random fraud cases that don't follow the rules (15%)
    if random.random() < 0.15:
        fraud_label = random.choice([True, False])
    
    # Add borderline cases with 50/50 outcome (20%)
    if 40 <= fraud_risk <= 60:
        if random.random() < 0.40:
            fraud_label = random.choice([True, False])

    return {
        "customer_age": customer_age,
        "customer_tenure_months": customer_tenure_months,
        "vehicle_age_years": vehicle_age_years,
        "idv": idv,
        "claim_amount": base_claim,
        "repair_estimate": repair_estimate,
        "police_report_filed": police_report_filed,
        "accident_type": accident_type,
        "damage_type": damage_type,
        "previous_claims_count": previous_claims_count,
        "engine_guard_addon": engine_guard_addon,
        "tyre_rim_addon": tyre_rim_addon,
        "vehicle_fire": vehicle_fire,
        "theft": theft,
        "fraud_label": fraud_label
    }

def generate_dataset(n=10000):
    data = [generate_claim() for _ in range(n)]
    return pd.DataFrame(data)

df = generate_dataset(10000)
df.to_csv("../data/synthetic_fraud_dataset.csv", index=False)
print(df.head())
