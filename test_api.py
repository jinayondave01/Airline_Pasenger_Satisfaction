"""Smoke test with assertions:  python test_api.py   (no server needed)"""
from app import app

PASSENGER = {
    "Gender": "Female", "Age": 35, "Customer Type": "Returning", "Type of Travel": "Business",
    "Class": "Business", "Flight Distance": 1200, "Departure Delay": 0, "Arrival Delay": 0,
    "Departure and Arrival Time Convenience": 4, "Ease of Online Booking": 4, "Check-in Service": 4,
    "Online Boarding": 5, "Gate Location": 3, "On-board Service": 4, "Seat Comfort": 5,
    "Leg Room Service": 4, "Cleanliness": 4, "Food and Drink": 4, "In-flight Service": 4,
    "In-flight Wifi Service": 4, "In-flight Entertainment": 5, "Baggage Handling": 4,
}
c = app.test_client()

r = c.get("/health").get_json()
assert r["status"] == "ok" and r["explain_enabled"], r

r = c.post("/api/predict", json=PASSENGER); j = r.get_json()
assert r.status_code == 200 and 0 <= j["probability_satisfied"] <= 1 and j["label"] in (
    "Satisfied", "Neutral or Dissatisfied"), j
assert list(j)[0] == "prediction", "keys should keep insertion order"

r = c.post("/api/explain", json=PASSENGER); j = r.get_json()
feats = [x["feature"] for x in j["contributions"]]
assert r.status_code == 200 and len(feats) == len(set(feats)), "duplicate features"
assert not any("_" in f for f in feats), f"one-hot names leaked: {feats}"
for x in j["contributions"]:                       
    assert x["value"] == PASSENGER[x["feature"]], x

assert c.post("/api/predict", json={**PASSENGER, "Class": "Premium", "Seat Comfort": 9}).status_code == 400
assert c.post("/api/predict", json={**PASSENGER, "Baggage Handling": 0}).status_code == 400   
assert c.post("/api/predict", json={**PASSENGER, "Seat Comfort": 0}).status_code == 400       
assert c.post("/api/predict", json={**PASSENGER, "In-flight Wifi Service": 0}).status_code == 200   
assert c.post("/api/predict", json={**PASSENGER, "Online Boarding": 0}).status_code == 200          
assert c.post("/api/predict", json={**PASSENGER, "Foo": 1}).status_code == 400
assert c.post("/api/predict", json={**PASSENGER, "Arrival Delay": None}).status_code == 200   
r = c.post("/api/predict", json={**PASSENGER, "Age": 100}); assert r.status_code == 200 and r.get_json()["warnings"]
assert c.post("/api/predict", data="x", content_type="text/plain").status_code == 400
print("All checks passed")
