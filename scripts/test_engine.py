import sys
import os

sys.path.insert(0, os.path.abspath("backend"))
from app.services.siamese_engine import SiameseEngine, process_brand_inference

engine = SiameseEngine.get_instance()
print("Catalog count:", len(engine.catalog))
print("Embeddings shape:", engine.embeddings.shape)

test_img_path = os.path.join("backend", "app", "data", "brands", "consultas80", "1.png")
with open(test_img_path, "rb") as f:
    query_bytes = f.read()

results = process_brand_inference(query_bytes, top_k=5)
print("Top 5 results for 1.png query:")
for r in results:
    print(f"  - ID: {r['id']}, Name: {r['name']}, Sim: {r['similarity_score']}%, Dist: {r['distance']}, URL: {r['image_url']}")

print("\nEngine test successful!")
