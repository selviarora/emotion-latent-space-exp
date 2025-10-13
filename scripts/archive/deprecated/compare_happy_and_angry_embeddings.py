import numpy as np

happy = np.load("embeddings/actor04/happy_layer5.npy")
angry = np.load("embeddings/actor04/angry_layer5.npy")

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

cos = cosine_similarity(happy, angry)
l2  = float(np.linalg.norm(happy - angry))

print(f"Cosine similarity (Happy vs Angry): {cos:.3f}   # closer to 1 = more similar")
print(f"Euclidean distance (Happy vs Angry): {l2:.3f}  # larger = more different")
