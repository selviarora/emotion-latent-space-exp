# scripts/cosine_matrix_hybrid.py
import glob, os, numpy as np

def cosine(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.dot(u, v))

def main():
    # Collect in a fixed order: A1..A4 × (happy, angry)
    entries = [
        ("actor01_happy", "embeddings/actor01/happy_hybrid.npy"),
        ("actor01_angry", "embeddings/actor01/angry_hybrid.npy"),
        ("actor02_happy", "embeddings/actor02/happy_hybrid.npy"),
        ("actor02_angry", "embeddings/actor02/angry_hybrid.npy"),
        ("actor03_happy", "embeddings/actor03/happy_hybrid.npy"),
        ("actor03_angry", "embeddings/actor03/angry_hybrid.npy"),
        ("actor04_happy", "embeddings/actor04/happy_hybrid.npy"),
        ("actor04_angry", "embeddings/actor04/angry_hybrid.npy"),
    ]

    labels = [k for k,_ in entries]
    X = [np.load(p) for _,p in entries]
    X = np.stack(X, axis=0)  # [N, 776]

    N = len(labels)
    M = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            M[i, j] = cosine(X[i], X[j])

    # pretty print
    colw = max(len(s) for s in labels)
    header = " " * (colw+1) + " ".join([f"{s:>{colw}}" for s in labels])
    print("Cosine similarity matrix:")
    print(header)
    for i in range(N):
        row = f"{labels[i]:>{colw}} " + " ".join([f"{M[i,j]:>{colw}.3f}" for j in range(N)])
        print(row)

    # summary stats like before
    # cross-speaker SAME-emotion (happy-vs-happy across actors, angry-vs-angry across actors)
    same_emotion = []
    same_speaker_cross_emotion = []
    for a in range(1,5):
        for b in range(1,5):
            if a==b: continue
            same_emotion.append(M[labels.index(f"actor0{a}_happy"), labels.index(f"actor0{b}_happy")])
            same_emotion.append(M[labels.index(f"actor0{a}_angry"), labels.index(f"actor0{b}_angry")])
    for a in range(1,5):
        same_speaker_cross_emotion.append(M[labels.index(f"actor0{a}_happy"), labels.index(f"actor0{a}_angry")])

    print(f"\nAvg cross-speaker SAME-emotion cosine: {np.mean(same_emotion):.3f}")
    print(f"Avg same-speaker cross-emotion cosine: {np.mean(same_speaker_cross_emotion):.3f}")



if __name__ == "__main__":
    main()
