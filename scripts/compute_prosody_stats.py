import numpy as np, glob
# Some shells don't expand {,} in glob; safer explicit list:
paths = sorted(glob.glob("embeddings/actor*/happy_prosody.npy") + glob.glob("embeddings/actor*/angry_prosody.npy"))

X = np.stack([np.load(p) for p in paths], axis=0)  # [N, 8]
mu = X.mean(axis=0)                                # (8,)
sd = X.std(axis=0)                                 # (8,)
np.save("embeddings/prosody_mean.npy", mu)
np.save("embeddings/prosody_std.npy",  sd)
print("[save] embeddings/prosody_mean.npy", mu)
print("[save] embeddings/prosody_std.npy",  sd)
