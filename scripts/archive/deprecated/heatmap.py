import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Your similarity matrix (copy from output)
labels = ['A1_H', 'A1_A', 'A2_H', 'A2_A', 'A3_H', 'A3_A', 'A4_H', 'A4_A']

M = np.array([
    [1.000,  0.457,  0.636,  0.074, -0.042,  0.499,  0.455, -0.296],
    [0.457,  1.000,  0.141,  0.353,  0.506,  0.517,  0.628,  0.367],
    [0.636,  0.141,  1.000,  0.373,  0.158,  0.245,  0.513,  0.127],
    [0.074,  0.353,  0.373,  1.000,  0.621,  0.415,  0.477,  0.719],
    [-0.042, 0.506,  0.158,  0.621,  1.000,  0.151,  0.543,  0.790],
    [0.499,  0.517,  0.245,  0.415,  0.151,  1.000,  0.176,  0.081],
    [0.455,  0.628,  0.513,  0.477,  0.543,  0.176,  1.000,  0.456],
    [-0.296, 0.367,  0.127,  0.719,  0.790,  0.081,  0.456,  1.000]
])

plt.figure(figsize=(10, 8))
sns.heatmap(M, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlBu_r',
            xticklabels=labels,
            yticklabels=labels,
            center=0.5,
            vmin=-0.3, 
            vmax=1.0,
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('Emotion Similarity Matrix (Normalized Hybrid Features)')
plt.tight_layout()
plt.savefig('similarity_heatmap_normalized.png', dpi=300)
plt.show()