# Testing Prosody Variance: Signal vs Noise?

## The Question
Is prosody variance (±0.275) capturing real emotion subtypes, or is it just noise?

## Hypothesis 1: Variance is SIGNAL (subtypes exist)
**Predictions if true:**
- [ ] With 24 actors, we see consistent subgroups (e.g., 10-12 "explosive" actors, 10-12 "controlled" actors)
- [ ] Prosody variance correlates with RAVDESS intensity labels (strong vs normal)
- [ ] Multiple samples from same actor cluster together in prosody space
- [ ] Features are interpretable (high f0_std = explosive, low = controlled)

## Hypothesis 2: Variance is NOISE (unstable)
**Predictions if true:**
- [ ] With 24 actors, variance stays high but no clear patterns
- [ ] Same actor's multiple samples are scattered (high within-actor variance)
- [ ] No correlation with intensity or other metadata
- [ ] Cannot interpret what drives the differences

## Test Plan

### Phase 1: Extract All Data
- Extract prosody for all 24 actors × 2 emotions × 2 intensities = 96 samples minimum
- Compute statistics across full dataset

### Phase 2: Clustering
- K-means on prosody features (k=2,3,4)
- Check: Do natural clusters emerge within each emotion?
- Check: Are clusters interpretable from feature values?

### Phase 3: Consistency Tests
- Multiple samples per actor: Do they cluster together?
- Cross-validate: Train classifier on 16 actors, test on 8
- Stability: Is prosody discrimination consistent across train/test split?

### Phase 4: Subtype Characterization
- If clusters exist, profile them:
  - Cluster 1: High f0_std, high energy → "Explosive angry"
  - Cluster 2: Low f0_std, controlled → "Cold angry"
- Check if they match human intuition (listen to samples)

## Decision Criteria

**Use Prosody-Only if:**
- Clear subgroups emerge with 24 actors
- Low within-actor variance, high between-subtype variance
- Features are interpretable and consistent

**Use Hybrid if:**
- Prosody shows no systematic structure with more data
- High within-actor variance persists
- Adding wav2vec2 stabilizes without losing too much separation

**Use Wav2vec2-finetuned if:**
- Neither prosody nor hybrid show clear subtypes
- Best to just fine-tune for standard emotion classification

