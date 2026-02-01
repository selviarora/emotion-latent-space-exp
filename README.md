# emotion latent space experiment

probing wav2vec2's internal layers to see if emotion is an emergent geometric property of its latent space

---

## what is this

so i had this idea - wav2vec2 is trained on raw audio with no labels at all, just self-supervised learning. but when you look at the embeddings it produces... emotional speech clusters together. angry samples end up near other angry samples. calm near calm.

which got me thinking: is there a direction in this space that corresponds to emotion? like could you literally just do `embedding + anger_direction` and make speech sound angrier?

turns out: yes. and it works way better than i expected.

## how it works

i took the CREMA-D dataset (actors saying the same sentences with different emotions) and extracted wav2vec2 embeddings from layer 5. for each actor, i computed `angry_embedding - calm_embedding`. then averaged all those difference vectors together.

that gives you one 768-dim vector - the "emotion axis".

now for any new audio:
1. extract its wav2vec2 embedding
2. add some multiple of the emotion axis to it
3. run through a mapper network i trained (converts embeddings back to mel spectrograms)
4. vocode with hifi-gan

and it actually shifts the emotion. not perfectly - there's some audio quality loss - but you can clearly hear the difference.

## whats in here

the important files:
- `tts.py` - text to speech with emotion blending
- `scripts/core/emotion_steer_final.py` - the actual steering code
- `scripts/core/train_mapper_final.py` - training the embedding→mel mapper
- `scripts/wav2vec2_experiments/build_global_axis.py` - builds the emotion axis

pretrained stuff in `models/`:
- `emotion_axis_layer5.npy` - the emotion direction (768 floats)
- `hifigan/` - vocoder weights

data:
- `crema_d_data/` - the dataset
- `embeddings/` - precomputed wav2vec features
- `checkpoints/` - mapper model weights

## usage

steer existing audio:
```bash
python scripts/core/emotion_steer_final.py --input voice.wav --delta 0.5 --output out.wav
```
delta goes from -1 (calmer) to +1 (angrier)

tts with emotion control:
```bash
python tts.py --text "whatever" --ref_calm calm.wav --ref_angry angry.wav --alpha 0.3 --out out.wav
```

## interesting findings

- layer 5 works best. earlier layers dont have enough semantic info, later layers hurt audio quality
- the axis generalizes across speakers - even ones the model never saw during axis construction
- you can interpolate smoothly along the axis, its not just binary
- the emotion structure emerges purely from self-supervised pretraining on unlabeled audio

## dependencies

torch, transformers, librosa, soundfile, TTS (coqui), numpy

## todo

- extend to more emotions beyond calm/angry
- improve audio quality of reconstructions
- try hubert and wavlm
- perceptual study
