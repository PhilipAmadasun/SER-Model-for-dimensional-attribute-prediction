# Pytorch SER Model

This repository contains a Speech Emotion Recognition (SER) model inspired by [3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes). The model is implemented in PyTorch and supports batch inference.

## Dataset
The current checkpoints have been trained on a small subset of the [MSP-Podcast dataset](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html).

## Project Goals
The main objectives of this project are:
- ✅ Direct loading with PyTorch for greater flexibility in development.  
- ✅ Defined class structure with a feedforward method for easier integration.  
- 🔄 Improved prediction accuracy for **valence** and **arousal**.  

### Library Versions
The following library versions were used for this work:
-    torch                             2.5.1
-    torchaudio                        2.5.1
-    torchvision                       0.20.1
  
## Training Details
- The model was trained using **Concordance Correlation Coefficient (CCC) loss**, as in the original implementation.
- Evaluation metrics will be provided soon.
- The best-performing checkpoint is available—feel free to test it out and provide feedback!

## Usage Example (Jupyter Notebook Format)
To use the model in a Jupyter Notebook, run the following code:

```python
import os
import glob
import torch
import torchaudio
from SER_Model_setup import SERModel

# Define folder path and get list of wav files
wav_folder = "my_audio_wavs"
wav_paths = glob.glob(os.path.join(wav_folder, "*.wav"))

device = "cuda:1"

# Load checkpoint
checkpoint_path = "ser_checkpoints/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Initialize and load model
model = SERModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load and preprocess audio files
waveforms_list = []
lengths = []
for fp in wav_paths:
    audio, sr = torchaudio.load(fp)
    if sr != model.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
        audio = resampler(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    lengths.append(audio.shape[-1])
    waveforms_list.append(audio)

# Create batched waveforms and masks
max_len = max(lengths)
batch_size = len(waveforms_list)

batched_waveforms = torch.zeros(batch_size, 1, max_len, dtype=torch.float32)
masks = torch.zeros(batch_size, max_len, dtype=torch.float32)

for i, audio in enumerate(waveforms_list):
    cur_len = audio.shape[-1]
    batched_waveforms[i, :, :cur_len] = audio
    masks[i, :cur_len] = 1.0  # valid portion

batched_waveforms = batched_waveforms.to(device)
masks = masks.to(device)

# Normalize if required
normalize = True  # Change this if normalization is not needed
if normalize:
    mean = model.mean.to(device)
    std = model.std.to(device)
    batched_waveforms = (batched_waveforms - mean) / (std + 1e-6)

# Run inference
with torch.no_grad():
    predictions = model(batched_waveforms, masks)

# Print predictions
print(predictions)
```

## Future Work
I plan to integrate a **Density Adaptive Attention Block** before or after the transformer layers to explore potential performance improvements.

## Reference
If you're interested in Density Adaptive Attention, check out the following paper:
```bibtex
@article{ioannides2024density,
  title={Density Adaptive Attention is All You Need: Robust Parameter-Efficient Fine-Tuning Across Multiple Modalities},
  author={Ioannides, Georgios and Chadha, Aman and Elkins, Aaron},
  journal={arXiv preprint arXiv:2401.11143},
  year={2024}
}
```
