# Pytorch SER Model

This repository contains a Speech Emotion Recognition (SER) model inspired by [3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes). The model is implemented in PyTorch and supports batch inference.

## Dataset
The current checkpoints have been trained on a small subset of the [MSP-Podcast dataset](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html).

## Project Goals
The main objectives of this project are:
- ✅ Direct loading with PyTorch for greater flexibility in development.  
- ✅ Defined class structure with a feedforward method for easier integration.  
- 🔄 Improved prediction accuracy for **valence** and **arousal**.
- compile model to tensorRT for optimized use on edge device (i.e nvidia Orin or AGX)

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
## Compiling model to TensorRT
To my knowledge, the safest route to model compiling for models that accept inputs of varied length is to first convert to onnx. From my cursory research, torch is sometimes bad at figuring out if a input is a safe size to work with, even if you provide the maximum, minimum and general size of input the model would process. This will lead to errors. For example:
Using **torch_tensorrt** for compiling 
```python
import torch_tensorrt
import torch
from SER_Model_setup import SERModel

model = SERModel().half().eval().cuda()
model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])

# ── shape profiles ─────────────────────────────────────────
# Both tensors have shape (B=1,  T) where T is time‑samples.
# Keep T identical in min/opt/max for *both* inputs so TRT
# can fuse them into one optimization profile.
#
profiles = [
    torch_tensorrt.Input(                       # waveform
        min_shape=(1, 16000),
        opt_shape=(1, 80000),
        max_shape=(1,160000),
        dtype=torch.half),
    torch_tensorrt.Input(                       # mask
        min_shape=(1, 16000),
        opt_shape=(1, 80000),
        max_shape=(1,160000),
        dtype=torch.float)          # mask stays fp32/bool works too
]

trt_mod = torch_tensorrt.compile(
            model,
            inputs=profiles,
            enabled_precisions={torch.half},
            workspace_size=2<<30)              # 2 GB
torch.jit.save(trt_mod, "ser_trt_mask.ts")      # binary embeds TRT engine                                   
```
So the sample rate of audio this model processes will be at 16000 Hz (1 second), I basically provide a range of 1 sec to 10 sec (160000 Hz). I provide a kind of general length around 80000 Hz (5 seconds) which the model would mostly handle (or around that size).
I ran into errors like this:
```
trt_mod = torch_tensorrt.compile(
...             model,
...             inputs=profiles,
...             enabled_precisions={torch.half},
...             workspace_size=2<<30) 
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0] Error while creating guard:
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0] Name: ''
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     Source: shape_env
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     Create Function: SHAPE_ENV
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     Guard Types: None
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     Code List: None
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     Object Weakref: None
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     Guarded Class Weakref: None
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0] Traceback (most recent call last):
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]   File "/home/uamadasun/.venv/ser/lib/python3.12/site-packages/torch/_guards.py", line 293, in create
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     return self.create_fn(builder, self)
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]   File "/home/uamadasun/.venv/ser/lib/python3.12/site-packages/torch/_dynamo/guards.py", line 1868, in SHAPE_ENV
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]     code_parts, verbose_code_parts = output_graph.shape_env.produce_guards_verbose(
E0419 20:53:58.731000 1283699 torch/_guards.py:295] [0/0]

..............................
torch._dynamo.exc.UserError: Constraints violated (_1)! For more information, run with TORCH_LOGS="+dynamic".
  - Not all values of _1 = L['waveform'].size()[1] in the specified range 16000 <= _1 <= 160000 are valid because _1 was inferred to be a constant (80000).
  - Not all values of _1 = L['attention_mask'].size()[1] in the specified range 16000 <= _1 <= 160000 are valid because _1 was inferred to be a constant (80000).

Suggested fixes:
  _1 = 80000           
```
Where basically the more stable solution is to just set the size to 80000 leaving the compiled model unable to handle varied inout size.

The onnx framework is more suited for handling the computation around this issue. It's more suited for 

### Conversion to onnx:

```python
import torch, torch.onnx
from SER_Model_setup import SERModel

device = "cuda"
model = SERModel().to(device).eval()

ckpt = torch.load("ser_checkpoints/best_weights.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)

wav  = torch.randn(1, 32000, device=device)
mask = torch.ones (1, 32000, device=device)

torch.onnx.export(
    model, (wav, mask), "ser_dyn.onnx",
    opset_version=17,
    input_names = ["waveform","mask"],
    output_names= ["scores"],
    dynamic_axes={"waveform":{1:"time"}, "mask":{1:"time"}})

print("✓  ser_dyn.onnx regenerated with trained weights")
```
I compare the results of the original torch and onnx models to make sure I'm getting around the same predictions and precision.
```python
#!/usr/bin/env python3
# test_ser_onnx.py  – compare PyTorch vs. ONNX‑Runtime (CUDA)
# ---------------------------------------------------------------
import sys, pathlib, os, datetime, numpy as np, torch, torchaudio, onnx, onnxruntime as ort
from SER_Model_setup import SERModel

# ---------- CLI arg --------------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python test_ser_onnx.py <audio.wav>")
    sys.exit(1)
wav_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
assert wav_path.is_file(), f"{wav_path} not found"

# ---------- paths ----------------------------------------------------------
ckpt_path = pathlib.Path("ser_checkpoints/best_weights.pt")
onnx_path = pathlib.Path("ser_dyn.onnx")

# ---------- 0.  (Re)‑export ONNX if needed --------------------------------
def onnx_outdated(onnx_p: pathlib.Path, ckpt_p: pathlib.Path) -> bool:
    if not onnx_p.exists():
        return True
    return onnx_p.stat().st_mtime < ckpt_p.stat().st_mtime   # older than ckpt

if onnx_outdated(onnx_path, ckpt_path):
    print("🔄  Exporting ONNX with trained weights …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SERModel().to(device).eval()
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    wav_dummy  = torch.randn(1, 32_000, device=device)
    mask_dummy = torch.ones (1, 32_000, device=device)

    torch.onnx.export(
        model, (wav_dummy, mask_dummy), onnx_path.as_posix(),
        opset_version=17,
        input_names = ["waveform","mask"],
        output_names= ["scores"],
        dynamic_axes={"waveform":{1:"time"}, "mask":{1:"time"}})
    print(f"✅  {onnx_path} written "
          f"({datetime.datetime.now().strftime('%H:%M:%S')})")
else:
    print("ℹ️  Using existing ONNX (already newer than checkpoint)")

# ---------- 1.  PyTorch model on CUDA --------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pt_model  = SERModel().to(device).eval()
ckpt      = torch.load(ckpt_path, map_location=device)
pt_model.load_state_dict(ckpt["model_state_dict"], strict=False)
mean, std = pt_model.mean.to(device), pt_model.std.to(device)
sr_target = pt_model.sample_rate                                   # 16 kHz

# ---------- 2.  Load & preprocess WAV --------------------------------------
wave, sr = torchaudio.load(wav_path)
if sr != sr_target:
    wave = torchaudio.functional.resample(wave, sr, sr_target)
if wave.shape[0] > 1:                                             # stereo→mono
    wave = wave.mean(0, keepdim=True)
wave = wave.to(device)
wave = (wave - mean) / (std + 1e-6)
T    = wave.shape[1]
mask = torch.ones(1, T, device=device)

# ---------- 3.  PyTorch inference ------------------------------------------
with torch.no_grad():
    pt_scores = pt_model(wave, mask).cpu().numpy()[0]

# ---------- 4.  ONNX‑Runtime (CUDA EP) -------------------------------------
cuda_ep = [("CUDAExecutionProvider", {"device_id": 0})]
sess    = ort.InferenceSession(onnx_path.as_posix(),
                               providers=cuda_ep + ["CPUExecutionProvider"])
inputs  = {"waveform": wave.cpu().numpy().astype(np.float32),
           "mask":     mask.cpu().numpy().astype(np.float32)}
ort_scores = sess.run(["scores"], inputs)[0][0]

# ---------- 5.  Compare & print --------------------------------------------
print("\n=== Emotion scores ===")
print(f"PyTorch (GPU): {pt_scores}")
print(f"ONNX‑RT (GPU): {ort_scores}")
print(f"max |Δ|       : {(np.abs(pt_scores - ort_scores)).max():.6f}")
```

Looking decent so far
```
./test_ser_onnx.py my_audio_wavs/test_audio.wav
=== Emotion scores ===
PyTorch (GPU): [5.9269137 3.6953883 6.0616913]
ONNX‑RT (GPU): [5.9269595 3.6959462 6.061699 ]
max |Δ|       : 0.000558

./test_ser_onnx.py my_audio_wavs/test_audio1.wav
=== Emotion scores ===
PyTorch (GPU): [4.5547924 0.8715324 5.0094886]
ONNX‑RT (GPU): [4.5539904  0.87481815 5.008607  ]
max |Δ|       : 0.003286
```

## Future Work
* Integrate a **Density Adaptive Attention Block** before or after the transformer layers to explore potential performance improvements.
* Test feature extraction via Log Mel Spectrogram instead if pretrain WavLM SSL layers for lighter overhead, might need to test with DAAM to improve or atleast mitigate accuracy.

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
