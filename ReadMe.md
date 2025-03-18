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

Here is a small snippet of code on how to load and use the model:
```
>>> import os
>>> import glob
>>> import torch
>>> import torchaudio
>>> from SER_Model_setup import SERModel
>>> wav_folder = "my_audio_wavs"
>>> wav_paths = glob.glob(os.path.join(wav_folder, "*.wav"))
>>> device="cuda:1"
>>> 
>>> checkpoint = "ser_checkpoints/best_model.pt"
>>> checkpoint_path = "ser_checkpoints/best_model.pt"
>>> checkpoint = torch.load(checkpoint_path, map_location=device)
<stdin>:1: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
>>> model = SERModel()
>>> model.load_state_dict(checkpoint['model_state_dict'])
<All keys matched successfully>
>>> model.to(device)
SERModel(
  (ssl_model): WavLMModel(
    (feature_extractor): WavLMFeatureEncoder(
      (conv_layers): ModuleList(
        (0): WavLMLayerNormConvLayer(
          (conv): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
          (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (activation): GELU(approximate='none')
        )
        (1-4): 4 x WavLMLayerNormConvLayer(
          (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
          (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (activation): GELU(approximate='none')
        )
        (5-6): 2 x WavLMLayerNormConvLayer(
          (conv): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
          (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (activation): GELU(approximate='none')
        )
      )
    )
    (feature_projection): WavLMFeatureProjection(
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (projection): Linear(in_features=512, out_features=1024, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): WavLMEncoderStableLayerNorm(
      (pos_conv_embed): WavLMPositionalConvEmbedding(
        (conv): Conv1d(1024, 1024, kernel_size=(128,), stride=(1,), padding=(64,), groups=16)
        (activation): GELU(approximate='none')
        (padding): ReplicationPad1d((0, 1))
      )
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (layers): ModuleList(
        (0-23): 24 x WavLMEncoderLayerStableLayerNorm(
          (attention): WavLMAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (gru_rel_pos_linear): Linear(in_features=64, out_features=8, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (feed_forward): WavLMFeedForward(
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
            (intermediate_act_fn): GELU(approximate='none')
            (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (pool_model): AttentiveStatisticsPooling(
    (sap_linear): Linear(in_features=1024, out_features=1024, bias=True)
  )
  (ser_model): EmotionRegression(
    (fc): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=2048, out_features=1024, bias=True)
        (1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
      )
    )
    (out): Sequential(
      (0): Linear(in_features=1024, out_features=3, bias=True)
    )
    (inp_drop): Dropout(p=0.5, inplace=False)
  )
)
>>> model.eval()
SERModel(
  (ssl_model): WavLMModel(
    (feature_extractor): WavLMFeatureEncoder(
      (conv_layers): ModuleList(
        (0): WavLMLayerNormConvLayer(
          (conv): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
          (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (activation): GELU(approximate='none')
        )
        (1-4): 4 x WavLMLayerNormConvLayer(
          (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
          (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (activation): GELU(approximate='none')
        )
        (5-6): 2 x WavLMLayerNormConvLayer(
          (conv): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
          (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (activation): GELU(approximate='none')
        )
      )
    )
    (feature_projection): WavLMFeatureProjection(
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (projection): Linear(in_features=512, out_features=1024, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): WavLMEncoderStableLayerNorm(
      (pos_conv_embed): WavLMPositionalConvEmbedding(
        (conv): Conv1d(1024, 1024, kernel_size=(128,), stride=(1,), padding=(64,), groups=16)
        (activation): GELU(approximate='none')
        (padding): ReplicationPad1d((0, 1))
      )
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (layers): ModuleList(
        (0-23): 24 x WavLMEncoderLayerStableLayerNorm(
          (attention): WavLMAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (gru_rel_pos_linear): Linear(in_features=64, out_features=8, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (feed_forward): WavLMFeedForward(
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
            (intermediate_act_fn): GELU(approximate='none')
            (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (pool_model): AttentiveStatisticsPooling(
    (sap_linear): Linear(in_features=1024, out_features=1024, bias=True)
  )
  (ser_model): EmotionRegression(
    (fc): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=2048, out_features=1024, bias=True)
        (1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
      )
    )
    (out): Sequential(
      (0): Linear(in_features=1024, out_features=3, bias=True)
    )
    (inp_drop): Dropout(p=0.5, inplace=False)
  )
)
>>> waveforms_list = []
>>> lengths = []
>>> for fp in wav_paths:
...     audio, sr = torchaudio.load(fp)
...     if sr != model.sample_rate:
...             resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
...             audio = resampler(audio)
...     if audio.shape[0] > 1:
...             audio = torch.mean(audio, dim=0, keepdim=True)
...     lengths.append(audio.shape[-1])
...     waveforms_list.append(audio)
... 
>>> max_len = max(lengths)
>>> batch_size = len(waveforms_list)
>>> batched_waveforms = torch.zeros(batch_size, 1, max_len, dtype=torch.float32)
>>> masks = torch.zeros(batch_size, max_len, dtype=torch.float32)
>>> for i, audio in enumerate(waveforms_list):
...     cur_len = audio.shape[-1]
...     batched_waveforms[i, :, :cur_len] = audio
...     masks[i, :cur_len] = 1.0  # valid portion
... 
>>> 
>>> batched_waveforms = batched_waveforms.to(device)
>>> masks = masks.to(device)
>>> if normalize:
...     mean = model.mean.to(device)
...     std = model.std.to(device)
...     batched_waveforms = (batched_waveforms - mean) / (std + 1e-6)
... 
>>> 
>>> with torch.no_grad():
...     predictions = model(batched_waveforms, masks)
... 
>>> predictions
tensor([[5.9269, 3.6954, 6.0617],
        [4.5463, 0.9685, 4.9926]], device='cuda:1')
>>> with torch.no_grad():
...     
KeyboardInterrupt
>>> with torch.no_grad():
...     predictions = model(batched_waveforms, masks)
... 
>>> predictions
tensor([[5.9269, 3.6954, 6.0617],
        [4.5463, 0.9685, 4.9926]], device='cuda:1')
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
