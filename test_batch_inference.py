import os
import glob
import torch
import torchaudio
from SER_Model_setup import SERModel  # Adjust if your model code is elsewhere

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Loads the SERModel and weights from a checkpoint, moves to device, sets eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create the model architecture
    model = SERModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def batch_inference(model, file_paths, device='cpu', normalize=True):
    """
    Perform true batch inference on multiple .wav files in one forward pass.
    
    Args:
        model (SERModel): The loaded SER model in eval mode
        file_paths (list[str]): List of paths to .wav files
        device (str or torch.device): 'cpu' or 'cuda'
        normalize (bool): Whether to normalize waveforms (subtract mean, divide std)
    
    Returns:
        dict: {filename: {"arousal": float, "valence": float, "dominance": float}}
    """

    # ----------------------------------------
    # 1) Load & store all waveforms in memory
    # ----------------------------------------
    waveforms_list = []
    lengths = []
    for fp in file_paths:
        # Load audio
        audio, sr = torchaudio.load(fp)
        
        # Resample if needed
        if sr != model.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, model.sample_rate)
            audio = resampler(audio)
        
        # Convert stereo -> mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # audio shape => [1, num_samples]
        lengths.append(audio.shape[-1])
        waveforms_list.append(audio)
    
    # ----------------------------------------
    # 2) Determine max length
    # ----------------------------------------
    max_len = max(lengths)
    
    # ----------------------------------------
    # 3) Pad each waveform to max length & build masks
    # ----------------------------------------
    batch_size = len(waveforms_list)
    batched_waveforms = torch.zeros(batch_size, 1, max_len, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, audio in enumerate(waveforms_list):
        cur_len = audio.shape[-1]
        batched_waveforms[i, :, :cur_len] = audio
        masks[i, :cur_len] = 1.0  # valid portion

    # ----------------------------------------
    # 4) Move batched data to device BEFORE normalization
    # ----------------------------------------
    batched_waveforms = batched_waveforms.to(device)
    masks = masks.to(device)
    
    # ----------------------------------------
    # 5) Normalize if needed (model.mean, model.std)
    # ----------------------------------------
    if normalize:
        # model.mean and model.std are buffers; ensure they're on the correct device
        mean = model.mean.to(device)
        std = model.std.to(device)
        batched_waveforms = (batched_waveforms - mean) / (std + 1e-6)
    
    # ----------------------------------------
    # 6) Single forward pass
    # ----------------------------------------
    with torch.no_grad():
        predictions = model(batched_waveforms, masks)
        # predictions shape => [batch_size, 3]
    
    # ----------------------------------------
    # 7) Build result dict
    # ----------------------------------------
    results = {}
    for i, fp in enumerate(file_paths):
        arousal   = predictions[i, 0].item()
        valence   = predictions[i, 1].item()
        dominance = predictions[i, 2].item()
        filename = os.path.basename(fp)
        results[filename] = {
            "arousal": arousal,
            "valence": valence,
            "dominance": dominance
        }
    
    return results

if __name__ == "__main__":
    # -----------------------------------------
    # Example usage
    # -----------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = "ser_checkpoints/best_model.pt"
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    
    # Suppose you have a folder of .wav files
    wav_folder = "my_audio_wavs"
    wav_paths = glob.glob(os.path.join(wav_folder, "*.wav"))
    
    # Do a single pass of batch inference
    all_results = batch_inference(model, wav_paths, device=device, normalize=True)
    
    # Print results
    for fname, preds in all_results.items():
        print(f"{fname}: Arousal={preds['arousal']:.3f}, "
              f"Valence={preds['valence']:.3f}, Dominance={preds['dominance']:.3f}")
