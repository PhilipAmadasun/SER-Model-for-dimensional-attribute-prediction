import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa


############################################################
# 1) HELPER FUNCTION: DOWNSAMPLE THE MASK
############################################################

def downsample_mask(mask, strides):
    """
    Downsample a [B, T] or [B, 1, T] mask using the same strides as the feature extractor.
    We treat the mask as 0/1 and do max-pooling so if any frame in the stride window is 1,
    the downsampled time-step is 1.
    """
    # If mask is [B, 1, T], squeeze out that extra dim => [B, T]
    if mask.dim() == 3 and mask.size(1) == 1:
        mask = mask.squeeze(1)  # => [B, T]

    # Now shape => [B, T]. Make it float => [B, 1, T]
    mask = mask.float().unsqueeze(1)

    for s in strides:
        # max_pool1d wants [N, C, L]; kernel_size=s, stride=s
        mask = F.max_pool1d(mask, kernel_size=s, stride=s)

    # Now shape => [B, 1, T_downsampled]
    mask = mask.squeeze(1).bool()  # => [B, T_downsampled]
    return mask


############################################################
# 2) WAVLM FRONT END + PROJECTION + ENCODER
############################################################

class WavLMLayerNormConvLayer(nn.Module):
    """Convolutional layer with layer normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: (B, in_channels, T)
        x = self.conv(x)                 # => (B, out_channels, T')
        x = x.transpose(1, 2)            # => (B, T', out_channels)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)            # => (B, out_channels, T')
        x = self.activation(x)
        return x


class WavLMFeatureEncoder(nn.Module):
    """WavLM feature encoder that converts raw waveform to feature embeddings."""
    def __init__(self):
        super().__init__()
        # 7 conv layers with strides [5, 2, 2, 2, 2, 2, 2]
        self.conv_layers = nn.ModuleList()
        
        # Layer 0: (1->512), kernel=10, stride=5
        self.conv_layers.append(
            WavLMLayerNormConvLayer(1, 512, kernel_size=10, stride=5)
        )
        
        # Layers 1-4: (512->512), kernel=3, stride=2
        for _ in range(4):
            self.conv_layers.append(
                WavLMLayerNormConvLayer(512, 512, kernel_size=3, stride=2)
            )
        
        # Layers 5-6: (512->512), kernel=2, stride=2
        for _ in range(2):
            self.conv_layers.append(
                WavLMLayerNormConvLayer(512, 512, kernel_size=2, stride=2)
            )
    
    def forward(self, x):
        """
        x expected shape: (B, T) or (B, 1, T)
        """
        if x.dim() == 4:  # (B, 1, 1, T)
            x = x.squeeze(2)
        elif x.dim() == 2:  # (B, T)
            x = x.unsqueeze(1)
        
        for layer in self.conv_layers:
            x = layer(x)
        
        # Now shape => (B, 512, T_down)
        return x.transpose(1, 2)  # => (B, T_down, 512)


class WavLMFeatureProjection(nn.Module):
    """Projects features extracted by the encoder to the model dimension"""
    def __init__(self, feature_dim=512, model_dim=1024, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x => (B, T_down, feature_dim=512)
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class WavLMPositionalConvEmbedding(nn.Module):
    """Convolutional positional embeddings for WavLM"""
    def __init__(self, model_dim=1024, kernel_size=128, groups=16):
        super().__init__()
        
        self.conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=True
        )
        self.activation = nn.GELU()
        self.padding = nn.ReplicationPad1d((0, 1))
    
    def forward(self, x):
        # x => (B, T, model_dim)
        x_trans = x.transpose(1, 2)  # => (B, model_dim, T)
        
        pos_emb = self.conv(x_trans)
        if pos_emb.size(-1) != x_trans.size(-1):
            if pos_emb.size(-1) < x_trans.size(-1):
                pos_emb = self.padding(pos_emb)
                if pos_emb.size(-1) > x_trans.size(-1):
                    pos_emb = pos_emb[:, :, :x_trans.size(-1)]
            else:
                pos_emb = pos_emb[:, :, :x_trans.size(-1)]
        
        pos_emb = self.activation(pos_emb)
        pos_emb = pos_emb.transpose(1, 2)  # => (B, T, model_dim)
        
        return pos_emb


class WavLMAttention(nn.Module):
    """Multi-head attention with relative positional embedding"""
    def __init__(self, model_dim=1024, num_heads=16, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        # Example placeholders for relative pos embeddings
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, 16, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(64, 8)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: (B, T_down, model_dim)
        attention_mask: (B, 1, 1, T_down) or None
        """
        B, T_down, _ = hidden_states.shape
        
        keys = self.k_proj(hidden_states)       # => (B, T, model_dim)
        values = self.v_proj(hidden_states)
        queries = self.q_proj(hidden_states)
        
        # Reshape for multi-head
        keys = keys.view(B, T_down, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T_down, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T_down, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attention_scores => (B, num_heads, T, T)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim**0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, values)
        
        # (B, T_down, model_dim)
        context = context.transpose(1, 2).contiguous().view(B, T_down, self.model_dim)
        output = self.out_proj(context)
        
        return output


class WavLMFeedForward(nn.Module):
    """Feed-forward network after attention"""
    def __init__(self, model_dim=1024, intermediate_dim=4096, dropout=0.1):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(0.0)
        self.intermediate_dense = nn.Linear(model_dim, intermediate_dim)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(intermediate_dim, model_dim)
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        
        return hidden_states


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    """Encoder layer with pre-LayerNorm design"""
    def __init__(self, model_dim=1024, intermediate_dim=4096, num_heads=16, dropout=0.1):
        super().__init__()
        self.attention = WavLMAttention(model_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        
        self.feed_forward = WavLMFeedForward(model_dim, intermediate_dim, dropout)
        self.final_layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, hidden_states, attention_mask=None):
        # LN before attention
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # LN before feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class WavLMEncoderStableLayerNorm(nn.Module):
    """WavLM encoder with multiple transformer layers"""
    def __init__(self, model_dim=1024, intermediate_dim=4096, num_layers=24, num_heads=16, dropout=0.1):
        super().__init__()
        
        self.pos_conv_embed = WavLMPositionalConvEmbedding(model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList(
            [WavLMEncoderLayerStableLayerNorm(model_dim, intermediate_dim, num_heads, dropout) 
             for _ in range(num_layers)]
        )
    
    def forward(self, hidden_states, attention_mask=None):
        # Add positional embeddings
        pos_emb = self.pos_conv_embed(hidden_states)
        
        if pos_emb.size(1) != hidden_states.size(1):
            if pos_emb.size(1) > hidden_states.size(1):
                pos_emb = pos_emb[:, :hidden_states.size(1), :]
            else:
                # pad
                pad = torch.zeros_like(hidden_states[:, pos_emb.size(1):, :])
                pos_emb = torch.cat([pos_emb, pad], dim=1)
        
        hidden_states = hidden_states + pos_emb
        hidden_states = self.dropout(hidden_states)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WavLMModel(nn.Module):
    """Complete WavLM model: feature extraction + Transformer encoder"""
    def __init__(self, model_dim=1024, intermediate_dim=4096, num_layers=24, num_heads=16, dropout=0.1):
        super().__init__()
        self.feature_extractor = WavLMFeatureEncoder()
        self.feature_projection = WavLMFeatureProjection(512, model_dim, dropout)
        self.encoder = WavLMEncoderStableLayerNorm(model_dim, intermediate_dim, num_layers, num_heads, dropout)
        
        self.masked_spec_embed = nn.Parameter(torch.zeros(model_dim))
    
    def forward(self, waveform, attention_mask=None):
        # 1) CNN front end => (B, T_down, 512)
        extracted_features = self.feature_extractor(waveform)
        # 2) Project => (B, T_down, model_dim)
        hidden_states = self.feature_projection(extracted_features)
        # 3) Transformer => (B, T_down, model_dim)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states


############################################################
# 3) POOLING & REGRESSION
############################################################

class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling layer"""
    def __init__(self, input_dim=1024):
        super().__init__()
        self.sap_linear = nn.Linear(input_dim, input_dim)
        self.attention = nn.Parameter(torch.zeros(input_dim, 1))
        nn.init.xavier_normal_(self.attention)
    
    def forward(self, x, mask=None):
        # x => (B, T_down, input_dim)
        h = torch.tanh(self.sap_linear(x))        # => (B, T_down, input_dim)
        w = torch.matmul(h, self.attention)       # => (B, T_down, 1)
        
        if mask is not None:
            # mask => (B, T_down, 1), True => valid
            w = w.masked_fill(~mask, -1e9)
        
        w = F.softmax(w, dim=1)  # => (B, T_down, 1)
        
        mean = torch.sum(x * w, dim=1)
        var = torch.sum((x**2) * w, dim=1) - mean**2
        
        eps = 1e-5
        std = torch.sqrt(var + eps)
        
        # => (B, 2*input_dim)
        return torch.cat([mean, std], dim=1)


class EmotionRegression(nn.Module):
    """Regression module for emotional attributes prediction"""
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=3, dropout=0.5):
        super().__init__()
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        self.out = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.inp_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # x => (B, 2*model_dim=2048)
        x = self.inp_drop(x)
        for layer in self.fc:
            x = layer(x)
        x = self.out(x)
        return x


############################################################
# 4) THE SER MODEL WITH TRIM/PAD FOR THE MASK
############################################################

class SERModel(nn.Module):
    """Complete Speech Emotion Recognition model"""
    
    def __init__(self, model_dim=1024, intermediate_dim=4096, num_layers=24, num_heads=16, dropout=0.1):
        super().__init__()
        
        # WavLM-based front end + Transformer
        self.ssl_model = WavLMModel(model_dim, intermediate_dim, num_layers, num_heads, dropout)
        
        # ---------------------------------------------------------
        # 1) Freeze WavLM layers
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        # ---------------------------------------------------------
        
        # Attentive pooling
        self.pool_model = AttentiveStatisticsPooling(model_dim)
        
        # Final regression
        self.ser_model = EmotionRegression(2 * model_dim, model_dim, 3, dropout=0.5)
        
        # Buffers to hold mean/std for normalization
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))
        
        # Sample rate for inference
        self.sample_rate = 16000
        
        # Must match the 7 convolution layers in WavLMFeatureEncoder
        self.downsample_strides = [5, 2, 2, 2, 2, 2, 2]
    
    def forward(self, waveform, attention_mask=None):
        """
        Forward pass with optional attention_mask. We:
          1) Extract features => (B, T_down, 512)
          2) Downsample the mask => (B, T_down), fix any mismatch
          3) Convert to float_mask_for_attention => (B, 1, 1, T_down)
             and pooling_mask => (B, T_down, 1)
          4) Pass through WavLM encoder => (B, T_down, model_dim)
          5) Pool => (B, 2*model_dim)
          6) Regress => (B, 3)
        """
        # 1) Extract CNN features
        extracted_features = self.ssl_model.feature_extractor(waveform)
        T_down = extracted_features.size(1)  # actual frames after conv
        
        # Prepare attention & pooling masks
        float_mask_for_attention = None
        pooling_mask = None
        
        if attention_mask is not None:
            # 2) Downsample => (B, T_approx)
            ds_mask = downsample_mask(attention_mask, self.downsample_strides)
            
            # Fix final mismatch if needed
            M = ds_mask.size(1)
            if M > T_down:
                # Trim
                ds_mask = ds_mask[:, :T_down]
            elif M < T_down:
                # Pad with zeros => 'invalid' for extra frames
                pad_len = T_down - M
                pad = torch.zeros(ds_mask.size(0), pad_len, device=ds_mask.device, dtype=ds_mask.dtype)
                ds_mask = torch.cat([ds_mask, pad], dim=1)
            
            # Now ds_mask => (B, T_down)
            # 3) Build float_mask => (B, 1, 1, T_down)
            #    Valid=1 => 0.0, Invalid=0 => -1e9
            float_mask_for_attention = (1.0 - ds_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            
            # For pooling => (B, T_down, 1), True=valid
            pooling_mask = ds_mask.unsqueeze(-1)  # => (B, T_down, 1), bool or 0/1 is fine
            # If ds_mask is float, convert to bool => ds_mask > 0
            pooling_mask = pooling_mask.bool()
        
        # 4) Pass through feature projection + transformer
        hidden_states = self.ssl_model.feature_projection(extracted_features)
        hidden_states = self.ssl_model.encoder(hidden_states, float_mask_for_attention)
        
        # 5) Pool
        pooled_output = self.pool_model(hidden_states, pooling_mask)
        
        # 6) Regression
        emotion_scores = self.ser_model(pooled_output)
        return emotion_scores
    
    @torch.no_grad()
    def predict_from_file(self, file_path, normalize=True):
        """
        Predict emotional attributes from a .wav file
        """
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if normalize:
            waveform = (waveform - self.mean) / (self.std + 1e-6)
        
        mask = torch.ones(1, waveform.shape[1], device=waveform.device)  # (1, T)
        
        self.eval()
        predictions = self.forward(waveform, mask)  # => (1, 3)
        
        return {
            'arousal': predictions[0, 0].item(),
            'valence': predictions[0, 1].item(),
            'dominance': predictions[0, 2].item()
        }


############################################################
# 5) DATASET + TRAINING (UNCHANGED EXCEPT FOR YOUR USE)
############################################################

class SERDataset(Dataset):
    """Dataset for Speech Emotion Recognition"""
    def __init__(self, data_dir, file_list_path, mean=0.0, std=1.0, max_length=None):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.max_length = max_length
        
        with open(file_list_path, 'r') as f:
            self.file_list = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = torch.load(file_path, weights_only=False)
        
        waveform = data['waveform']
        # Squeeze if needed
        if waveform.dim() > 2:
            waveform = waveform.squeeze()
            if waveform.dim() == 0:
                waveform = waveform.unsqueeze(0)
        
        # Make shape => (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # (1, T)
        mask = torch.ones_like(waveform)
        
        # Trim/pad if max_length
        if self.max_length is not None:
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
                mask = mask[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                pad_amt = self.max_length - waveform.shape[1]
                waveform = torch.cat([waveform, torch.zeros(1, pad_amt)], dim=1)
                mask = torch.cat([mask, torch.zeros(1, pad_amt)], dim=1)
        
        # Normalize
        waveform = (waveform - self.mean) / (self.std + 1e-6)
        
        labels = torch.tensor([
            data['labels']['arousal'],
            data['labels']['valence'],
            data['labels']['dominance']
        ], dtype=torch.float32)
        
        return waveform, mask, labels


def ccc_loss(predictions, targets, epsilon=1e-6):
    pred_mean = torch.mean(predictions, dim=0)
    target_mean = torch.mean(targets, dim=0)
    pred_var = torch.var(predictions, dim=0, unbiased=False)
    target_var = torch.var(targets, dim=0, unbiased=False)
    
    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean
    covariance = torch.mean(pred_centered * target_centered, dim=0)
    
    numerator = 2 * covariance
    denominator = pred_var + target_var + (pred_mean - target_mean)**2 + epsilon
    ccc = numerator / denominator
    
    return torch.mean(1 - ccc)


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for waveforms, masks, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)'):
            waveforms = waveforms.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            predictions = model(waveforms, masks)
            loss = ccc_loss(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for waveforms, masks, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Val)'):
                waveforms = waveforms.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                predictions = model(waveforms, masks)
                
                ccc_val = ccc_loss(predictions, labels)
                mse_val = F.mse_loss(predictions, labels)
                
                val_loss += ccc_val.item()
                val_mse += mse_val.item()
        
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss (1-CCC): {train_loss:.4f}")
        print(f"  Val Loss (1-CCC):   {val_loss:.4f}")
        print(f"  Val MSE:            {val_mse:.4f}")
        
        ckpt_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'history': history
        }, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mse': val_mse,
                'history': history
            }, best_model_path)
            print(f"  *** New best model saved to {best_model_path} ***")
    
    return model, history


def load_pretrained_wavlm(model, wavlm_path):
    """
    Load pretrained WavLM weights into the SSL model
    """
    pretrained_dict = torch.load(wavlm_path, map_location='cpu')
    ssl_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('ssl_model.'):
            ssl_dict[k[10:]] = v
    model.ssl_model.load_state_dict(ssl_dict, strict=False)
    return model
