"""
Custom Indonesian License Plate OCR Recognizer
Fine-tuned from deep-text-recognition-benchmark TPS-ResNet-BiLSTM-Attn model.

This module provides a custom OCR model optimized for Indonesian license plates.
It can be used as a drop-in replacement for EasyOCR in the ALPR pipeline.

Usage:
    from custom_plate_ocr import IndonesianPlateOCR
    ocr = IndonesianPlateOCR("indonesian_plate_ocr_model")
    text = ocr.recognize(image_np)
"""

import os
import json
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for custom OCR")


# ============================================================================
# Model Architecture Components (TPS-ResNet-BiLSTM-Attn)
# Based on: https://github.com/clovaai/deep-text-recognition-benchmark
# ============================================================================

if TORCH_AVAILABLE:
    
    class TPS_SpatialTransformerNetwork(nn.Module):
        """Thin Plate Spline Spatial Transformer Network for text rectification."""
        
        def __init__(self, F, I_size, I_r_size, I_channel_num=1):
            super(TPS_SpatialTransformerNetwork, self).__init__()
            self.F = F
            self.I_size = I_size
            self.I_r_size = I_r_size
            self.I_channel_num = I_channel_num
            self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
            self.GridGenerator = GridGenerator(self.F, self.I_r_size)

        def forward(self, batch_I):
            batch_C_prime = self.LocalizationNetwork(batch_I)
            build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
            batch_P_prime = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
            batch_I_r = F.grid_sample(batch_I, batch_P_prime, padding_mode='border', align_corners=True)
            return batch_I_r


    class LocalizationNetwork(nn.Module):
        """Localization Network for TPS."""
        
        def __init__(self, F, I_channel_num):
            super(LocalizationNetwork, self).__init__()
            self.F = F
            self.I_channel_num = I_channel_num
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1)
            )
            self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
            self.localization_fc2 = nn.Linear(256, self.F * 2)
            
            # Initialize with identity transformation
            self.localization_fc2.weight.data.fill_(0)
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
            ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

        def forward(self, batch_I):
            batch_size = batch_I.size(0)
            features = self.conv(batch_I).view(batch_size, -1)
            batch_C_prime = self.localization_fc2(self.localization_fc1(features))
            return batch_C_prime.view(batch_size, self.F, 2)


    class GridGenerator(nn.Module):
        """Grid Generator for TPS transformation."""
        
        def __init__(self, F, I_r_size):
            super(GridGenerator, self).__init__()
            self.eps = 1e-6
            self.I_r_height, self.I_r_width = I_r_size
            self.F = F
            self.C = self._build_C(self.F)
            self.P = self._build_P(self.I_r_width, self.I_r_height)
            self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())
            self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())

        def _build_C(self, F):
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = -1 * np.ones(int(F / 2))
            ctrl_pts_y_bottom = np.ones(int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            return C

        def _build_inv_delta_C(self, F, C):
            hat_C = np.zeros((F, F), dtype=float)
            for i in range(0, F):
                for j in range(i, F):
                    r = np.linalg.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
            np.fill_diagonal(hat_C, 1)
            hat_C = (hat_C ** 2) * np.log(hat_C)
            delta_C = np.concatenate([
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)
            ], axis=0)
            inv_delta_C = np.linalg.inv(delta_C)
            return inv_delta_C

        def _build_P(self, I_r_width, I_r_height):
            I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
            I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
            P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
            return P.reshape([-1, 2])

        def _build_P_hat(self, F, C, P):
            n = P.shape[0]
            P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
            C_tile = np.expand_dims(C, axis=0)
            P_diff = P_tile - C_tile
            rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
            rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
            P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
            return P_hat

        def build_P_prime(self, batch_C_prime):
            batch_size = batch_C_prime.size(0)
            batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
            batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
            batch_C_prime_with_zeros = torch.cat(
                (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(batch_C_prime.device)), dim=1
            )
            batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
            batch_P_prime = torch.bmm(batch_P_hat, batch_T)
            return batch_P_prime


    class ResNet_FeatureExtractor(nn.Module):
        """ResNet-based Feature Extractor for OCR."""
        
        def __init__(self, input_channel, output_channel=512):
            super(ResNet_FeatureExtractor, self).__init__()
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, 32, 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(256, output_channel, 3, 1, 1, bias=False), nn.BatchNorm2d(output_channel), nn.ReLU(True),
            )

        def forward(self, input):
            return self.ConvNet(input)


    class BidirectionalLSTM(nn.Module):
        """Bidirectional LSTM for sequence modeling."""
        
        def __init__(self, input_size, hidden_size, output_size):
            super(BidirectionalLSTM, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input):
            self.rnn.flatten_parameters()
            recurrent, _ = self.rnn(input)
            output = self.linear(recurrent)
            return output


    class Attention(nn.Module):
        """Attention-based sequence decoder."""
        
        def __init__(self, input_size, hidden_size, num_classes):
            super(Attention, self).__init__()
            self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            self.generator = nn.Linear(hidden_size, num_classes)

        def _char_to_onehot(self, input_char, onehot_dim):
            input_char = input_char.unsqueeze(1)
            batch_size = input_char.size(0)
            one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(input_char.device)
            one_hot = one_hot.scatter_(1, input_char, 1)
            return one_hot

        def forward(self, batch_H, text, is_train=True, batch_max_length=25):
            batch_size = batch_H.size(0)
            num_steps = batch_max_length + 1

            output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(batch_H.device)
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(batch_H.device),
                torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(batch_H.device)
            )

            if is_train:
                for i in range(num_steps):
                    char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                    output_hiddens[:, i, :] = hidden[0]
                probs = self.generator(output_hiddens)
            else:
                targets = torch.LongTensor(batch_size).fill_(0).to(batch_H.device)
                probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(batch_H.device)

                for i in range(num_steps):
                    char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                    probs_step = self.generator(hidden[0])
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input

            return probs


    class AttentionCell(nn.Module):
        """Single attention cell."""
        
        def __init__(self, input_size, hidden_size, num_embeddings):
            super(AttentionCell, self).__init__()
            self.i2h = nn.Linear(input_size, hidden_size, bias=False)
            self.h2h = nn.Linear(hidden_size, hidden_size)
            self.score = nn.Linear(hidden_size, 1, bias=False)
            self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, prev_hidden, batch_H, char_onehots):
            batch_H_proj = self.i2h(batch_H)
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
            e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))
            alpha = F.softmax(e, dim=1)
            context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
            concat_context = torch.cat([context, char_onehots], 1)
            cur_hidden = self.rnn(concat_context, prev_hidden)
            return cur_hidden, alpha


    class OCRModel(nn.Module):
        """Complete TPS-ResNet-BiLSTM-Attn OCR Model."""
        
        def __init__(self, opt):
            super(OCRModel, self).__init__()
            self.opt = opt

            # Transformation (TPS)
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, 
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW), 
                I_channel_num=opt.input_channel
            )

            # Feature Extraction (ResNet)
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
            self.FeatureExtraction_output = opt.output_channel
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

            # Sequence Modeling (BiLSTM)
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size)
            )
            self.SequenceModeling_output = opt.hidden_size

            # Prediction (Attention)
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

        def forward(self, input, text, is_train=True):
            # Transformation
            input = self.Transformation(input)
            # Feature Extraction
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
            visual_feature = visual_feature.squeeze(3)
            # Sequence Modeling
            contextual_feature = self.SequenceModeling(visual_feature)
            # Prediction
            prediction = self.Prediction(
                contextual_feature.contiguous(), text, is_train, 
                batch_max_length=self.opt.batch_max_length
            )
            return prediction


# ============================================================================
# Main OCR Class for Inference
# ============================================================================

class AttrDict(dict):
    """Dictionary that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class IndonesianPlateOCR:
    """
    Indonesian License Plate OCR using fine-tuned TPS-ResNet-BiLSTM-Attn model.
    
    Usage:
        ocr = IndonesianPlateOCR("path/to/model_dir")
        text = ocr.recognize(image_np)  # image_np is BGR or grayscale numpy array
    """
    
    def __init__(self, model_dir, device=None):
        self.model_dir = model_dir
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.opt = None
        self.character = None
        self.is_loaded = False
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available - custom OCR model cannot be loaded")
            return
            
        if not os.path.exists(model_dir):
            print(f"Custom OCR model directory not found: {model_dir}")
            return
            
        try:
            self._load_model()
            self.is_loaded = True
            print(f"✓ Loaded Indonesian Plate OCR model on {self.device}")
        except Exception as e:
            print(f"Error loading custom OCR model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def _load_model(self):
        """Load the fine-tuned model from disk."""
        # Load config
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Build character list with special tokens
        base_chars = config.get('character', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')
        self.character = ' [s]' + base_chars  # [GO]=space, [s]=end token
        
        # Model options
        self.opt = AttrDict({
            'Transformation': config.get('Transformation', 'TPS'),
            'FeatureExtraction': config.get('FeatureExtraction', 'ResNet'),
            'SequenceModeling': config.get('SequenceModeling', 'BiLSTM'),
            'Prediction': config.get('Prediction', 'Attn'),
            'num_fiducial': config.get('num_fiducial', 20),
            'input_channel': config.get('input_channel', 1),
            'output_channel': config.get('output_channel', 512),
            'hidden_size': config.get('hidden_size', 256),
            'imgH': config.get('imgH', 32),
            'imgW': config.get('imgW', 100),
            'batch_max_length': config.get('batch_max_length', 15),
            'num_class': len(self.character),
        })
        
        # Build model
        self.model = OCRModel(self.opt)
        
        # Load weights
        model_path = os.path.join(self.model_dir, "best_accuracy.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle DataParallel state dict (remove 'module.' prefix)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, image):
        """
        Preprocess image for the model.
        
        Args:
            image: numpy array (BGR or grayscale)
            
        Returns:
            torch.Tensor: preprocessed image tensor
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        h, w = self.opt.imgH, self.opt.imgW
        image = cv2.resize(image, (w, h))
        
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        # Add batch and channel dimensions: (1, 1, H, W)
        image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        return image.to(self.device)
    
    def recognize(self, image):
        """
        Recognize text in a license plate image.
        
        Args:
            image: numpy array (BGR or grayscale)
            
        Returns:
            str: Recognized text
        """
        if not self.is_loaded:
            return ""
            
        if image is None or image.size == 0:
            return ""
        
        try:
            input_tensor = self.preprocess(image)
            
            # Create dummy text tensor for inference
            text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(input_tensor, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                return self._decode(preds_index[0])
                
        except Exception as e:
            print(f"OCR recognition error: {e}")
            return ""
    
    def _decode(self, pred_index):
        """
        Decode prediction indices to string.
        
        Args:
            pred_index: tensor of predicted character indices
            
        Returns:
            str: decoded text
        """
        pred_str = ''
        for idx in pred_index:
            idx = int(idx)
            if idx == 0:  # [GO] token (space at index 0)
                continue
            if idx < len(self.character) and self.character[idx] == '[':
                # Check for [s] end token
                if idx + 2 < len(self.character) and self.character[idx:idx+3] == '[s]':
                    break
            if idx < len(self.character):
                char = self.character[idx]
                if char not in ['[', 's', ']']:  # Skip end token chars
                    pred_str += char
        
        return pred_str.strip()


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        print(f"Testing IndonesianPlateOCR with model from: {model_dir}")
        ocr = IndonesianPlateOCR(model_dir)
        
        if ocr.is_loaded:
            print("✓ Model loaded successfully!")
            print(f"  Device: {ocr.device}")
            print(f"  Characters: {len(ocr.character)}")
            print(f"  Input size: {ocr.opt.imgW}x{ocr.opt.imgH}")
        else:
            print("✗ Failed to load model")
    else:
        print("Usage: python custom_plate_ocr.py <model_directory>")
