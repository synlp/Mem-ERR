import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    def __init__(self, initial_memory):
        super().__init__()
        if initial_memory.ndim != 2:
            raise ValueError("memory must be a matrix")
        self.memory = nn.Parameter(initial_memory.detach().clone().float())
        dimension = initial_memory.shape[1]
        self.key_projection = nn.Linear(dimension, dimension, bias=False)
        self.value_projection = nn.Linear(dimension, dimension, bias=False)

    def forward(self, hidden):
        if hidden.shape[-1] != self.memory.shape[1]:
            raise ValueError("hidden and memory dimensions differ")
        keys = self.key_projection(self.memory)
        values = self.value_projection(self.memory)
        weights = torch.softmax(hidden @ keys.transpose(0, 1), dim=-1)
        readout = weights @ values
        return hidden + readout


class VisualFusionEncoder(nn.Module):
    def __init__(self, visual_dimension=768, hidden_dimension=512, layers=12, heads=8, feedforward_dimension=2048, dropout=0.1, sequence_length=50):
        super().__init__()
        self.projection = nn.Linear(visual_dimension, hidden_dimension)
        self.position_embeddings = nn.Parameter(torch.zeros(1, sequence_length, hidden_dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, raw_features, segmented_features):
        if raw_features.shape != segmented_features.shape:
            raise ValueError("raw and segmented features must have equal shapes")
        fused = self.projection(raw_features + segmented_features)
        if fused.shape[1] > self.position_embeddings.shape[1]:
            raise ValueError("visual sequence exceeds position embedding length")
        return self.encoder(fused + self.position_embeddings[:, :fused.shape[1]])


class MemERR(nn.Module):
    def __init__(self, model_name, initial_memory, hidden_dimension=512, layers=12, heads=8, feedforward_dimension=2048, dropout=0.1):
        super().__init__()
        from transformers import CLIPVisionModel

        if initial_memory.shape[1] != hidden_dimension:
            raise ValueError("memory dimension must match the fusion encoder dimension")
        self.model_name = model_name
        self.raw_encoder = CLIPVisionModel.from_pretrained(model_name)
        self.segmented_encoder = CLIPVisionModel.from_pretrained(model_name)
        visual_config = self.raw_encoder.config
        sequence_length = (visual_config.image_size // visual_config.patch_size) ** 2 + 1
        self.fusion = VisualFusionEncoder(
            visual_dimension=visual_config.hidden_size,
            hidden_dimension=hidden_dimension,
            layers=layers,
            heads=heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            sequence_length=sequence_length,
        )
        self.memory = MemoryModule(initial_memory)

    def forward(self, raw_pixels, segmented_pixels):
        raw_features = self.raw_encoder(pixel_values=raw_pixels).last_hidden_state
        segmented_features = self.segmented_encoder(pixel_values=segmented_pixels).last_hidden_state
        hidden = self.fusion(raw_features, segmented_features)
        return self.memory(hidden)
