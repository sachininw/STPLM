from abc import ABC, abstractmethod

class AbstractTransformerLayer(ABC):
    @abstractmethod
    def __init__(self,
                embed_dim,
                num_heads,
                dropout,
                norm,
                norm_first: bool,
                causal: bool,
                ):
        pass
    
    @abstractmethod
    def forward(self, x, attn_mask, output_attentions):
        pass

from .flowformer import FlowformerLayer
from .transformer_encoder import VanillaTransformerEncoderLayer
from .transformer_decoder import Decoder
from .neighborhood_rep import NeighborhoodRepresentation
from .modulator import Modulator
from .gmmvi_latent import GMVAELatentLayer
from.cross_attention import CrossAttentionLayer