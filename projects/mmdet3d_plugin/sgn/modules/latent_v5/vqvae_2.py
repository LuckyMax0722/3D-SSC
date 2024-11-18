import torch
from torch import nn

from .vector_quantizer import VectorQuantizer
from .encoder import C_Encoder
from .decoder import C_Decoder


class vqvae_2(torch.nn.Module):
    def __init__(self, init_size, num_classes, vq_size, l_size, l_attention) -> None:
        super(vqvae_2, self).__init__()

        embedding_dim = num_classes
        
        self.VQ = VectorQuantizer(num_embeddings = num_classes*vq_size, embedding_dim = embedding_dim)

        self.encoder = C_Encoder(nclasses=num_classes, init_size=init_size, l_size=l_size, attention=l_attention)
        self.quant_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1, stride=1)
        
        self.decoder = C_Decoder(nclasses=num_classes, init_size=init_size, l_size=l_size, attention=l_attention)
        self.post_quant_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1, stride=1)
        
        self.criterion = torch.nn.CrossEntropyLoss()

    def encode(self, x):
        latent = self.encoder(x) 
        latent = self.quant_conv(latent)
        return latent

    def decode(self, quantized_latent):
        quantized_latent = self.post_quant_conv(quantized_latent)
        recons = self.decoder(quantized_latent)
        return recons
    
    def vector_quantize(self, latent):
        quantized_latent, vq_loss, quantized_latent_ind, latents_shape = self.VQ(latent)
        return quantized_latent, vq_loss, quantized_latent_ind, latents_shape

    def coodbook(self,quantized_latent_ind, latents_shape):
        quantized_latent = self.VQ.codebook_to_embedding(quantized_latent_ind.view(-1,1), latents_shape)
        return quantized_latent

    def multi_criterion(self, recons, x):
        return self.criterion(recons, x)
    
    def forward(self, x):
        latent = self.encode(x) 

        quantized_latent, vq_loss, _, _ = self.vector_quantize(latent) 
        
        recons = self.decode(quantized_latent)
        
        recons_loss = self.multi_criterion(recons, x)
        
        return quantized_latent, vq_loss, recons_loss
    
    def forward_encoder(self, x):
        latent = self.encode(x) 

        quantized_latent, _, _, _ = self.vector_quantize(latent) 
    
        return quantized_latent
    
    def forward_decoder(self, x):
        recons = self.decode(x)
        
        return recons
