from typing import Literal, Optional

import torch
import torch.nn as nn

from ac_foley.ext.autoencoder.vae import VAE, get_my_vae
from ac_foley.ext.bigvgan_v2.bigvgan import BigVGAN as BigVGANv2
from ac_foley.model.utils.distributions import DiagonalGaussianDistribution


class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_ckpt_path,
                 need_vae_encoder: bool = True):
        super().__init__()
        self.vae: VAE = get_my_vae().eval()
        vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location='cpu')
        self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()
        self.vocoder = BigVGANv2.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x',
                                                    use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
