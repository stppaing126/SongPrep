import random
import numpy as np
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import autocast

from mucodec.descript_quantize3 import ResidualVectorQuantize
from mucodec.our_MERT_BESTRQ.test import load_model


class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        """Project back from diffusion space to the actual sample space."""
        return z

class Feature1DProcessor(SampleProcessor):
    def __init__(self, dim: int = 100, power_std = 1., \
                 num_samples: int = 100_000, cal_num_frames: int = 600):
        super().__init__()

        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.cal_num_frames = cal_num_frames
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(dim))
        self.register_buffer('sum_x2', torch.zeros(dim))
        self.register_buffer('sum_target_x2', torch.zeros(dim))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        if(self.counts < 10):
            mean = torch.zeros_like(mean)
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        if(self.counts < 10):
            std = torch.ones_like(std)
        return std

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        if self.counts.item() < self.num_samples:
            self.counts += len(x)
            self.sum_x += x[:,:,0:self.cal_num_frames].mean(dim=(2,)).sum(dim=0)
            self.sum_x2 += x[:,:,0:self.cal_num_frames].pow(2).mean(dim=(2,)).sum(dim=0)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, -1, 1)) * rescale.view(1, -1, 1)
        return x

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        rescale = (self.std / self.target_std) ** self.power_std
        # print(rescale, self.mean)
        x = x * rescale.view(1, -1, 1) + self.mean.view(1, -1, 1)
        return x

def pad_or_tunc_tolen(prior_text_encoder_hidden_states, prior_text_mask, prior_prompt_embeds, len_size=77):
    if(prior_text_encoder_hidden_states.shape[1]<len_size):
        prior_text_encoder_hidden_states = torch.cat([prior_text_encoder_hidden_states, \
            torch.zeros(prior_text_mask.shape[0], len_size-prior_text_mask.shape[1], \
            prior_text_encoder_hidden_states.shape[2], device=prior_text_mask.device, \
            dtype=prior_text_encoder_hidden_states.dtype)],1)
        prior_text_mask = torch.cat([prior_text_mask, torch.zeros(prior_text_mask.shape[0], len_size-prior_text_mask.shape[1], device=prior_text_mask.device, dtype=prior_text_mask.dtype)],1)
    else:
        prior_text_encoder_hidden_states = prior_text_encoder_hidden_states[:,0:len_size]
        prior_text_mask = prior_text_mask[:,0:len_size]
    prior_text_encoder_hidden_states = prior_text_encoder_hidden_states.permute(0,2,1).contiguous()
    return prior_text_encoder_hidden_states, prior_text_mask, prior_prompt_embeds


class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self,
        num_channels,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        hubert_layer=None,
        ssl_layer=None,
        uncondition=True,
        ssl_path=None,
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition
        self.num_channels = num_channels
        self.hubert_layer = hubert_layer
        self.ssl_layer = ssl_layer

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.normfeat = Feature1DProcessor(dim=64)

        self.sample_rate = 48000
        self.num_samples_perseg = self.sample_rate * 20 // 1000
        self.rsp48toclap = torchaudio.transforms.Resample(48000, 24000)
        self.rsq48towav2vec = torchaudio.transforms.Resample(48000, 16000)
        # self.wav2vec = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", trust_remote_code=True)
        # self.wav2vec_processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", trust_remote_code=True)
        self.bestrq = load_model(
            model_dir='mucodec/our_MERT_BESTRQ/mert_fairseq', # path/to/our-MERT/mert_fairseq
            checkpoint_dir=ssl_path,
        )
        self.rsq48tobestrq = torchaudio.transforms.Resample(48000, 24000)
        self.rsq48tohubert = torchaudio.transforms.Resample(48000, 16000)
        for v in self.bestrq.parameters():v.requires_grad = False
        self.rvq_bestrq_emb = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 16_384, codebook_dim = 32, quantizer_dropout = 0.0, stale_tolerance=200)
        for v in self.rvq_bestrq_emb.parameters():v.requires_grad = False

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def preprocess_audio(self, input_audios, threshold=0.9):
        assert len(input_audios.shape) == 2, input_audios.shape
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios/norm_value.unsqueeze(-1)

    def extract_bestrq_embeds(self, input_audio_0,input_audio_1,layer):
        self.bestrq.eval()
        # print("audio shape:",input_audio_0.shape)
        input_wav_mean = (input_audio_0 + input_audio_1) / 2.0
        # print("input_wav_mean.shape:",input_wav_mean.shape)
        # input_wav_mean = torch.randn(2,1720320*2).to(input_audio_0.device)
        input_wav_mean = self.bestrq(self.rsq48tobestrq(input_wav_mean), features_only = True)
        layer_results = input_wav_mean['layer_results']
        # print("layer_results.shape:",layer_results[layer].shape)
        bestrq_emb = layer_results[layer]
        bestrq_emb = bestrq_emb.permute(0,2,1).contiguous()
        #[b,t,1024] t=t/960
        #35.84s->batch,896,1024
        return bestrq_emb

    def forward(self, input_audios, train_rvq=True, layer=6):
        if not hasattr(self,"device"):
            self.device = input_audios.device
        if not hasattr(self,"dtype"):
            self.dtype = input_audios.dtype
        device = self.device
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        with torch.no_grad():
            with autocast(enabled=False):
                bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
            bestrq_emb = bestrq_emb.detach()

        if(train_rvq):
            quantized_bestrq_emb, _, _, commitment_loss_bestrq_emb, codebook_loss_bestrq_emb,_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t
        else:
            bestrq_emb = bestrq_emb.float()
            self.rvq_bestrq_emb.eval()
            # with autocast(enabled=False):
            quantized_bestrq_emb, _, _, commitment_loss_bestrq_emb, codebook_loss_bestrq_emb,_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t
            commitment_loss_bestrq_emb = commitment_loss_bestrq_emb.detach()
            codebook_loss_bestrq_emb = codebook_loss_bestrq_emb.detach()
            quantized_bestrq_emb = quantized_bestrq_emb.detach()

        commitment_loss = commitment_loss_bestrq_emb
        codebook_loss = codebook_loss_bestrq_emb

        return commitment_loss.mean(), codebook_loss.mean()

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes(self, input_audios, additional_feats,layer):
        input_audio_0 = input_audios[[0],:]
        input_audio_1 = input_audios[[1],:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.bestrq.eval()
        bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
        bestrq_emb = bestrq_emb.detach()

        self.rvq_bestrq_emb.eval()
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t


        if('spk' in additional_feats):
            self.xvecmodel.eval()
            spk_embeds = self.extract_spk_embeds(input_audios)
        else:
            spk_embeds = None

        return [codes_bestrq_emb], [bestrq_emb], spk_embeds

    @torch.no_grad()
    def fetch_codes_batch(self, input_audios, additional_feats,layer):
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.bestrq.eval()
        bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
        bestrq_emb = bestrq_emb.detach()
        self.rvq_bestrq_emb.eval()
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t

        if('spk' in additional_feats):
            self.xvecmodel.eval()
            spk_embeds = self.extract_spk_embeds(input_audios)
        else:
            spk_embeds = None
        return [codes_bestrq_emb], [bestrq_emb], spk_embeds

    @torch.no_grad()
    def inference(self, input_audios, lyric, true_latents, latent_length, additional_feats, guidance_scale=2, num_steps=20,
                  disable_progress=True,layer=5,scenario='start_seg'):
        codes, embeds, spk_embeds = self.fetch_codes(input_audios, additional_feats,layer)

        latents = self.inference_codes(codes, spk_embeds, true_latents, latent_length, additional_feats, \
            guidance_scale=guidance_scale, num_steps=num_steps, \
            disable_progress=disable_progress,scenario=scenario)
        return latents
    
    @torch.no_grad()
    def inference_rtf(self, input_audios, lyric, true_latents, latent_length, additional_feats, guidance_scale=2, num_steps=20,
                  disable_progress=True,layer=5,scenario='start_seg'):
        codes, embeds, spk_embeds = self.fetch_codes(input_audios, additional_feats,layer)
        import time
        start = time.time()
        latents = self.inference_codes(codes, spk_embeds, true_latents, latent_length, additional_feats, \
            guidance_scale=guidance_scale, num_steps=num_steps, \
            disable_progress=disable_progress,scenario=scenario)
        return latents,time.time()-start
