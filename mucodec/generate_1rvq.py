import torch
import torchaudio
import numpy as np
from safetensors.torch import load_file

from mucodec.model_1rvq import PromptCondAudioDiffusion


class Tango:
    def __init__(self, \
        model_path, \
        ssl_path,
        layer_num=6, \
        device="cuda:0"):
        
        self.sample_rate = 48000
        self.device = device

        self.layer_num = layer_num

        self.MAX_DURATION = 360
        main_config = {
            "num_channels":32,
            "unet_model_name":None,
            "unet_model_config_path":"configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
            "snr_gamma":None,
            "ssl_path":ssl_path,
        }
        self.model = PromptCondAudioDiffusion(**main_config).to(device)
        if model_path.endswith(".safetensors"):
            main_weights = load_file(model_path)
        else:
            main_weights = torch.load(model_path, map_location=device)
        self.model.load_state_dict(main_weights, strict=False)
        print ("Successfully loaded checkpoint from:", model_path)
        
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)
        print("scaling factor: ", self.model.normfeat.std)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def sound2code(self, orig_samples, batch_size=3):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        audios = self.preprocess_audio(audios)
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        print("output_len: ", output_len)

        while(audios.shape[-1] < min_samples):
            audios = torch.cat([audios, audios], -1)
        int_max_len=audios.shape[-1]//min_samples+1
        audios = torch.cat([audios, audios], -1)
        audios=audios[:,:int(int_max_len*(min_samples))]
        codes_list=[]

        audio_input = audios.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_input.shape[0], batch_size):
            # import pdb; pdb.set_trace()
            codes, _, spk_embeds = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer=self.layer_num)
            codes_list.append(torch.cat(codes, 1))
            # print("codes_list",codes_list[0].shape)

        codes = torch.cat(codes_list, 0).permute(1,0,2).reshape(1, -1)[None] # B 3 T -> 3 B T
        codes=codes[:,:,:output_len]

        return codes

    @torch.no_grad()
    def preprocess_audio(self, input_audios, threshold=0.8):
        assert len(input_audios.shape) == 3, input_audios.shape
        nchan = input_audios.shape[1]
        input_audios = input_audios.reshape(input_audios.shape[0], -1)
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios.reshape(input_audios.shape[0], nchan, -1)/norm_value.unsqueeze(-1).unsqueeze(-1)

if __name__=="__main__":
    '''
    PYTHONPATH=/apdcephfs_cq2/share_1297902/speech_user/erichtchen/Project/latent_diffusion/audiolm-pytorch:/apdcephfs_cq2/share_1297902/speech_user/erichtchen/Project/latent_diffusion/musiclm-pytorch TRANSFORMERS_CACHE=/apdcephfs_cq2/share_1297902/speech_train/speech_transfer_data/erichtchen/modelzoo/semanticlmdiff/modelzoo/hub
    '''
    # model_path = sys.argv[1]
    # model_name = os.path.split(model_path)[-1].split(".")[0]

    # tango = Tango(model_path=model_path)
    # src_dir = "/apdcephfs_cq11/share_300883980/tanwei/music_recons_eval/audio_segments/"
    # dst_dir = os.path.join("/apdcephfs_cq11/share_300883980/tanwei/music_recons_eval/exp/", model_name)
    # os.makedirs(dst_dir, exist_ok=True)

    # for wav_name in os.listdir(src_dir):
    #     wav_path = os.path.join(src_dir, wav_name)
    #     dst_wav_path = os.path.join(dst_dir, wav_name)
    #     src_wave, fs = torchaudio.load(wav_path)
    #     if (fs != 48000):
    #         src_wave = torchaudio.functional.resample(src_wave, fs, 48000)
    #     codes = tango.sound2code(src_wave)
    #     dst_wave = tango.code2sound(codes, prompt=src_wave)
    #     min_len = min(src_wave.size(1), dst_wave.size(1))
    #     print(src_wave.shape, dst_wave.shape)
    #     test_wave = torch.cat([src_wave[..., :min_len], dst_wave[..., :min_len]], dim=0)
    #     torchaudio.save(dst_wav_path, test_wave.detach().cpu(), 48000)

    tango = Tango(model_path='./saved/model_1rvq/model_2_fixed.safetensors')
    # complete_mat = kaldiio.load_mat("/apdcephfs_sh8/share_302150444/music_token/Flow1dVAE-v0.0/exp_stereo_7_1x4_fixed/suno_en_20250722/JOB0//0/token.ark:12")[:,0:1,:]
    complete_mat = np.load("/apdcephfs_cq12/share_300883980/music_data/dpo_data_en_yc0620/npy_files/276286151_s5.npy")
    complete_mat = torch.tensor(np.expand_dims(complete_mat, axis=0))
    wave = tango.code2sound(complete_mat).cpu()
    torchaudio.save("tmp.wav", wave.detach().cpu(), 48000)
    exit()


    # scp_file = "/apdcephfs_cq12/share_300883980/user/tanwei/tango_descript_inference2/Flow1dVAE-v0.0/test_sample/exp_stereo_7_1x4_fixed/0/token.scp"
    # out_dir = "/apdcephfs_cq12/share_300883980/user/tanwei/tango_descript_inference2/Flow1dVAE-v0.0/test_sample/exp_stereo_7_1x4_fixed"
    # tango = Tango(model_path='./saved/model_1rvq/model_2_fixed.safetensors')

    # with open(scp_file) as fp:
    #     lines = fp.readlines()
    #     for line in lines: 
    #         idx, ark_path = line.strip().split()
    #         complete_mat = kaldiio.load_mat(ark_path)[:,0:1,:]
    #         wave = tango.code2sound(complete_mat).cpu()
    #         torchaudio.save(os.path.join(out_dir, f"{idx}.wav"), wave.detach().cpu(), 48000)
    # exit()


    # # import jsonlines
    # # tango = Tango(model_path="/apdcephfs_nj7/share_301796285/user/yaoxunxu/diffusion_train/saved/train_vocal_bestrq_flow_1x1_multi_7_fp16_large_mean_newssl_16384_1dvae_large_repa_hubert_4800_39_7_freeze_orirvq/latest//model_1.safetensors")
    # tango = Tango(model_path='./saved/model_1rvq/model_2_fixed.safetensors')

    # wave = tango.code2sound(torch.tensor(np.load("/apdcephfs_cq10/share_1297902/user/tomasyu/shunlei/codeclm_song_train_clean_v28/example_layer7/codeclm_full_alldata/20250421162633_cfg1.5_temp0.9_record1_11_170000/suno1229_416265/01_当代R&B_s0.flac_tokens.npy"))[None, [0], :], None).cpu()
    # print(wave.shape)
    # torchaudio.save('/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/tango_descript_inference2/Flow1dVAE-v0.0/samples/temp/01.wav', \
    #         wave.detach().cpu(), 48000)
    # wave = tango.code2sound(torch.tensor(np.load("/apdcephfs_cq10/share_1297902/user/tomasyu/shunlei/codeclm_song_train_clean_v28/example_layer7/codeclm_full_alldata/20250421162633_cfg1.5_temp0.9_record1_11_170000/suno1229_416265/02_当代R&B_s0.flac_tokens.npy"))[None, [0], :], None).cpu()
    # print(wave.shape)
    # torchaudio.save('/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/tango_descript_inference2/Flow1dVAE-v0.0/samples/temp/02.wav', \
    #         wave.detach().cpu(), 48000)
    # wave = tango.code2sound(torch.tensor(np.load("/apdcephfs_cq10/share_1297902/user/tomasyu/shunlei/codeclm_song_train_clean_v28/example_layer7/codeclm_full_alldata/20250421162633_cfg1.5_temp0.9_record1_11_170000/suno1229_416265/03_国风（流行）_s0.flac_tokens.npy"))[None, [0], :], None).cpu()
    # print(wave.shape)
    # torchaudio.save('/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/tango_descript_inference2/Flow1dVAE-v0.0/samples/temp/03.wav', \
    #         wave.detach().cpu(), 48000)
    # exit()

    # root_dir = '/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/tango_descript_inference2/Flow1dVAE-v0.0/samples/temp_kunlun/orig'
    # filelist = []
    # for f in [os.path.join(root_dir, f) for f in os.listdir(root_dir) if '.flac' in f or '.wav' in f or '.mp3' in f]:
    #     a, fs = torchaudio.load(f)
    #     if(fs!=48000):
    #         a = torchaudio.functional.resample(a, fs, 48000)
    #     if(a.shape[0]==1):
    #         a = torch.cat([a,a],0)
    #     ori_len = a.shape[-1]
    #     filelist.append([a, '', [0, a.shape[-1]/48000.], f,ori_len])
        
    # for sample_idx, (orig_samples, lyric, st_et, fname,ori_len) in enumerate(filelist):
    #     print(fname, lyric)
    #     # wave = tango.sound2sound(orig_samples,None)
    #     wave = tango.sound2sound_vae(orig_samples.cuda())
    #     wave = wave[:,0:ori_len]
    #     torchaudio.save('/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/tango_descript_inference2/Flow1dVAE-v0.0/samples/temp_kunlun/reconstruct_vae/{}'.format(os.path.basename(fname.replace('.flac','.wav').replace('.mp3','.wav'))), \
    #         wave.detach().cpu(), 48000)
    # exit()


    # wave = tango.code2sound(torch.tensor(np.load("/apdcephfs_sh8/share_302150444/user/tanwei/Megatron-LLM/examples/qwen2_song_codeclm/bash/song-token.npy"))[None, None, :], None).cpu()
    # print(wave.shape)
    # torchaudio.save('/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/tango_descript_inference2/Flow1dVAE-v0.0/samples/temp/temp.wav', \
    #         wave.detach().cpu(), 48000)
    # exit()

    # root_dir = '/apdcephfs_nj7/share_301796285/user/yaoxunxu/diffusion_train/data/cq_wav'
    # filelist = []
    # for f in [os.path.join(root_dir, f) for f in os.listdir(root_dir) if '.flac' in f or '.wav' in f or '.mp3' in f]:
    #     a, fs = torchaudio.load(f)
    #     if(fs!=48000):
    #         a = torchaudio.functional.resample(a, fs, 48000)
    #     if(a.shape[0]==1):
    #         a = torch.cat([a,a],0)
    #     ori_len = a.shape[-1]
    #     filelist.append([a, '', [0, a.shape[-1]/48000.], f,ori_len])
        
    # for sample_idx, (orig_samples, lyric, st_et, fname,ori_len) in enumerate(filelist):
    #     print(fname, lyric)
    #     wave = tango.sound2sound(orig_samples,None)
    #     wave = wave[:,0:ori_len]
    #     torchaudio.save('samples/freezervq_100k_start3/{}'.format(os.path.basename(fname.replace('.flac','.wav').replace('.mp3','.wav'))), \
    #         wave.detach().cpu(), 48000)
    # exit()

    # # filelist = []
    # # # with open("samples/vocaldataset20240327_filtered/text.jsonl",'r') as ftext2music:
    # # #     for item in jsonlines.Reader(ftext2music):
    # # #         filelist.append([torchaudio.load(item['path'])[0], item['lyric'], [item['st'], item['et']], item['path']])
    # # root_dir = '/apdcephfs_nj7/share_301796285/user/yaoxunxu/diffusion_train/data/cq_wav'
    # # for f in [os.path.join(root_dir, f) for f in os.listdir(root_dir) if '.flac' in f or '.wav' in f or '.mp3' in f]:
    # #     a, fs = torchaudio.load(f)
    # #     if(fs!=48000):
    # #         a = torchaudio.functional.resample(a, fs, 48000)
    # #     if(a.shape[0]==1):
    # #         a = torch.cat([a,a],0)
    # #     ori_len = a.shape[-1]
    # #     filelist.append([a, '', [0, a.shape[-1]/48000.], f,ori_len])
    
    # jsonl_path = '/apdcephfs_cq7/share_1297902/common/erichtchen/Project/latent_diffusion/diffusion_train/samples/humanLabel/label.jsonl'
    # filelist = []
    # with open(jsonl_path, 'r') as fjson:
    #     for line in tqdm(fjson):
    #         d = json.loads(line)
    #         f = d['path'].replace('.mp3', '.wav')
    #         a, fs = torchaudio.load(f)
    #         if(fs!=48000):
    #             a = torchaudio.functional.resample(a, fs, 48000)
    #         if(a.shape[0]==1):
    #             a = torch.cat([a, a],0)
    #         filelist.append([a, '', [0, a.shape[-1]/48000.], f, a.shape[-1]])

    # os.makedirs('samples/freezervq_100k_other3',exist_ok=True)

    # for sample_idx, (orig_samples, lyric, st_et, fname,ori_len) in enumerate(filelist):
    #     print(fname, lyric)
    #     codes = tango.sound2code(orig_samples).cpu()
    #     torch.save(codes.cpu(), '{}'.format(os.path.basename(fname.replace('.flac','_1rvq.pt').replace('.mp3','_1rvq.pt').replace('.wav', '_1rvq.pt'))))
    #     exit()

    # for sample_idx, (orig_samples, lyric, st_et, fname,ori_len) in enumerate(filelist):
    #     print(fname, lyric)
    #     wave = tango.sound2sound(orig_samples,orig_samples[:, 50*fs:60*fs])
    #     wave = wave[:,0:ori_len]
    #     torchaudio.save('samples/freezervq_100k_other3/{}'.format(os.path.basename(fname.replace('.flac','.wav').replace('.mp3','.wav'))), \
    #         wave.detach().cpu(), 48000)
    #     # if sample_idx == 2:
    #     #     exit()
    
