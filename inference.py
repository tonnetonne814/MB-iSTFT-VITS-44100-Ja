import warnings
warnings.filterwarnings(action='ignore')

import os
import time
import torch
import utils
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from g2p import pyopenjtalk_g2p_prosody
import soundcard as sc
import soundfile as sf


def get_text(text, hps):
    text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def inference(args):

    config_path = args.config
    G_model_path = args.model_path

    # check device
    if  torch.cuda.is_available() is True:
        print("Enter the device number to use.")
        key = input("GPU:0, CPU:1 ===> ")
        if key == "0":
            device="cuda:0"
        elif key=="1":
            device="cpu"
        print(f"Device : {device}")
    else:
        print(f"CUDA is not available. Device : cpu")
        device = "cpu"

    # load config.json
    hps = utils.get_hparams_from_file(config_path)
    
    # load checkpoint
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(G_model_path, net_g, None)

    # play audio by system default
    speaker = sc.get_speaker(sc.default_speaker().name)

    # parameter settings
    noise_scale     = torch.tensor(0.66)    # adjust z_p noise
    noise_scale_w   = torch.tensor(0.8)    # adjust SDP noise
    length_scale    = torch.tensor(1.0)     # adjust sound length scale (talk speed)

    if args.is_save is True:
        n_save = 0
        save_dir = os.path.join("./infer_logs/")
        os.makedirs(save_dir, exist_ok=True)

    ### Dummy Input ###
    with torch.inference_mode():
        stn_phn = pyopenjtalk_g2p_prosody("速度計測のためのダミーインプットです。")
        stn_tst = get_text(stn_phn, hps)
        # generate audio
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, 
                            x_tst_lengths, 
                            noise_scale=noise_scale, 
                            noise_scale_w=noise_scale_w, 
                            length_scale=length_scale)[0][0,0].data.cpu().float().numpy()

    while True:

        # get text
        text = input("Enter text. ==> ")
        if text=="":
            print("Empty input is detected... Exit...")
            break
        
        # measure the execution time 
        torch.cuda.synchronize()
        start = time.time()

        # required_grad is False
        with torch.inference_mode():
            stn_phn = pyopenjtalk_g2p_prosody(text)
            stn_tst = get_text(stn_phn, hps)

            # generate audio
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = net_g.infer(x_tst, 
                                x_tst_lengths, 
                                noise_scale=noise_scale, 
                                noise_scale_w=noise_scale_w, 
                                length_scale=length_scale)[0][0,0].data.cpu().float().numpy()

        # measure the execution time 
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print(f"Gen Time : {elapsed_time}")
        
        # play audio
        speaker.play(audio, hps.data.sampling_rate)
        
        # save audio
        if args.is_save is True:
            n_save += 1
            data = audio
            try:
                save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text}.wav")
                sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")
            except:
                save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text[:10]}〜.wav")
                sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")

            print(f"Audio is saved at : {save_path}")


    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        #default="./logs/ITA_CORPUS/config.json" ,    
                        help='Path to configuration file')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        #default="./logs/ITA_CORPUS/G_1200.pth",
                        help='Path to checkpoint')
    parser.add_argument('--is_save',
                        type=str,
                        default=True,
                        help='Whether to save output or not')
    args = parser.parse_args()
    
    inference(args)