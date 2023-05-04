import librosa
import os
import soundfile
from tqdm import tqdm
import random
import argparse
import numpy as np
 

os.makedirs("./filelists/",exist_ok=True)
os.makedirs("./checkpoints/",exist_ok=True)


def jsut_preprocess(dataset_dir:str = "./jsut_ver1.1/basic5000/", results_folder="./dataset/jsut/", target_sr:int = 44100):
    os.makedirs(results_folder, exist_ok=True)

    wav_dir = os.path.join(dataset_dir, "wav")
    #"""
    for filename in tqdm(os.listdir(wav_dir)):
        wav_path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(wav_path)
        y_converted = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        save_path = os.path.join(results_folder, filename)
        soundfile.write(save_path, y_converted, target_sr) 
    #"""
    txt_path = os.path.join(dataset_dir, "transcript_utf8.txt")
    text_list = list()
    for txt in read_txt(txt_path):
        if txt == "\n":
            continue
        name, sentence = txt.split(":")
        sentence = sentence.replace("\n", "")
        wav_filepath = os.path.join(results_folder, name +".wav")
        out_txt = wav_filepath + "|" + sentence + "\n"
        text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/jsut_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/jsut_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/jsut_test_{target_sr}.txt", test_list)
    
    return 0

def ita_preprocess(dataset_dir:str = "./path/to/ita_corpus", results_folder="./dataset/ita/", target_sr:int = 44100):
    
    os.makedirs(results_folder, exist_ok=True)
    folder_list = ["recitation", "emotion"]
    #"""
    for folder in folder_list:
        wav_dir = os.path.join(dataset_dir, folder)
        filelist = os.listdir(wav_dir)
        results_folder_dir = os.path.join(results_folder,folder)
        os.makedirs(results_folder_dir, exist_ok=True)
        
        for filename in tqdm(filelist):
            wav_path = os.path.join(wav_dir, filename)
            y, sr = librosa.load(wav_path)
            y_converted = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            for idx in range(999):
                if str(idx).zfill(3) in filename:
                    break
                
            if folder == "recitation":  
                filename_out = "RECITATION324_" + str(idx).zfill(3) + ".wav"
            elif folder == "emotion":
                filename_out = "EMOTION100_"+ str(idx).zfill(3) + ".wav"
            else:
                print("ERROR. Check ITA corpus.")
                continue
            
            save_path = os.path.join(results_folder_dir, filename_out)
            soundfile.write(save_path, y_converted, target_sr)
    #"""
    txt_path = os.path.join(results_folder, "transcript_utf8.txt")
    text_list = list()
    for txt in read_txt(txt_path):
        if txt == "\n":
            continue
        name, sentence = txt.split(":")
        sentence, kana = sentence.split(",")
        sentence = sentence.replace("\n", "")
        if "RECITATION" in name:
            wav_filepath = os.path.join(results_folder,"recitation", name +".wav")
        elif "EMOTION" in name:
            wav_filepath = os.path.join(results_folder,"emotion", name +".wav")
            
        out_txt = wav_filepath + "|" + sentence + "\n"
        text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/ita_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/ita_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/ita_test_{target_sr}.txt", test_list)
    
    return 0


def homebrew_preprocess(dataset_dir:str = "./homebrew/", results_folder="./dataset/homebrew/", target_sr:int = 44100):
 
    os.makedirs(results_folder, exist_ok=True)

    wav_dir = dataset_dir
    for filename in tqdm(os.listdir(wav_dir)):
        wav_path = os.path.join(wav_dir, filename)
        y, sr = librosa.load(wav_path)
        y_converted = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        save_path = os.path.join(results_folder, filename)
        soundfile.write(save_path, y_converted, target_sr)
        
    txt_path = os.path.join(results_folder, "transcript_utf8.txt")
    text_list = list()
    for txt in read_txt(txt_path):
        if txt == "\n":
            continue
        name, sentence = txt.split(":")
        sentence = sentence.replace("\n", "")
        wav_filepath = os.path.join(results_folder, name +".wav")
        out_txt = wav_filepath + "|" + sentence + "\n"
        text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * 0.005)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/homebrew_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/homebrew_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/homebrew_test_{target_sr}.txt", test_list)
    
    return 0

def read_txt(path):
    with open(path, mode="r", encoding="utf-8")as f:
        lines = f.readlines()
    return lines

def write_txt(path, lines):
    with open(path, mode="w", encoding="utf-8")as f:
        f.writelines(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name',
                        type=str,
                        #required=True, 
                        default="jsut",
                        help='jsut or ita or homebrew')
    parser.add_argument('--folder_path',
                        type=str,
                        #required=True, 
                        default="./basic5000/",
                        help='Path to jvs corpus folder')
    parser.add_argument('--sampling_rate',
                        type=str,
                        #required=True,
                        default=44100, 
                        help='Target sampling rate')

    args = parser.parse_args()
    
    if args.dataset_name == "jsut":
        jsut_preprocess(dataset_dir=args.folder_path, target_sr=int(args.sampling_rate))
    elif args.dataset_name == "ita":
        ita_preprocess(dataset_dir=args.folder_path, target_sr=int(args.sampling_rate))
    elif args.dataset_name == "homebrew":
        homebrew_preprocess(dataset_dir=args.folder_path, target_sr=int(args.sampling_rate))
    else:
        print("ERROR. Check dataset_name.")