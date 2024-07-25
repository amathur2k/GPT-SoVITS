#This file will be used to seperate out the finetuning codebase
#Step 1: Slice the Audio
import os,shutil,sys,pdb,re
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import json,yaml,warnings,torch
import platform
import psutil
import signal
import workhorse
#Start

vid = 'romeo'
workhorse.open_slice(inp=os.path.join("inputs", rf"{vid}.wav"),
                    opt_root=os.path.join("outputs", rf"{vid}_slicer_opt"),
                    threshold='-34', min_length='4000', min_interval='300', hop_size='10',
                    max_sil_kept='500', _max=0.9, alpha=0.25, n_parts=4)

workhorse.open_asr(asr_inp_dir=os.path.join("outputs", rf"{vid}_slicer_opt"),
                   asr_opt_dir=os.path.join("outputs", rf"{vid}_asr_opt"),
                   asr_model='Faster Whisper (多语种)',asr_model_size='large-v3',asr_lang='en')


workhorse.open1abc(inp_text =
                    os.path.join("outputs", rf"{vid}_asr_opt", rf"{vid}_slicer_opt.list"),
                   inp_wav_dir = '',exp_name = vid, gpu_numbers1a = '0-0', gpu_numbers1Ba = '0-0',
                   gpu_numbers1c = '0-0',bert_pretrained_dir = 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
                   ssl_pretrained_dir = 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
                   pretrained_s2G_path = 'GPT_SoVITS/pretrained_models/s2G488k.pth')


workhorse.open1Ba(batch_size = 2, total_epoch = 24, exp_name = vid,text_low_lr_rate = 0.4 ,if_save_latest = True,
                  if_save_every_weights = True,save_every_epoch =4,gpu_numbers1Ba = "0",
                  pretrained_s2G = 'GPT_SoVITS/pretrained_models/s2G488k.pth',
                  pretrained_s2D = 'GPT_SoVITS/pretrained_models/s2D488k.pth')

workhorse.open1Bb(batch_size = 1,total_epoch = 40,exp_name = vid,if_dpo = False,if_save_latest = True,
                  if_save_every_weights = True,save_every_epoch = 5,gpu_numbers = "0",
                  pretrained_s1 ='GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')



#"C:\Users\amath\PycharmProjects\GPT-SoVITS2\outputs\merged.wav"
#"C:\Users\amath\PycharmProjects\GPT-SoVITS2\outputs\slicer_opt"
#"C:\Users\amath\PycharmProjects\GPT-SoVITS2\outputs\asr_opt"

"""
ps_slice=[]
def open_slice(inp,opt_root,threshold = '-34',min_length = '4000',min_interval = '300',hop_size = '10',
               max_sil_kept = '500',_max:float = 0.9,alpha:float = 0.25 ,n_parts: int = 4):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    if(os.path.exists(inp)==False):
        yield "输入路径不存在",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if os.path.isfile(inp):n_parts=1
    elif os.path.isdir(inp):pass
    else:
        yield "输入路径存在但既不是文件也不是文件夹",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (python_exec,inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "切割执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps_slice:
            p.wait()
        ps_slice=[]
        yield "切割结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的切割任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice=[]
    return "已终止所有切割进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
"""
