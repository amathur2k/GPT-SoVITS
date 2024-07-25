import os, re, logging, sys
import LangSegment
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import soundfile as sf

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#sys.path.append(r"C:\GPT-SoVITS-beta\GPT-SoVITS-beta0706")


vid = "romeov3"
os.environ["gpt_path"]=os.path.join("GPT_weights", rf'{vid}-e10.ckpt')
os.environ["sovits_path"] = os.path.join("SoVITS_weights", rf'{vid}_e12_s252.pth')

import workhorse_inference

#os.environ["gpt_path"]='GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt'
#gpt_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
#os.environ["sovits_path"]='GPT_SoVITS/pretrained_models/s2G488k.pth'
#sovits_path: GPT_SoVITS/pretrained_models/s2G488k.pth
os.environ["cnhubert_base_path"]='GPT_SoVITS/pretrained_models/chinese-hubert-base'
#scnhubert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
os.environ["bert_path"]='GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
#bert_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
os.environ["_CUDA_VISIBLE_DEVICES"]='0'
#_CUDA_VISIBLE_DEVICES: 0
os.environ["is_half"]='True'
#is_half: True
os.environ["infer_ttswebui"]='9872'
#infer_ttswebui: 9872
os.environ["is_share"]='False'
#is_share: False


workhorse_inference.get_tts_wav(ref_wav_path = r"/home/ubuntu/GPT-SoVITS/outputs/romeo_slicer_opt/romeo.wav_0000020160_0000192960.wav",
                    prompt_text = 'Hello, my name is romeo, and today has been a good day.',
                    prompt_language = 'English', text = 'I would like to endorse Kamala Harris as the next president of the United States. She is not only kind, affable and dilignt but also sharp as a whip.', text_language = 'English',
                    opfile = rf"outputs\{vid}_generated.wav",
                    how_to_cut='Slice once every 4 sentences', top_k=5, top_p=1, temperature=1, ref_free = False)
