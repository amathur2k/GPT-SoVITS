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


vid = "prerna"
os.environ["gpt_path"]=rf'C:\Users\amath\PycharmProjects\GPT-SoVITS2\GPT_weights\prerna-e15.ckpt'
os.environ["sovits_path"] = rf'C:\Users\amath\PycharmProjects\GPT-SoVITS2\SoVITS_weights\prerna_e12_s624.pth'

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


workhorse_inference.get_tts_wav(ref_wav_path = r"C:\Users\amath\PycharmProjects\GPT-SoVITS2\outputs\prerna_slicer_opt\prerna.wav_0000076800_0000305920.wav",
                    prompt_text = 'Now, Over the last two or three monetary policy cycles, we have seen',
                    prompt_language = 'English', text = 'Umm! ! Here is the analysis for the 2024 Budget. I think its a great budget with emphasis on Job creation and fiscal prudence', text_language = 'English',
                    opfile = rf"outputs\{vid}_generated.wav",
                    how_to_cut='Slice once every 4 sentences', top_k=5, top_p=1, temperature=1, ref_free = False)
