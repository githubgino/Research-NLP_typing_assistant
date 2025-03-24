import torch
from transformers import pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcribe = pipeline(task="automatic-speech-recognition", model="tamilnlpSLIIT/whisper-ta-v2", chunk_length_s=30, device=device)
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

def transcribe_audio(audio):
    return transcribe(audio)['text']