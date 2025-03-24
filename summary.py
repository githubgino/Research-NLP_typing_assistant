# Use a pipeline as a high-level helper
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# Download the model
model_name = "tamilnlpSLIIT/tamil-summary-XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def do_summary(text):
    """
    Perform text summarization using the model that has been trained. This uses HuggingFace.
    """
    complete_summary = ""
    text = text.split("\n")
    for txt in text:
        input_ids = tokenizer(
            [WHITESPACE_HANDLER(txt)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=256,
            min_length=64,
            no_repeat_ngram_size=2,
            num_beams=1
        )[0]

        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        complete_summary += summary + " "

    return complete_summary