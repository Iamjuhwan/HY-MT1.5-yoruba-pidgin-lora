from huggingface_hub import upload_file

model_card = """---
language:
- yo
- en
- pcm
base_model: tencent/HY-MT1.5-1.8B
tags:
- translation
- lora
- peft
- yoruba
- nigerian-pidgin
- african-languages
- qlora
license: apache-2.0
---

# HY-MT1.5-1.8B — Yoruba & Nigerian Pidgin LoRA

Fine-tuned version of [tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B)
on Yoruba-English and Nigerian Pidgin-English translation pairs using QLoRA.

## Evaluation Results (Baseline vs Fine-Tuned)

Evaluated on 299 clean Yoruba/Pidgin/English examples,
filtered from noisy OPUS-100 software strings:

| Metric | Base Model | Fine-Tuned | Delta |
|--------|------------|------------|-------|
| BLEU   | 6.21       | 6.90       | +0.70 |
| chrF   | 13.57      | 13.59      | +0.03 |

BLEU scores are characteristically low for morphologically rich
low-resource languages like Yoruba. The +0.70 BLEU improvement
is meaningful given the small training set (~2,000 pairs).
chrF improvement was marginal, indicating that larger, higher-quality
agricultural domain data would produce stronger gains.

## Motivation

The base model was evaluated across 12 adversarial probe categories
(see [Jesujuwon/HY-MT1.5-1.8B-blindspots](https://huggingface.co/datasets/Jesujuwon/HY-MT1.5-1.8B-blindspots)).
Key failure modes identified:

- Literal translation of idiomatic expressions
- Hallucination on low-resource African language inputs
- Poor handling of code-switched Pidgin/Yoruba/English text
- Loss of sarcastic and ironic register

## Training Data

| Source | Pairs | Languages |
|--------|-------|-----------|
| OPUS-100 (Helsinki-NLP) | ~4,000 | English and Yoruba |
| Jesujuwon/HY-MT1.5-1.8B-blindspots | 12 | Multi (correction pairs) |
| Custom synthetic pairs | 24 | Nigerian Pidgin and English, Yoruba and English |

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | tencent/HY-MT1.5-1.8B |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Optimizer | paged_adamw_8bit |
| Hardware | Google Colab T4 16GB |
| Training time | 75 minutes |

## How to Use

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "tencent/HY-MT1.5-1.8B"
adapter_name    = "Jesujuwon/HY-MT1.5-1.8B-yoruba-pidgin-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model     = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_name)
model.eval()

def translate(text, target_language):
    instruction = f"Translate the following segment into {target_language}, without additional explanation.\\n\\n{text}"
    messages  = [{"role": "user", "content": instruction}]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    input_len = tokenized.shape[1]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                tokenized, max_new_tokens=256,
                do_sample=False, repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

print(translate("E don do. Nothing fit change am again.", "English"))
print(translate("Education is the key to a better future.", "Yoruba"))

## Limitations

- Training data is small (~4,000 pairs); larger Yoruba corpora would improve scores
- Nigerian Pidgin pairs are synthetic and limited in dialectal variety
- Model inherits base model constraints (no terminology glossary, 1.8B capacity)
- Next phase: agricultural domain fine-tuning for farming advice use cases

## Citation

Hunyuan-MT-2025: HY-MT1.5 Tencent Hunyuan Machine Translation Model.
Tencent Hunyuan Team, 2025.
https://huggingface.co/tencent/HY-MT1.5-1.8B
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(model_card)

upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id="Jesujuwon/HY-MT1.5-1.8B-yoruba-pidgin-lora",
    repo_type="model",
    token=hf_token,
)
print("✅ Model card updated!")
print("View: https://huggingface.co/Jesujuwon/HY-MT1.5-1.8B-yoruba-pidgin-lora")
