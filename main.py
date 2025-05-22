import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- generation hyperparameters ---
MAX_NEW_TOKENS = 64   # shorter replies for faster first response
GEN_KW = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    eos_token_id=None  # will be set later after tokenizer is ready
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="THUDM/glm-4-9b-chat-hf")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    resume_download=True,    # enable interrupted-download resume
    trust_remote_code=True,
    use_fast=False           # fallback to slow tokenizer to avoid ModelWrapper error
)
# ---- ensure pad token & attention mask ----
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # GLM-4 uses same token for PAD/EOS

GEN_KW["eos_token_id"] = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    resume_download=True,    # enable interrupted-download resume
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
    trust_remote_code=True
).eval()

history = []  # 存储多轮对话

def build_prompt(history, user_msg):
    """将历史问答串成符合 GLM‑4 的格式"""
    prompt = ""
    for turn in history:
        prompt += f"<|user|>\n{turn['user']}<|endoftext|>\n<|assistant|>\n{turn['assistant']}<|endoftext|>\n"
    prompt += f"<|user|>\n{user_msg}<|endoftext|>\n<|assistant|>\n"
    return prompt

while True:
    user_msg = input("🧑‍💻 你：").strip()
    if user_msg.lower() in {"exit", "quit"}:
        break
    encoded = tokenizer(
        build_prompt(history, user_msg),
        return_tensors="pt",
        padding=True
    )
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **GEN_KW
        )
    answer = tokenizer.decode(output_ids[0, input_ids.shape[-1]:],
                              skip_special_tokens=True)
    print("🤖 助手：", answer)
    history.append({"user": user_msg, "assistant": answer})