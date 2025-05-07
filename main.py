import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="THUDM/glm-4-9b-chat-hf")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, device_map="auto",
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
    input_ids = tokenizer(build_prompt(history, user_msg),
                          return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(output_ids[0, input_ids.shape[-1]:],
                              skip_special_tokens=True)
    print("🤖 助手：", answer)
    history.append({"user": user_msg, "assistant": answer})