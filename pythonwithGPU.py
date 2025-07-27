# pip install vllm transformers trl datasets accelerate evaluate

from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
import random

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(model=base_model, tensor_parallel_size=1)

def judge(sample, reference):
    # 例: コードならユニットテストを実行して pass/fail を返す
    # ここではダミーで文字列一致率
    import difflib
    ratio = difflib.SequenceMatcher(None, sample.strip(), reference.strip()).ratio()
    return ratio

def generate_batch(prompts, n=6, temperature=0.8):
    params = SamplingParams(n=n, temperature=temperature, max_tokens=1024)
    outputs = llm.generate(prompts, params)
    # outputs は各プロンプトに対する n 個の候補
    grouped = []
    for out in outputs:
        grouped.append([o.text for o in out.outputs])
    return grouped

# データ読み込み（例: GSM8K サブセット）
data = load_dataset("gsm8k", "main", split="train[:100]")

pairs = []
for ex in data:
    prompt = ex["question"]
    refs = ex["answer"]
    cand_groups = generate_batch([prompt])[0]
    scored = [(c, judge(c, refs)) for c in cand_groups]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored[0][1] < 0.5:
        continue  # 自動採点が低いものは学習に使わない
    # DPO 用に「勝ち/負け」ペアを作る
    win, lose = scored[0][0], scored[-1][0]
    pairs.append({"prompt": prompt, "chosen": win, "rejected": lose})

train_ds = Dataset.from_list(pairs)

# DPO で自己改善
cfg = DPOConfig(
    output_dir="dpo-self-improve",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_train_epochs=1,
    beta=0.1,
)

trainer = DPOTrainer(
    model=base_model,
    ref_model=base_model,  # 参照モデルに元モデルを使う
    args=cfg,
    train_dataset=train_ds,
)

trainer.train()
trainer.save_model("self-improved-llm")
