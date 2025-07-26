# CPUでも動く、transformersベースの自己改善風スクリプト
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import difflib
import random

model_id = "microsoft/phi-2"  # 軽量モデルで代用
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def judge(output, reference):
    return difflib.SequenceMatcher(None, output.strip(), reference.strip()).ratio()

prompt = "What is the capital of France?\nAnswer:"
reference = "The capital of France is Paris."

# 複数生成（温度を高めに）
candidates = [generator(prompt, max_new_tokens=50, do_sample=True, temperature=1.0)[0]["generated_text"] for _ in range(5)]

# 評価して勝敗
scored = [(text, judge(text, reference)) for text in candidates]
scored.sort(key=lambda x: x[1], reverse=True)

print("\n=== BEST ===")
print(scored[0][0])
print("\n=== WORST ===")
print(scored[-1][0])
