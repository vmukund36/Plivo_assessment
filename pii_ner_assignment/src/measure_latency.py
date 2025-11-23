import json
import time
import argparse
import statistics
import torch
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cpu") # Default to CPU for standardized testing
    args = ap.parse_args()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)

    # 2. Check for Quantized Model
    # We prioritize the quantized model if it exists to measure the optimized speed.
    q_path = os.path.join(args.model_dir, "quantized_model.pt")
    if os.path.exists(q_path):
        print(f"Loading Quantized JIT Model from {q_path}...")
        model = torch.jit.load(q_path)
        is_quantized = True
    else:
        print(f"Loading Standard Model from {args.model_dir}...")
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        is_quantized = False

    model.to(args.device)
    model.eval()

    # 3. Load Data
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found in input file.")
        return

    times_ms = []

    # 4. Warmup
    # We run a few passes to ensure caches are warm and JIT optimization (if any) settles.
    print("Warming up...")
    for _ in range(10):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)
        
        with torch.no_grad():
            if is_quantized:
                # JIT wrapper returns tensor directly
                _ = model(input_ids, attention_mask)
            else:
                # Standard model returns ModelOutput
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # 5. Measure Latency
    print(f"Running latency test over {args.runs} iterations...")
    for i in range(args.runs):
        t = texts[i % len(texts)]
        
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)

        start = time.perf_counter()
        with torch.no_grad():
            if is_quantized:
                _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.perf_counter()
        
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()