import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os

# --- Logic Filters for Precision ---
def validate_span(text, label):
    """Returns False if the span is obviously a false positive based on rules."""
    text = text.lower().strip()
    
    if label == "EMAIL":
        # Must contain 'at' or '@'. 
        # In noisy STT, it might be "example at gmail dot com"
        if "at" not in text and "@" not in text:
            return False
        if len(text) < 5: 
            return False

    if label == "PHONE":
        # Should contain at least some digits or number-words
        # Simple check: verify length or specific phone keywords
        # Discard purely alphabetical common words
        if len(text) < 3: return False
        # If it's just a common word like "phone", "number", ignore
        if text in ["phone", "number", "call", "mobile"]:
            return False

    if label == "CREDIT_CARD":
        # Should be mostly digits or number words. 
        if len(text) < 8: return False

    return True
# -----------------------------------

def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0: continue # Skip special tokens
        
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=128) # Reduced for speed
    ap.add_argument("--device", default="cpu") # Force CPU for consistent behavior with quantization
    args = ap.parse_args()

    # Check if quantized model exists
    if os.path.exists(os.path.join(args.model_dir, "quantized_model.pt")):
        print("Loading Quantized Model...")
        model = torch.jit.load(os.path.join(args.model_dir, "quantized_model.pt"))
        is_quantized = True
    else:
        print("Loading Standard Model...")
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        is_quantized = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                if is_quantized:
                    # The JIT Wrapper returns the logits tensor directly
                    logits = model(input_ids, attention_mask)
                else:
                    # Standard HF model returns an object; access .logits
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits[0]
                
                # If logits is still a batch (1, seq_len, labels), take [0]
                if len(logits.shape) == 3:
                    logits = logits[0]

                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            raw_spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            
            for s, e, lab in raw_spans:
                span_text = text[s:e]
                # APPLY VALIDATION FILTER
                if validate_span(span_text, lab):
                    ents.append({
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    })
            
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()