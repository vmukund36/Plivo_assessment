import torch
import os
from transformers import AutoModelForTokenClassification

class JITWrapper(torch.nn.Module):
    """
    Wrapper to force the model to return only logits (Tensor) 
    instead of a dictionary, making it compatible with torch.jit.trace.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # We only care about logits for inference
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def quantize(model_dir):
    print(f"Quantizing model in {model_dir}...")
    
    # 1. Load PyTorch Model
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # 2. Apply Dynamic Quantization (Linear layers -> Int8)
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

    # 3. Wrap the model to ensure it outputs a Tensor, not a Dict
    wrapped_model = JITWrapper(quantized_model)

    # 4. Trace the model
    # Create dummy input matching the expected input shape
    dummy_input_ids = torch.zeros(1, 128, dtype=torch.long)
    dummy_mask = torch.zeros(1, 128, dtype=torch.long)
    
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapped_model, (dummy_input_ids, dummy_mask))

    # 5. Save
    save_path = os.path.join(model_dir, "quantized_model.pt")
    torch.jit.save(traced_model, save_path)
    print(f"Quantized model saved to {save_path}")

if __name__ == "__main__":
    quantize("/home/ic40962/SafetyRepLLM/Plivo/pii_ner_assignment/src/out")