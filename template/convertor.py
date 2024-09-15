from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.onnx import export
from pathlib import Path
import torch

# Function to convert Hugging Face models to ONNX format
def convert_to_onnx(model_name, onnx_output_path):
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Prepare an example input for tracing
    text = "This is an example text for summarization."
    inputs = tokenizer(text, return_tensors="pt")

    # Specify the ONNX file path
    onnx_path = Path(onnx_output_path)

    # Export the model to ONNX format
    export(model=model,
           tokenizer=tokenizer,
           opset=11,  # ONNX opset version
           output=onnx_path,
           input_names=["input_ids", "attention_mask"],
           output_names=["output"],
           dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"},
                         "attention_mask": {0: "batch_size", 1: "seq_len"},
                         "output": {0: "batch_size", 1: "seq_len"}})

    print(f"Model converted and saved to: {onnx_output_path}")

# Example models that you can convert (BART, T5, PEGASUS)
model_names = {
    "BART": "facebook/bart-large-cnn",
    "T5": "t5-large",
    "PEGASUS": "google/pegasus-xsum"
}

# Path where the ONNX model will be saved
output_dir = "./onnx_models"

# Convert BART, T5, and PEGASUS models to ONNX
for model_key, model_name in model_names.items():
    convert_to_onnx(model_name, f"{output_dir}/{model_key}.onnx")