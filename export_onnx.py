import os
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

if __name__ == "__main__":
    output_dir = "models"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
 
    output_fpath = os.path.join(output_dir, "gte-kor-turbo.onnx")
    convert(framework="pt", model="sjy-dv/gte-kor-turbo", output=Path(output_fpath), opset=15)