import os

def generate_mem_file(tflite_path, mem_path):
    print(f"Reading {tflite_path}...")
    
    # Read the binary TFLite file
    with open(tflite_path, "rb") as f:
        model_data = f.read()

    # Create the Verilog .mem file
    with open(mem_path, "w") as f:
        for byte in model_data:
            # Write each byte as a 2-character uppercase hex string on its own line
            f.write(f"{byte:02X}\n")

    print(f"Success! Wrote {len(model_data)} bytes to {mem_path} for Vivado.")

if __name__ == "__main__":
    tflite_path = "chainsaw_detector_quantized.tflite"
    mem_path = "weights.mem"
    generate_mem_file(tflite_path, mem_path)