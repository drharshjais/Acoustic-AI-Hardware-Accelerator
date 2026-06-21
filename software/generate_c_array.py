import os

def tflite_to_c_array(tflite_path, c_file_path, array_name="model_data"):
    # Read the binary TFLite file
    with open(tflite_path, "rb") as f:
        model_data = f.read()

    # Create the C header file
    with open(c_file_path, "w") as f:
        f.write(f"// Automatically generated C array from {tflite_path}\n")
        f.write("#ifndef CHAINSAW_MODEL_H\n")
        f.write("#define CHAINSAW_MODEL_H\n\n")
        
        f.write(f"const unsigned int {array_name}_len = {len(model_data)};\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
        # Write the bytes as hex values, formatting them neatly
        hex_data = [f"0x{byte:02x}" for byte in model_data]
        
        # Write 12 bytes per line for readability
        for i in range(0, len(hex_data), 12):
            line = ", ".join(hex_data[i:i+12])
            f.write(f"  {line},\n")
            
        f.write("};\n\n")
        f.write("#endif // CHAINSAW_MODEL_H\n")

    print(f"Successfully wrote {len(model_data)} bytes to {c_file_path}")

if __name__ == "__main__":
    tflite_path = "chainsaw_detector_quantized.tflite"
    c_file_path = "chainsaw_model.h"
    tflite_to_c_array(tflite_path, c_file_path)