import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time

MODEL_PATH = "chainsaw_detector_quantized.tflite"
SAMPLE_RATE = 22050  

# FIXED: 157 frames at a 512 hop_length requires exactly 3.645 seconds of audio!
DURATION = 3.65         
THRESHOLD = -80  # A safe mid-range threshold for int8 (-128 to 127)

print("Booting up Smart Forest Sensor...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

expected_shape = input_details[0]['shape']
in_scale, in_zero = input_details[0]['quantization']

def process_audio_and_predict(audio_data):
    vol = np.max(np.abs(audio_data))
    
    # WAKE UP CIRCUIT: Ignore silence
    if vol < 0.005:
        print(f"[DEBUG] Vol: {vol:.4f} | Output: [SLEEP] ", end=" | ")
        return -128  

    # 1. Create Spectrogram with STRICT matching parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data.flatten(), 
        sr=SAMPLE_RATE, 
        n_mels=64,
        n_fft=1024,       # Standard STFT window
        hop_length=512    # Standard hop length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. We now have enough audio! Just slice exactly the first 157 frames.
    mel_spec_db = mel_spec_db[:, :157]

    # 3. Match Hardware Shape
    if expected_shape[1] == 157 and expected_shape[2] == 64:
        mel_spec_db = mel_spec_db.T 

    # 4. Apply the Model's Exact Quantization Math
    if in_scale > 0:
        input_data = (mel_spec_db / in_scale) + in_zero
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        input_data = np.interp(np.clip(mel_spec_db, -80, 0), (-80, 0), (-128, 127)).astype(np.int8)

    input_data = input_data.reshape(expected_shape)

    # 5. Predict
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print(f"[DEBUG] Vol: {vol:.4f} | Output: {raw_prediction} ", end=" | ")
    
    if len(raw_prediction) > 1:
        return raw_prediction[1] 
    return int(np.max(raw_prediction))

print("\n" + "="*55)
print(" 🌲 ACOUSTIC DEFORESTATION SENSOR ONLINE 🌲")
print("="*55 + "\n")

try:
    while True:
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait() 

        score = process_audio_and_predict(recording)

        if score > THRESHOLD: 
            print("\n\n" + "🔴"*20)
            print(" 🚨 THREAT DETECTED! CHAINSAW ACOUSTIC MATCH 🚨")
            print(f" Neural Network Score: {score}")
            print(" >>> TRANSMITTING ALARM TO RANGER STATION VIA LORA <<<")
            print("🔴"*20 + "\n")
            time.sleep(2) 
        else:
            print(f"[SYSTEM] Status: Normal forest ambient noise.")

except KeyboardInterrupt:
    print("\n\n[SYSTEM] Sensor deactivated.")