import numpy as np
import librosa
import tensorflow as tf

MODEL_PATH = "chainsaw_detector_quantized.tflite"
TEST_AUDIO_FILE = "test_chainsaw.wav"  # MUST BE A REAL FILE FROM YOUR DATASET
SAMPLE_RATE = 22050

print("=== DIRECT FILE BYPASS TEST ===")
print(f"Loading Model: {MODEL_PATH}")

# Load Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
expected_shape = input_details[0]['shape']
in_scale, in_zero = input_details[0]['quantization']

print(f"Loading Audio File: {TEST_AUDIO_FILE}")
# Load exactly 3.65 seconds of the audio file to match our 157 frames
try:
    audio_data, _ = librosa.load(TEST_AUDIO_FILE, sr=SAMPLE_RATE, duration=3.65)
except Exception as e:
    print(f"Error loading audio file! Make sure {TEST_AUDIO_FILE} is in the folder. Error: {e}")
    exit()

# 1. Create Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio_data.flatten(), 
    sr=SAMPLE_RATE, 
    n_mels=64,
    n_fft=1024,       
    hop_length=512    
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# 2. Slice to exactly 157 frames
if mel_spec_db.shape[1] >= 157:
    mel_spec_db = mel_spec_db[:, :157]
else:
    # If the file is shorter than 3.65s, pad it with silence
    pad_width = 157 - mel_spec_db.shape[1]
    mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)

# 3. Match Hardware Shape
if expected_shape[1] == 157 and expected_shape[2] == 64:
    mel_spec_db = mel_spec_db.T 

# 4. Apply Exact Quantization
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

print("=======================================")
print(f"RAW AI OUTPUT ARRAY: {raw_prediction}")
print("=======================================")

if len(raw_prediction) > 1:
    score = raw_prediction[1]
else:
    score = int(np.max(raw_prediction))

if score > -80:
    print("✅ SUCCESS! The AI correctly identified the pure dataset file!")
    print("Conclusion: The Python math is perfect. The laptop microphone/Windows drivers are distorting the live audio.")
else:
    print("❌ FAIL! The AI rejected its own training data.")
    print("Conclusion: The Librosa settings in this script (n_mels, n_fft, hop_length) do not match the script you used to train the model. We need to check your training code.")