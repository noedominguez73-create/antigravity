import subprocess
import os

print("Testing Edge TTS execution via subprocess...")

text = "Hola, probando sistema de voz."
voice = "es-MX-JorgeNeural"
output_file = "test_audio.mp3"

cmd = [
    "edge-tts",
    "--text", text,
    "--write-media", output_file,
    "--voice", voice
]

print(f"Command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"Return Code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    
    if os.path.exists(output_file):
        print(f"✅ Auto check: File {output_file} exists created.")
        size = os.path.getsize(output_file)
        print(f"Size: {size} bytes")
    else:
        print("❌ File was not created.")

except Exception as e:
    print(f"❌ Exception: {e}")
