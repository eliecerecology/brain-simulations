import numpy as np
from scipy.io.wavfile import read, write

# Parameters
sample_rate = 44100  # Hz (standard audio)
bit_duration = 0.02  # seconds per bit (~50 baud, ~1.76 s total)
f0 = 1200.0  # Hz for bit '0'
f1 = 2200.0  # Hz for bit '1'

# Text and bit stream (single message for decode test)
text = "hello world mother fucka"
bits = ''.join(format(ord(c), '08b') for c in text)
print("Bits to encode:", bits)

# Generate FSK signal
t = np.linspace(0, bit_duration, int(sample_rate * bit_duration), endpoint=False)
signal = np.array([])
for bit in bits:
    freq = f1 if bit == '1' else f0
    tone = np.sin(2 * np.pi * freq * t)
    signal = np.concatenate((signal, tone))

# Normalize to 16-bit integer range and write
signal = signal / np.max(np.abs(signal))
wav_path = "hello_world_fsk.wav"
write(wav_path, sample_rate, (signal * 32767).astype(np.int16))
print(f"Generated {wav_path}")

# ----- Decode FSK signal back to text -----
def decode_fsk_wav(filepath, sample_rate, bit_duration, f0, f1):
    """Decode an FSK WAV file back to a bit string, then to text."""
    sr, data = read(filepath)
    if sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, got {sr}")
    # Convert to float in [-1, 1]
    sig = data.astype(np.float64) / 32768.0
    n_per_bit = int(sample_rate * bit_duration)
    n_bits = len(sig) // n_per_bit
    bits = []
    for i in range(n_bits):
        chunk = sig[i * n_per_bit : (i + 1) * n_per_bit]
        n = len(chunk)
        fft_vals = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)
        idx_f0 = np.argmin(np.abs(freqs - f0))
        idx_f1 = np.argmin(np.abs(freqs - f1))
        bit = "1" if np.abs(fft_vals[idx_f1]) >= np.abs(fft_vals[idx_f0]) else "0"
        bits.append(bit)
    bit_str = "".join(bits)
    # Convert bits to bytes then to string (8 bits per character)
    chars = []
    for j in range(0, len(bit_str), 8):
        byte_bits = bit_str[j : j + 8]
        if len(byte_bits) < 8:
            break
        chars.append(chr(int(byte_bits, 2)))
    return "".join(chars).rstrip("\x00")

decoded = decode_fsk_wav(wav_path, sample_rate, bit_duration, f0, f1)
print("Decoded from WAV:  ", repr(decoded))