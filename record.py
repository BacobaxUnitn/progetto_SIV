import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# Parameters for recording
CHUNK = 1024            # Number of audio samples per buffer
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1            # Mono audio
RATE = 44100            # Sampling rate in Hz
RECORD_SECONDS = 5      # Duration of recording in seconds

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the audio stream for recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Read data from the stream
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Convert the byte data to numpy array
audio_data = b''.join(frames)
audio_array = np.frombuffer(audio_data, dtype=np.int16)

# Create a time axis for the time-domain plot
time_axis = np.linspace(0, len(audio_array) / RATE, num=len(audio_array))

# Plot the time-domain signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, audio_array)
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Compute the Fourier Transform for the frequency-domain plot
fft_data = np.fft.fft(audio_array)
fft_freq = np.fft.fftfreq(len(audio_array), d=1/RATE)

# Use only the positive half of the spectrum
positive_freqs = fft_freq[:len(fft_freq)//2]
fft_magnitude = np.abs(fft_data[:len(fft_data)//2])

# Plot the frequency-domain signal
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, fft_magnitude)
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()