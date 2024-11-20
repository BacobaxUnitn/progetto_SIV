import numpy as np
import pyaudio

def freq_to_note_name(frequency):
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)  # Frequency of C0

    if frequency == 0:
        return None

    # Calculate the number of semitones from C0
    h = round(12 * np.log2(frequency / C0))
    octave = h // 12
    n = h % 12
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = note_names[n] + str(octave)
    return note_name

def get_dominant_freq(data, fs):

    window = np.hanning(len(data))
    data = data * window

    # Compute FFT
    fft_spectrum = np.fft.rfft(data)
    fft_magnitude = np.abs(fft_spectrum)
    freqs = np.fft.rfftfreq(len(data), d=1/fs)

    # Ignore frequencies below 50 Hz
    idx = np.where(freqs > 50)
    freqs = freqs[idx]
    fft_magnitude = fft_magnitude[idx]

    # Find the peak frequency
    peak_idx = np.argmax(fft_magnitude)
    peak_freq = freqs[peak_idx]
    return peak_freq

def main():
    fs = 44100  # Sample rate
    CHUNK = 1024  # Number of frames per buffer

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening... Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert byte data to numpy array
            data = np.frombuffer(data, dtype=np.int16)
            freq = get_dominant_freq(data, fs)
            note = freq_to_note_name(freq)
            if note:
                print(f"Frequency: {freq:.2f} Hz, Note: {note}")
            else:
                print("No note detected")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()