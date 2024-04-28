import librosa
import matplotlib.pyplot as plt
from audio_analyzer import SoundProcessor
from haptic_pattern_creator import AudioHapticPattern

def process_audio(file_path):
    audio_data, rate = librosa.load(file_path)
    duration = len(audio_data) / rate

    analyzer = SoundProcessor(rate)

    # Separate harmonic and percussive components
    harmonic_part, percussive_part = librosa.effects.hpss(audio_data)

    beats_info = analyzer.detect_beats(percussive_part, rate)
    onsets_info, onsets_strength = analyzer.detect_onsets(audio_data, rate)

    mel_spectrogram = analyzer.generate_melspectrogram(audio_data, rate)
    harmonic_mel = analyzer.generate_melspectrogram(harmonic_part, rate, plot_title="Harmonic Mel Spectrogram")
    percussive_mel = analyzer.generate_melspectrogram(percussive_part, rate, plot_title="Percussive Mel Spectrogram")

    chroma_features = {
        'harmonic': analyzer.create_chromagram(harmonic_part, rate, plot_title="Harmonic Chromagram"),
        'original': analyzer.create_chromagram(audio_data, rate, plot_title="Original Chromagram"),
        'percussive': analyzer.create_chromagram(percussive_part, rate, plot_title="Percussive Chromagram")
    }

    mfcc_features = {
        'original': analyzer.calculate_mfcc(mel_spectrogram, rate),
        'harmonic': analyzer.calculate_mfcc(harmonic_mel, rate),
        'percussive': analyzer.calculate_mfcc(percussive_mel, rate)
    }

    mfcc_synced = analyzer.sync_melody_to_beats(mfcc_features['original'], beats_info[1], rate)

    chroma_synced = {
        'harmonic': analyzer.sync_chroma_to_beats(chroma_features['harmonic'], beats_info[1], rate),
        'percussive': analyzer.sync_chroma_to_beats(chroma_features['percussive'], beats_info[1], rate),
        'original': analyzer.sync_chroma_to_beats(chroma_features['original'], beats_info[1], rate)
    }

    haptic_creator = AudioHapticPattern()

    for time, strength in zip(onsets_info, onsets_strength):
        haptic_creator.add_transient_haptic(time, strength / 5, strength / 5)

    haptic_creator.save_pattern(f'{file_path[:-4]}.ahap', '.')

if __name__ == '__main__':
    audio_file_path = input("Enter the path to the audio file: ")
    process_audio(audio_file_path)
