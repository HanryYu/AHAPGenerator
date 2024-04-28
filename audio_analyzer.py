import os
import datetime
import json
import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

class SoundProcessor:
    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate



    def scale_amplitude(self, signal):
        factor = (15 * 0.005 ** signal + 1)
        result = signal * factor
        if result < 0.05:
            result *= 6
        elif result < 0.1:
            result *= 4
        elif result < 0.2:
            result *= 2.5
        return result
    


    def compute_rmse(self, samples):
        step_size = 256
        window_size = 512
        power = np.array([
            sum(abs(samples[i:i+window_size]**2))
            for i in range(0, len(samples), step_size)
        ])
        rmse_values = librosa.feature.rms(
            y=samples, frame_length=window_size, hop_length=step_size, center=True)[0]
        return rmse_values
    


    def detect_onsets(self, audio_data, sample_rate):
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        return onset_times, onset_strength



    def process_audio(file_path):
        audio_data, rate = librosa.load(file_path)
        duration = len(audio_data) / rate

        analyzer = SoundProcessor(rate)

        # Separate harmonic and percussive components
        harmonic_part, percussive_part = librosa.effects.hpss(audio_data)

        beats_info = analyzer.detect_beats(percussive_part, rate)
        onsets_info, onsets_strength = analyzer.detect_onsets(audio_data)

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

        mfcc_synced = analyzer.sync_melody_to_beats(mfcc_features['original'], beats_info[1])

        plt.figure(figsize=(40, 6))
        plt.plot(onsets_strength)
        plt.vlines(onsets_info, 0, max(onsets_strength), "red", alpha=0.25)

        chroma_synced = {
            'harmonic': analyzer.sync_chroma_to_beats(chroma_features['harmonic'], beats_info[1], rate),
            'percussive': analyzer.sync_chroma_to_beats(chroma_features['percussive'], beats_info[1], rate),
            'original': analyzer.sync_chroma_to_beats(chroma_features['original'], beats_info[1], rate)
        }

        haptic_creator = AudioHapticPattern()

        for time, strength in zip(onsets_info, onsets_strength):
            haptic_creator.add_transient_haptic(time, strength / 5, strength / 5)

        haptic_creator.save_pattern(f'{file_path[:-4]}.ahap', '.')


    
    def sync_melody_to_beats(self, melody_features, beat_times, sample_rate=22050, step_size=512, show_plot=False):
        beat_indices = librosa.time_to_frames(beat_times, sr=sample_rate, hop_length=step_size)
        synced_melody = librosa.util.sync(melody_features, beat_indices)
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            librosa.display.specshow(melody_features)
            plt.title('Melody Features')
            plt.colorbar()
            plt.subplot(2, 1, 2)
            librosa.display.specshow(synced_melody, x_axis='time', x_coords=librosa.frames_to_time(beat_indices, sr=sample_rate, hop_length=step_size))
            plt.title('Beat-synced Melody Features')
            plt.colorbar()
            plt.tight_layout()
        return synced_melody
    


    def sync_chroma_to_beats(self, chroma_data, beat_times, sample_rate, step_size=512, show_plot=False):
        beat_indices = librosa.time_to_frames(beat_times, sr=sample_rate, hop_length=step_size)
        chroma_synced = librosa.util.sync(chroma_data, beat_indices, aggregate=np.median)
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            librosa.display.specshow(chroma_data, sr=sample_rate, hop_length=step_size, y_axis='chroma', x_axis='time')
            plt.title('Original Chroma')
            plt.colorbar()
            plt.subplot(2, 1, 2)
            librosa.display.specshow(chroma_synced, y_axis='chroma', x_axis='time', x_coords=librosa.frames_to_time(beat_indices, sr=sample_rate, hop_length=step_size))
            plt.title('Beat-synced Chroma')
            plt.colorbar()
            plt.tight_layout()
        return chroma_synced
    


    def generate_melspectrogram(self, audio_data, sample_rate, show_plot=False, plot_title='Mel Spectrogram'):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        if show_plot:
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
            plt.title(plot_title)
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
        return mel_spectrogram
    


    def detect_beats(self, percussive_signal, sample_rate, show_plot=False, spectrogram=None):
        beat_tempo, beat_positions = librosa.beat.beat_track(y=percussive_signal, sr=sample_rate, units="time")
        if show_plot:
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
            plt.vlines(librosa.frames_to_time(beat_positions), 1, 0.5 * sample_rate, colors='w', linestyles='-', linewidth=2, alpha=0.5)
            plt.axis('tight')
            plt.colorbar(format='%+02.0f dB')
            plt.tight_layout()
            plt.show()
        return beat_tempo, beat_positions



    def create_chromagram(self, audio_data, sample_rate, show_plot=False, plot_title="Chromagram"):
        chroma_features = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate, bins_per_octave=36)
        if show_plot:
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(chroma_features, sr=sample_rate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
            plt.title(plot_title)
            plt.colorbar()
            plt.tight_layout()
        return chroma_features
    

    
    def calculate_mfcc(self, spectrogram, sample_rate, show_plot=False):
        mfcc_features = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram, ref=np.max), n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc_features)
        delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(3, 1, 1)
            librosa.display.specshow(mfcc_features)
            plt.ylabel('MFCC')
            plt.colorbar()
            plt.subplot(3, 1, 2)
            librosa.display.specshow(delta_mfcc)
            plt.ylabel('MFCC-$\Delta$')
            plt.colorbar()
            plt.subplot(3, 1, 3)
            librosa.display.specshow(delta2_mfcc, sr=sample_rate, x_axis='time')
            plt.ylabel('MFCC-$\Delta^2$')
            plt.colorbar()
            plt.tight_layout()
        combined_mfcc = np.vstack([mfcc_features, delta_mfcc, delta2_mfcc])
        return combined_mfcc
