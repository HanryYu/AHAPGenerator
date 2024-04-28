import os
import datetime
import json
import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

class AudioHapticPattern:
    def __init__(self):
        self.pattern_data = {
            "Version": 1.0,
            "Metadata": {
                "Project": "Basis",
                "CreationDate": str(datetime.datetime.now()),
                "Description": "Generated AHAP file from audio analysis.",
                "Author": "Ryan Du"
            },
            "Events": []
        }



    def add_haptic_event(self, event_type, timestamp, params, duration=None, waveform_path=None):
        event = {
            "Event": {
                "Timestamp": timestamp,
                "Type": event_type,
                "Parameters": params
            }
        }
        if duration is not None:
            event["Event"]["Duration"] = duration
        if waveform_path is not None:
            event["Event"]["Waveform"] = waveform_path
        self.pattern_data["Events"].append(event)



    def add_transient_haptic(self, timestamp, intensity=0.5, sharpness=0.5):
        params = [
            {"ID": "Intensity", "Value": intensity},
            {"ID": "Sharpness", "Value": sharpness}
        ]
        self.add_haptic_event("HapticTransient", timestamp, params)



    def add_continuous_haptic(self, timestamp, duration=1, intensity=0.5, sharpness=0.5):
        params = [
            {"ID": "Intensity", "Value": intensity},
            {"ID": "Sharpness", "Value": sharpness}
        ]
        self.add_haptic_event("HapticContinuous", timestamp, params, duration=duration)

    def add_custom_audio_event(self, timestamp, audio_path, volume=0.75):
        params = [{"ID": "Volume", "Value": volume}]
        self.add_haptic_event("AudioCustom", timestamp, params, waveform_path=audio_path)



    def add_control_point_curve(self, param_id, start_time, control_points):
        curve = {
            "ControlCurve": {
                "ParameterID": param_id,
                "StartTime": start_time,
                "ControlPoints": control_points
            }
        }
        self.pattern_data["Events"].append(curve)



    def display_pattern(self):
        print(json.dumps(self.pattern_data, indent=4))



    def save_pattern(self, file_name, directory):
        with open(os.path.join(directory, file_name), 'w') as file:
            json.dump(self.pattern_data, file, indent=4)