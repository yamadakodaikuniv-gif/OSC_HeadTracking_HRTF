from pythonosc import dispatcher as dp
from pythonosc import osc_server as osc
from scipy.spatial.transform import Rotation as R
import pysofaconventions as pysofa
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal

sofa = pysofa.SOFAFile("RIEC_hrir_subject_055.sofa", "r") 
sourcePositions = sofa.getVariableValue('SourcePosition')
sofa_azimuths = sourcePositions[:, 0]
sofa_elevations = sourcePositions[:, 1]
hrir_all = sofa.getDataIR()

#音源
source_audio, fs = sf.read("white_noise_10s_ref_70dB.wav")

sofa_fs = sofa.getVariableValue('Data.SamplingRate')[0]
print(f"音源のサンプリング周波数:{fs}Hz | SOFAのサンプリング周波数:{sofa_fs}Hz")

offset_z=None
current_index = 0
current_hrir_left = hrir_all[0, 0, :]
current_hrir_right = hrir_all[0, 1, :]
hrir_len = hrir_all.shape[2]

overlap_left = np.zeros(hrir_len - 1)
overlap_right = np.zeros(hrir_len - 1)

def sensor_data(address, *args):
    global offset_z
    try:
        q_x = float(args[0])
        q_y = float(args[1])
        q_z = float(args[2])
        q_w = float(args[3])

        rot = R.from_quat([q_x, q_y, q_z, q_w])
        angles = rot.as_euler('xyz',degrees=True)

        angles_x = angles[0]
        angles_y = angles[1]
        angles_z = angles[2]

        if offset_z is None:
            offset_z = angles_z
            print(f"【システム】正面を{round(offset_z, 1)}度に設定しました")

        head_az = angles_z - offset_z
        head_el = angles_x

        #仮想音源の球面座標
        fixed_source_az = 0.0
        fixed_source_el = 0.0

        target_az = fixed_source_az - head_az
        target_el = fixed_source_el - head_el

        target_az = target_az % 360
        if target_az < 0:
            target_az += 360

        diff_az = (sofa_azimuths - target_az + 180) % 360 - 180
        diff_el = sofa_elevations - target_el

        distances = (diff_az**2) + (diff_el**2)

        best_index=np.argmin(distances)
        global current_index
        current_index = best_index

        nearest_az = sofa_azimuths[best_index]
        nearest_el = sofa_elevations[best_index]

        print(f"現在:{round(head_az, 1)}度 -> 音源は相対的に:{round(target_az, 1)}度 (No.{best_index})")
    except ValueError:
        pass
    except IndexError:
        pass

disp = dp.Dispatcher()
disp.map("/ZIGSIM/QCFSNaDyuqW74_6f/quaternion", sensor_data)

playback_pos = 0

def audio_callback(outdata, frames, time, status):
    global playback_pos, current_index, current_hrir_left, current_hrir_right, overlap_left, overlap_right

    chunk = source_audio[playback_pos : playback_pos + frames]

    if len(chunk) < frames:
        playback_pos = 0
        chunk = np.pad(chunk, (0, frames - len(chunk)))

    target_hrir_left = hrir_all[current_index, 0, :]
    target_hrir_right = hrir_all[current_index, 1, :]

    alpha = 0.87
    current_hrir_left = (alpha*current_hrir_left)+((1.0 - alpha)*target_hrir_left)
    current_hrir_right = (alpha*current_hrir_right)+((1.0 - alpha)*target_hrir_right)

    conv_left = signal.fftconvolve(chunk, current_hrir_left, mode='full')
    conv_right = signal.fftconvolve(chunk, current_hrir_right, mode='full')

    out_left = conv_left[:frames]
    out_right = conv_right[:frames]

    tail_len = len(overlap_left)
    out_left[:tail_len] += overlap_left
    out_right[:tail_len] +=overlap_right

    overlap_left = conv_left[frames:]
    overlap_right = conv_right[frames:]

    outdata[:, 0] = out_left
    outdata[:, 1] = out_right

    playback_pos += frames

server = osc.ThreadingOSCUDPServer(("0.0.0.0", 5005), disp)

print("音響エンジンを起動します")
stream = sd.OutputStream(samplerate=fs, channels=2, callback=audio_callback, blocksize=1024)
stream.start()

print("サーバー起動　スマホからの通信を待っています")
server.serve_forever()
