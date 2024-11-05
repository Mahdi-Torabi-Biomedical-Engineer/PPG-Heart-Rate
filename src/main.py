import os
import logging
import warnings
import socket
import numpy as np
import tensorflow as tf
import cv2
import sklearn.preprocessing as skp
import tflib
import module
import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt

# Disable TensorFlow warnings and set backend float precision
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore', category=FutureWarning)
tf.keras.backend.set_floatx('float64')
tf.autograph.set_verbosity(0)

# Set up paths and directories
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
print(f"Running script: {filename}")

def connect(deviceID, serverAddress='127.0.0.1', serverPort=28000, bufferSize=4096):
    """
    Establishes a connection with the server for data streaming.
    Params:
        deviceID: Unique identifier for the device.
        serverAddress: IP address of the server.
        serverPort: Port to connect to.
        bufferSize: Size of data buffer.
    Returns:
        A connected socket instance.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    
    print("Connecting to server...")
    s.connect((serverAddress, serverPort))
    print("Connected to server.\n")
    
    # List available devices
    s.send("device_list\r\n".encode())
    response = s.recv(bufferSize)
    print("Available devices:", response.decode("utf-8"))
    
    # Connect to specific device
    print(f"Connecting to device {deviceID}...")
    s.send(f"device_connect {deviceID}\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))
    
    # Pause data receiving for setup
    s.send("pause ON\r\n".encode())
    response = s.recv(bufferSize)
    print("Data receiving paused:", response.decode("utf-8"))
    
    return s

def subscribe_to_data(s, acc=False, bvp=True, gsr=False, tmp=False, bufferSize=4096):
    """
    Subscribes to data channels as specified.
    Params:
        s: The connected socket instance.
        acc, bvp, gsr, tmp: Booleans indicating which data to subscribe to.
        bufferSize: Size of data buffer.
    Returns:
        The socket instance with active subscriptions.
    """
    channels = {'acc': acc, 'bvp': bvp, 'gsr': gsr, 'tmp': tmp}
    for channel, enabled in channels.items():
        if enabled:
            print(f"Subscribing to {channel.upper()} data...")
            s.send(f"device_subscribe {channel} ON\r\n".encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

    # Resume data streaming
    print("Resuming data streaming...")
    s.send("pause OFF\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))
    
    return s

@tf.function
def sample_P2E(P, model):
    """
    Generates synthetic ECG data from PPG data using the specified model.
    Params:
        P: Input PPG data.
        model: Pre-trained model for PPG-to-ECG transformation.
    Returns:
        Generated ECG data.
    """
    fake_ecg = model(P, training=False)
    return fake_ecg

# Configuration parameters
ecg_sampling_freq = 128
ppg_sampling_freq = 128
window_size = 4
ecg_segment_size = ecg_sampling_freq * window_size
ppg_segment_size = ppg_sampling_freq * window_size
model_dir = 'path/to/model'
output_dir = 'path/to/output'
os.makedirs(output_dir, exist_ok=True)
ep = open(os.path.join(output_dir, "ecg_ppg_recordings.txt"), "w+")

bufferSize = 10240
deviceID = 'AAAAAA'

# Initialize connection and subscriptions
e4 = connect(deviceID)
e4 = subscribe_to_data(e4)

# Load the model for PPG to ECG conversion
Gen_PPG2ECG = module.generator_attention()
tflib.Checkpoint(dict(Gen_PPG2ECG=Gen_PPG2ECG), model_dir).restore()

data_list = []
PPG = []
ECG = []
tb_step = 0

# TensorBoard writer for real-time monitoring
train_summary_writer = tf.summary.create_file_writer(os.path.join(output_dir, 'summary'))

# Main data processing loop
with train_summary_writer.as_default():
    try:
        while True:
            try:
                # Receive and process PPG data
                response = e4.recv(bufferSize).decode("utf-8")
                samples = response.split("\n")
                print(f"Receiving PPG data at {datetime.now()}")
                
                for sample in samples[1:-1]:  # Exclude first and last samples (possibly incomplete)
                    data = float(sample.split()[2].replace(',', '.'))
                    data_list.append(data)
                
                # Process the collected PPG data if buffer is full
                if len(data_list) >= 256:
                    print(f"Processing PPG data at {datetime.now()}")
                    x_ppg = np.array(data_list[:256])
                    data_list = data_list[256:]
                    
                    # Preprocessing steps
                    x_ppg = cv2.resize(x_ppg, (1, ppg_segment_size), interpolation=cv2.INTER_LINEAR).reshape(1, -1)
                    x_ppg = preprocessing.filter_ppg(x_ppg, ppg_sampling_freq)
                    x_ppg = skp.minmax_scale(x_ppg, (-1, 1), axis=1)
                    
                    # Generate ECG data from PPG
                    x_ecg = sample_P2E(x_ppg, Gen_PPG2ECG).numpy()
                    x_ecg = preprocessing.filter_ecg(x_ecg, ecg_sampling_freq)
                    x_ppg, x_ecg = x_ppg.reshape(-1), x_ecg.reshape(-1)
                    
                    print(f"Generated ECG data at {datetime.now()}")
                    
                    # Log data for TensorBoard visualization
                    for point in range(len(x_ecg)):
                        tf.summary.scalar('FECG', x_ecg[point], step=tb_step)
                        tf.summary.scalar('RPPG', x_ppg[point], step=tb_step)
                        tb_step += 1
                    
                    # Append results for storage
                    ECG.append(x_ecg)
                    PPG.append(x_ppg)
                    np.savetxt(ep, np.c_[x_ecg, x_ppg], fmt='%1.8f %1.8f')

            except Exception as e:
                print("Error encountered. Reconnecting...")
                e4 = connect(deviceID)
                e4 = subscribe_to_data(e4)

    except KeyboardInterrupt:
        print("Disconnecting from device...")
        e4.send("device_disconnect\r\n".encode())
        e4.close()
        ep.close()

# TensorBoard command for real-time monitoring
# tensorboard --logdir=./output/realtime --host localhost --port 8088 --samples_per_plugin scalars=0 --reload_interval 1
