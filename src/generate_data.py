#!/usr/bin/env python
"""
@name: generate_data.py
@desc: generate some data
@auth: Tim O'Brien
@date: Feb. 18th, 2016
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import util.evaluation as ev
import util.datapaths as dpath

# Debug?
DEBUG_PLOT = False

# Set some params
FS = 44100 # Enforce 44.1 kHz sample rate
N_FFT = 2048
HOP_LENGTH = int(N_FFT/2) # 50% overlap
N_MFCC = 13
N_MEL = 128

DB_LOW = -250.0 # silence in dB

T_CONTEXT = 3 # seconds of context for our features
N_FRAME_CONTEXT = librosa.time_to_frames(
    T_CONTEXT,
    sr=FS, hop_length=HOP_LENGTH, n_fft=N_FFT
    ) +1
# 64 frames on either side, for context

BOUNDARY_KERNEL = signal.gaussian(N_FRAME_CONTEXT, std=32) # For smoothing our y
#BOUNDARY_KERNEL = np.ones(N_FRAME_CONTEXT)

DTYPE = 'float32'

# FOR USE ON AMAZON EC2 AFTER COPYING FROM S3
#DATADIR = os.path.abspath(os.path.join('/mnt','audio'))
#SALAMIDIR = os.path.abspath(os.path.join('/mnt','salami', 'salami-data-public'))

# FOR USE ON WINDOWS MACHINE
#DATADIR = os.path.abspath('F:\salami-audio')
#SALAMIDIR = os.path.abspath('F:\salami-data-public')
#OUTPUTDIR = os.path.abspath('F:\CNNM-output-data')

# FOR USE ON CCRMA NETWORK
#DATADIR = os.path.abspath('/user/t/tsob/Documents/cs231n/proj/data')
#SALAMIDIR = os.path.abspath('/usr/ccrma/media/databases/mir_datasets/salami/salami-data-public')
#OUTPUTDIR = os.path.abspath('/zap/tsob/audio')

# My local machine
DATADIR = os.path.abspath('/home/franck/TRASH/SALAMI')
SALAMIDIR = os.path.abspath('/home/franck/TRASH/salami-data-public')
OUTPUTDIR = os.path.abspath('/home/franck/TRASH/cnn-music-structure/outputs')

# One big list of the valid SALAMI ids
# SIDS = [1258, 1522, 1491, 1391, 986,  1392, 1562, 1006, 1303, 1514, 982,  1095,
#         1130, 1216, 1204, 1536, 1492, 1230, 1503, 1096, 1220, 996,  976,  1010,
#         1120, 1064, 1292, 1008, 1431, 1206, 1074, 1356, 1406, 1559, 1566, 1112,
#         1278, 1540, 1423, 1170, 1372, 1014, 1496, 1327, 1439, 1528, 1311, 1226,
#         1138, 1016, 1364, 1484, 1338, 1254, 968,  998,  1302, 1075, 1018, 1166,
#         1239, 1080, 1032, 1447, 984,  1382, 1284, 1043, 1378, 1467, 1038, 1499,
#         1059, 1534, 1283, 1352, 1524, 1428, 1502, 1088, 1236, 1543, 1475, 1551,
#         990,  1589, 1282, 1459, 1379, 1542, 1131, 1460, 1050, 1128, 991,  1560,
#         1139, 1527, 1270, 1450, 1348, 1331, 1091, 1060, 1015, 1501, 1023, 1200,
#         1340, 1579, 1287, 1062, 1251, 1424, 1516, 1448, 1597, 1575, 1376, 1511,
#         1164, 1548, 1555, 1594, 1224, 1470, 1068, 1007, 1104, 1343, 1234, 1152,
#         1108, 1079, 1212, 972,  1190, 1271, 1136, 1300, 1003, 1103, 1434, 958,
#         1082, 1046, 1326, 1518, 999,  1388, 1472, 1507, 1036, 1316, 1274, 1198,
#         1083, 1435, 1387, 1587, 1572, 1290, 1565, 1504, 1127, 1146, 1462, 1268,
#         1094, 1520, 1366, 1347, 1483, 1517, 1319, 1092, 1498, 971,  1044, 1034,
#         1223, 1346, 1532, 1494, 1123, 1299, 1370, 1298, 1155, 1574, 1240, 1235,
#         1264, 1183, 1211, 1586, 1106, 1275, 1027, 1488, 1360, 1490, 1076, 1306,
#         1580, 1259, 1592, 1280, 1547, 1114, 1119, 1322, 1446, 1359, 1058, 1011,
#         1443, 1307, 1098, 1351, 1598, 1180, 1419, 1508, 995,  1550, 1051, 1194,
#         1215, 1247, 1395, 1159, 1531, 1432, 1396, 1276, 1055, 1334, 1272, 1374,
#         1355, 1390, 1022, 1571, 967,  1557, 1286, 1228, 975,  1024, 1314, 1158,
#         988,  1039, 1182, 955,  1564, 1279, 1544, 1332, 1294, 1308, 1515, 962,
#         1420, 1596, 1163, 1047, 1584, 1026, 1436, 1455, 1476, 1403, 1072, 1330,
#         1244, 1000, 1510, 1573, 994,  1028, 1549, 1179, 1162, 1552, 1238, 1371,
#         1438, 992,  1124, 1367, 1111, 1590, 980,  1242, 1567, 1556, 1054, 1354,
#         1539, 1116, 1148, 1004, 1533, 1232, 1339, 1324, 1291, 978,  1048, 1263,
#         1582, 1315, 1176, 1248, 1509, 1219, 1407, 1400, 1243, 1172, 1442, 1218,
#         1363, 1090, 1067, 1202, 1523, 1187, 1150, 1195, 956,  1452, 1186, 1563,
#         1312, 1519, 1427, 1042, 1412, 1595, 1323, 1184, 1086, 1554, 1546, 1246,
#         1260, 1479, 1099, 1318, 1368, 1558, 1122, 1384, 1525, 974,  1478, 1118,
#         1588, 1418, 1456, 963,  1078, 1408, 1402, 1444, 1142, 983,  1404, 1250,
#         1464, 1526, 1207, 1304, 1174, 1019, 1151, 1576, 1358, 1375, 1336, 1192,
#         1362, 1102, 1474, 1288, 1296, 1386, 1066, 1056, 970,  1512, 1399, 1416,
#         1188, 1070, 1107, 1063, 1295, 1581, 1266, 1012, 1175, 1422, 1134, 979,
#         1342, 1154, 1156, 1203, 1168, 1415, 1541, 1132, 1256, 1458, 1482, 1035,
#         1196, 1583, 1530, 1310, 1328, 1143, 1100, 1506, 1135, 1451, 1147, 1191,
#         1591, 960,  1110, 1414, 1383, 964,  1335, 1231, 1210, 1535, 1394, 1262,
#         959,  1214, 1350, 1570, 1084, 1495, 1020, 1071, 1568, 1380, 1144, 1487,
#        1222, 1199, 1538, 1160, 1578, 1468]

def get_sids(datadir=DATADIR):
    """
    Get the SALAMI IDS and their audio filenames, returned as lists, for a
    given directory containing all the audio files.
    """
    sids = [int(os.path.splitext(listitem)[0]) \
            for listitem in os.listdir(datadir)]
    paths = [listitem for listitem in os.listdir(datadir)]
    return sids, paths

# We Need to get only the SIDs for which the annotation files are available
ALL_SIDS = get_sids(datadir=DATADIR)[0]
SIDS = []
for sid in ALL_SIDS:
    files = ev.id2filenames(
    sid,
    ann_type="uppercase",
        salamipath=SALAMIDIR,
    )
    if files:
        SIDS.append(sid)
        
def get_data(
        sids,
        datadir=DATADIR,
        salamidir=SALAMIDIR,
        outputdir=OUTPUTDIR,
        prefix=''
    ):
    """
    Give me some data!
    """

    paths = os.listdir(datadir)
    train = {}

    for sid in sids:
        #try:
        print('Sid: ', sid)
        pathmask = [path.startswith(str(sid)) for path in paths]
        path_idx = pathmask.index(True)
        path = paths[path_idx]
        X_path, X_shape, y_path, y_shape = serialize_song(
            sid,
            path,
            datadir=datadir,
            salamidir=salamidir,
            outputdir=outputdir,
            prefix=''
            )
        # Put all the output into dictionaries
        train[str(sid)] = {}
        train[str(sid)]['X_path']  = X_path
        train[str(sid)]['y_path']  = y_path
        train[str(sid)]['X_shape'] = X_shape
        train[str(sid)]['y_shape'] = y_shape
        #except:
        #    print(f'File for sid {sid} not found')
        
    # Save the dicts for later
    np.save(
        os.path.join(outputdir,'datadict'+prefix+'.npy'),
        train
        )

    return train

def get_preparsed_data(datadict_path):
    """
    Give me some preparsed data!
    """
    train = np.load(datadict_path).tolist()
    return train

def serialize_song(
        sid,
        path,
        datadir=DATADIR,
        salamidir=SALAMIDIR,
        outputdir=OUTPUTDIR,
        prefix='data'
        ):
    """
    serialize_data_chunk()
    Serializes a chunk of data on disk, given SIDs and corresponding paths.

    Arguments:
        sids      :  the SIDs (int list)
        paths     :  paths to sids audio files (string list)
        datadir   : where the audio files are stored
        salamidir : i.e. the salami-data-public dir from a cloned SALAMI repo
        outputdir : for serialized data on disk
        prefix    : prefix for serialized data file on disk

    Outputs:
        X_path  : string paths to the serialized files
        X_shape : shape of data serialized in X_path
        y_path  : string paths to the serialized files
        y_shape : shape of data serialized in y_path
    """

    X, y = None, None
    X_path, X_shape, y_path, y_shape = None, None, None, None

    X_shape = [0, 1, N_MEL, N_FRAME_CONTEXT]
    y_shape = [0, 1]

    print("SID: {0},\tfile: {1}".format(sid, path))

    y_path = os.path.abspath(
        os.path.join(outputdir, prefix + str(sid) + '_y')
        )
    X_path = os.path.abspath(
        os.path.join(outputdir, prefix + str(sid) + '_X')
        )

    # Get the annotated segment times (sec)
    times = ev.id2segtimes(
        sid,
        ann_type="uppercase",
        salamipath=salamidir
        )
    times_frames = librosa.time_to_frames(
        times,
        sr=FS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
        )

    # Get signal
    sig, fs = librosa.load(
        os.path.join(datadir, path),
        FS
        )

    # Get feature frames
    sig_feat = librosa.feature.melspectrogram(
        y=sig,
        sr=fs,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MEL,
        fmax=1600
        )
    sig_feat = 20.0*np.log10(np.clip( sig_feat, a_min=1e-12, a_max=None)) # convert to dB
    sig_feat = sig_feat - np.max(sig_feat) # Normalize to 0dB
    sig_feat[sig_feat==-np.inf] = DB_LOW # screen out inf

    # Keep track of the number of frames for this song
    n_frames = sig_feat.shape[1]

    y_shape[0] = n_frames # increment the shape of our final output y data
    X_shape[0] = n_frames # increment the shape of our final output y data

    # Pad the frames, so we can have frames centered at the very start and
    # end of the song.
    sig_feat = np.hstack((
        np.ones((N_MEL, int(N_FRAME_CONTEXT/2))) * DB_LOW,
        sig_feat,
        np.ones((N_MEL, int(N_FRAME_CONTEXT/2))) * DB_LOW
        ))

    # Generate the boundary indicator

    y = np.memmap(
        y_path,
        dtype=DTYPE,
        mode='w+',
        shape=tuple(y_shape)
        )
    y[:] = np.zeros((n_frames,1))[:] # start with zeros
    y[np.minimum(times_frames,n_frames-1),0] = 1.0

    if(DEBUG_PLOT):
        plt.figure(figsize=(10,  3))
        plt.plot(
            y,
            label="Annotations"
            )

    # Smooth y with the gaussian kernel
    y[:,0] = np.convolve( y[:,0], BOUNDARY_KERNEL, 'same')
    y[:,0] = np.minimum(y[:,0],1.0) # nothing above 1

    if(DEBUG_PLOT):
        plt.plot(
            y,
            label="Smoothed"
            )
        plt.xlabel("Frame number")
        plt.ylabel("Segment boundary strength")
        plt.legend()
        # plt.colorbar()
        plt.savefig('./seg.pdf', bbox_inches='tight')
        # plt.show()

    # Generate the training data
    X = np.memmap(
            X_path,
            dtype=DTYPE,
            mode='w+',
            shape=tuple(X_shape)
            )

    for i_frame in range(n_frames):
        X[i_frame,0] = sig_feat[:,i_frame:i_frame+N_FRAME_CONTEXT]

    # debug plot
    if(DEBUG_PLOT):
        plt.figure()
        plt.subplot(211)
        plt.imshow(X[X.shape[0]/2,0])
        plt.colorbar()
        plt.subplot(212)
        plt.plot(y)
        plt.show()

    # Flush our binary data to file
    X.flush()
    y.flush()

    return X_path, X_shape, y_path, y_shape

if __name__ == "__main__":
    P = argparse.ArgumentParser(
        description='Generate some data for the CNN.'
        )
    P.add_argument(
        '-a', '--audiodir',
        help='Directory with salami audio files.',
        required=False,
        default=DATADIR
        )
    P.add_argument(
        '-ds', '--salamidir',
        help='Directory with salami annotation files.',
        required=False,
        default=SALAMIDIR
        )
    P.add_argument(
        '-w', '--workingdir',
        help='Directory for intermediate data and model files.',
        required=False,
        default=OUTPUTDIR
        )
    P.add_argument(
        '-t', '--train',
        help='Number of songs to include in training set.',
        required=False,
        default=1
        )
    P.add_argument(
        '-v', '--val',
        help='Number of songs to include in validation set.',
        required=False,
        default=1
        )
    P.add_argument(
        '-s', '--test',
        help='Number of songs to include in test set.',
        required=False,
        default=1
        )


    ARGS = P.parse_args()

    n_train = int(ARGS.train)
    n_val   = int(ARGS.val)
    n_test  = int(ARGS.test)
    n_total = n_train + n_val + n_test

    n_sids = len(SIDS)
    SID_SUBSET = np.random.choice(SIDS, size=n_total, replace=False)

    train = get_data(
        SID_SUBSET[:n_train],
        datadir=ARGS.audiodir,
        salamidir=ARGS.salamidir,
        outputdir=ARGS.workingdir,
        prefix='train')
    val   = get_data(
        SID_SUBSET[n_train:n_train+n_val],
        datadir=ARGS.audiodir,
        salamidir=ARGS.salamidir,
        outputdir=ARGS.workingdir,
        prefix='val'
        )
    test  = get_data(
        SID_SUBSET[n_train+n_val:],
        datadir=ARGS.audiodir,
        salamidir=ARGS.salamidir,
        outputdir=ARGS.workingdir,
        prefix='test'
        )
    print ('TRAINING SET:')
    print (train)
    print ('VALIDATION SET:')
    print (val)
    print ('TEST SET:')
    print (test)
