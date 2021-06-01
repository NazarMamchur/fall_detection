import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from math import ceil

from predict import get_keypoints

root_dir = 'path/to/UR Fall Dataset'
root_dir_mul = 'path/to/Multicam Dataset'

falls_dir = os.path.join(root_dir, 'falls')
no_falls_dir = os.path.join(root_dir, 'no_falls')
falls_labels_path = os.path.join(root_dir, 'urfall-cam0-falls.csv')
no_falls_labels_path = os.path.join(root_dir, 'urfall-cam0-adls.csv')

labels = {
    'chute01': [1080, 1108],
    'chute02': [375, 399],
    'chute03': [591, 625],
    'chute04': [[288, 314], [601, 638]],
    'chute05': [311, 336],
    'chute06': [583, 629],
    'chute07': [476, 507],
    'chute08': [271, 298],
    'chute09': [628, 651],
    'chute10': [512, 530],
    'chute11': [464, 489],
    'chute12': [605, 653],
    'chute13': [823, 863],
    'chute14': [989, 1023],
    'chute15': [755, 787],
    'chute16': [891, 940],
    'chute17': [730, 770],
    'chute18': [571, 601],
    'chute19': [499, 600],
    'chute20': [545, 672],
    'chute21': [864, 901],
    'chute22': [767, 808],
    'chute23': [[1520, 1595], [3574, 3614]]
}


def parse_model_results(rootdir, labels_path):
    labels = pd.read_csv(labels_path, header=None).iloc[:, :3]
    labels.columns = ['folder', 'file', 'label']

    res_df_cols = ['folder', 'file', 'label'] + [f'lm_{i}' for i in range(34)]
    res_df = pd.DataFrame(columns=res_df_cols)

    for index, row in labels.iterrows():
        folder_name = '{}-cam0-rgb'.format(row['folder'])
        file_name = folder_name + '-{0:0>3}.png'.format(row['file'])
        file_path = os.path.join(rootdir, folder_name, file_name)
        image_src = cv2.imread(file_path)
        _, kps_ = get_keypoints(image_src)
        result_flat = kps_.flatten()

        new_row = list(row) + list(result_flat)
        res_df.loc[len(res_df)] = new_row

    return res_df


def parse_model_results_multicamera(rootdir, labels_list):
    res_df_cols = ['folder', 'file', 'label'] + [f'lm_{i}' for i in range(34)]
    res_df = pd.DataFrame(columns=res_df_cols)
    for k, v in labels_list.items():
        v = [v] if not isinstance(v[0], list) else v
        path = os.path.join(rootdir, k)
        print(path)
        for filename in os.listdir(path):
            if filename.endswith(".avi"):
                file_path = os.path.join(path, filename)
                print(file_path)
                vidcap = cv2.VideoCapture(file_path)
                success,image = vidcap.read()
                count = 0
                while success:
                    _, kps_ = get_keypoints(image)
                    result_flat = kps_.flatten()
                    label = 1
                    for l in v:
                        if l[0] <= count <= l[1]:
                            lable = 0
                            if count in l:
                                print(count)
                    new_row = [k + '-' + filename, count, label] + list(result_flat)
                    res_df.loc[len(res_df)] = new_row
                    count += 1
                    success,image = vidcap.read()
    return res_df


def get_data_list(df, window):
    falls_list = []
    no_falls_list = []
    stride = int(0.2 * window)
    fall_treshold = ceil(0.05 * window)
    for fol in df['folder'].unique():
        df_inner = df[df['folder'] == fol]
        i, j = 0, window
        while j < len(df_inner):
            df_window = df_inner.iloc[i:j]
            fall_frames_count = df_window[df_window['label'] == 0]['label'].count()
            if fall_frames_count >= fall_treshold:
                falls_list.append(df_window.iloc[:, 3:])
            else:
                no_falls_list.append(df_window.iloc[:, 3:])
            i += stride
            j += stride
    return np.array(falls_list), np.array(no_falls_list)


def get_balanced_set(df, window):
    falls, no_falls = get_data_list(df, window)
    idx = np.random.randint(no_falls.shape[0], size=falls.shape[0])
    no_falls_cut = no_falls[idx, :]

    X = np.vstack((falls, no_falls_cut))
    y = to_categorical(np.vstack((np.ones((falls.shape[0], 1), dtype=int),
                                  np.zeros((no_falls_cut.shape[0], 1), dtype=int))))

    return X, y


if __name__ == '__main__':
    falls_res = parse_model_results(falls_dir, falls_labels_path)
    no_falls_res = parse_model_results(no_falls_dir, no_falls_labels_path)

    d = parse_model_results_multicamera(root_dir_mul, labels)
    dn = d.copy()
    for k, v in labels.items():
        v = [v] if not isinstance(v[0], list) else v
        for l in v:
            dn.loc[(dn['folder'] == k + '-cam1.avi') & (l[0] <= dn['file']) & (dn['file'] <= l[1]), 'label'] = 0
    dn[dn['label'] == 0].count()

    ur_data = falls_res.append(no_falls_res, ignore_index=True)

    X_ur, y_ur = get_balanced_set(ur_data, 50)
    X_multi, y_multi = get_balanced_set(dn, 50)
    print(X_ur.shape, y_ur.shape)
    print(X_multi.shape, y_multi.shape)
    X = np.vstack((X_ur, X_multi))
    y = np.vstack((y_ur, y_multi))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    print(n_timesteps, n_features, n_outputs)

    model = Sequential()
    model.add(LSTM(196, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(196, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=16, verbose=True)

    model.save('models/fall_detector.h5')

