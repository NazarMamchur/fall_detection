import tensorflow as tf
import keras
import numpy as np
import cv2

pose_model_path = 'models/posenet.tflite'
fall_model_path = 'models/fall_detector.h5'

model = keras.models.load_model(fall_model_path)
interpreter = tf.lite.Interpreter(model_path=pose_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


def get_keypoints(image):
    img = cv2.resize(image, (width, height))
    img_input = np.expand_dims(img.copy(), axis=0)
    img_input = (np.float32(img_input) - 127.5) / 127.5

    # Runs the computation
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    # Extract output data from the interpreter
    output_data = interpreter.get_tensor(output_details[0]['index'])
    offset_data = interpreter.get_tensor(output_details[1]['index'])
    heatmap_data = np.squeeze(output_data)
    offset_data = np.squeeze(offset_data)

    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(heatmap_data.shape[-1]):
        joint_heatmap = heatmap_data[..., i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)
        pose_kps[i, 0] = int(remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i])
        pose_kps[i, 1] = int(remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i + joint_num])
        pose_kps[i, 2] = 1

    return img, pose_kps


def draw(show_img, kps, ratio=None):
    parts_to_compare = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14),
                        (13, 15), (14, 16)]

    for pair in parts_to_compare:
        cv2.line(show_img, (kps[pair[0]][1], kps[pair[0]][0]), (kps[pair[1]][1], kps[pair[1]][0]),
                 color=(0, 0, 255), lineType=cv2.LINE_AA, thickness=1)

    for i in range(5, kps.shape[0]):
        if kps[i, 2]:
            if isinstance(ratio, tuple):
                cv2.circle(show_img, (int(round(kps[i, 1]*ratio[1])),
                                      int(round(kps[i, 0]*ratio[0]))), 2, (0, 255, 255), round(int(1*ratio[1])))
                continue
            cv2.circle(show_img, (kps[i, 1], kps[i, 0]), 2, (0, 255, 255), -1)
    return show_img


def get_fall_prediction(frame_stack):
    y_pred = model.predict(frame_stack)
    return y_pred

