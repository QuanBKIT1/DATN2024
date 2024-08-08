import onnxruntime as ort
import numpy as np
import cv2

# Define keypoint selected
joints = {
    27: np.concatenate(([0, 1, 2, 5, 6, 7, 8],    # upper body keypoints
                        [91, 95, 96, 99, 100, 103, 104, 107, 108, 111], [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),
    31: np.concatenate(([0, 1, 2, 5, 6, 7, 8],    # upper body keypoints
                        [71, 74, 77, 80],  # mouth keypoints
                        [91, 95, 96, 99, 100, 103, 104, 107, 108, 111], [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),
    49: np.concatenate(([0, 1, 2, 5, 6, 7, 8],     # upper body keypoints
                        np.arange(91, 133)   # hand keypoints
                        ), axis=0),


}

# Define edge connections
edges = {}

edges[27] = [(1, 2), (1, 3), (1, 6), (1, 7), (6, 8), (7, 9),  # upper body connection
             (8, 92), (92, 96), (92, 97), (92, 101), (92, 105),
             (92, 109), (97, 100), (101, 104), (105,
                                                108), (109, 112),  # Right hand connection
             (9, 113), (113, 117), (113, 118), (113, 122), (113, 126),
             (113, 130), (118, 121), (122, 125), (126,
                                                  129), (130, 133)  # Left hand connection
             ]


edges[31] = [(1, 2), (1, 3), (1, 6), (1, 7), (6, 8), (7, 9),  # upper body connection
             (72, 75), (72, 81), (75, 78), (78, 81),  # Mouth connection
             (8, 92), (92, 96), (92, 97), (92, 101), (92, 105),
             (92, 109), (97, 100), (101, 104), (105,
                                                108), (109, 112),  # Right hand connection
             (9, 113), (113, 117), (113, 118), (113, 122), (113, 126),
             (113, 130), (118, 121), (122, 125), (126,
                                                  129), (130, 133)  # Left hand connection
             ]

edges[49] = [(1, 2), (1, 3), (1, 6), (1, 7), (6, 8), (7, 9),  # upper body connection
             (8, 92), (92, 93), (92, 97), (92,
                                           101), (92, 105), (92, 109),
             (93, 94), (94, 95), (95, 96), (97, 98), (98, 99), (99, 100),
             (101, 102), (102, 103), (103, 104), (105,
                                                  106), (106, 107), (107, 108),
             # Right hand connection
             (109, 110), (110, 111), (111, 112),
             (9, 113), (113, 114), (113, 118), (113,
                                                122), (113, 126), (113, 130),
             (114, 115), (115, 116), (116, 117), (118,
                                                  119), (119, 120), (120, 121),
             (122, 123), (123, 124), (124, 125), (126,
                                                  127), (127, 128), (128, 129),
             (130, 131), (131, 132), (132, 133)
             # Left hand connection
             ]


def prepare_data(results, num_keypoint=27):
    """
    Result shape: (T,133,3)
    Keypoint selection: 27-31-49

    # Must in size (256,256)

    Return data preprocessing
    """

    data = np.array(results[:, joints[num_keypoint], :])

    # Extract frame with not in frame
    new_data = []
    for frame in data:
        if np.all(frame[:, :2] <= 300) and np.all(frame[:, :2] >= 0):
            new_data.append(frame)
    data = np.array(new_data)
    T = data.shape[0]
    if T < 150:
        num_repeats = 150 // T + 1
        data = np.tile(data, (num_repeats, 1, 1))[:150]
    else:
        data = data[:150]

    # Shape x: (150,num_keypoint,3) -> Reshape to (1,3,150,num_keypoint,1)
    data = data.transpose((2, 0, 1))
    data = np.expand_dims(data, axis=-1)
    #   Normalize data
    data = normalize(data)
    return data


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def build_session(onnx_file: str, device: str = 'cpu') -> ort.InferenceSession:
    """Build onnxruntime session.

    Args:
        onnx_file (str): ONNX file path.
        device (str): Device type for inference.

    Returns:
        sess (ort.InferenceSession): ONNXRuntime session.
    """
    providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
    sess = ort.InferenceSession(path_or_bytes=onnx_file, providers=providers)

    return sess


def normalize(data_numpy):

    data_numpy = np.array(data_numpy)

    data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - \
        data_numpy[0, :, 0, 0].mean(axis=0)

    data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - \
        data_numpy[1, :, 0, 0].mean(axis=0)

    return data_numpy


def draw_skeleton(img, result, num_keypoint, show_limbs=True):
    shown_keypoints = joints[num_keypoint]
    l_pair = np.array(edges[num_keypoint]) - 1
    part_line = {}
    kp_preds = result[:, :2]
    kp_scores = result[:, 2]

    # Draw keypoints
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= 0.1 or n not in shown_keypoints:
            continue
        cor_x, cor_y = int(
            round(kp_preds[n, 0])), int(round(kp_preds[n, 1]))
        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 2, (0, 0, 255), -1)

    if show_limbs:
        # Draw limbs
        for start_p, end_p in l_pair:
            if start_p in part_line and end_p in part_line:
                start_p = part_line[start_p]
                end_p = part_line[end_p]
                cv2.line(img, start_p, end_p, (0, 255, 0), 2)
    return img


