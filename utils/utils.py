import onnxruntime as ort
import numpy as np

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


flip_index = {
    'keypoint-27': np.array([0, 2, 1, 4, 3, 6, 5, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
    'keypoint-31': np.array([0, 2, 1, 4, 3, 6, 5, 9, 8, 7, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    'keypoint-49': np.array([0, 2, 1, 4, 3, 6, 5, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
}


def normalize(data_numpy):
    
    data_numpy = np.array(data_numpy)

    data_numpy[0, :, :, :] = data_numpy[0, :, :, :] - \
        data_numpy[0, :, 0, 0].mean(axis=0)
    
    data_numpy[1, :, :, :] = data_numpy[1, :, :, :] - \
        data_numpy[1, :, 0, 0].mean(axis=0)

    return data_numpy
