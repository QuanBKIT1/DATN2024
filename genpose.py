# Copyright (c) OpenMMLab. All rights reserved.
from tqdm import tqdm
import pickle
from utils.pre_processing import preprocess
from utils.post_processing import postprocess
from utils.inference import inference_batch_imgs
import cv2
import numpy as np
import onnxruntime as ort
import os
import argparse


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations using inference backend ONNXRuntime for WLASL2000 dataset')

    parser.add_argument('--onnx-path', type=str,
                        help='Path to .onnx file RTMPose inference with ONNXRuntime',
                        default='./onnx_model/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.onnx')

    parser.add_argument('--video-data-path', type=str,
                        help='Path of RGB video dataset',
                        default=r'D:\DATN\project\data\preprocessed_data\rgb\WLASL2000\train')

    parser.add_argument('--device', type=str,
                        help='The device to run the model, default is cpu',
                        default='cpu')

    parser.add_argument('--save-path', type=str,
                        help='The path to save output results (.pkl format for each video)',
                        default='./keypoints_rtmpose_wholebody/')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load config from parser
    onnx_file = args.onnx_path
    device = args.device
    result_path = args.save_path
    data_dir = args.video_data_path
    print("Whole body pose estimation using RTMPose for WLASL2000 dataset")
    sess = build_session(onnx_file, device)
    h, w = sess.get_inputs()[0].shape[2:]
    model_input_size = (w, h)
    print(
        f"Load model from {onnx_file} with input size {w}x{h}, run model on {sess.get_providers()}")

    # Get data path of dataset
    paths = []
    for root, _, fnames in os.walk(data_dir):
        for fname in fnames:
            path1 = os.path.join(root, fname)
            paths.append(path1)
    paths.sort()

    print(
        f"Start generate whole body keypoints from {len(paths)} videos in {data_dir} folder")

    os.makedirs(result_path, exist_ok=True)

    for path in tqdm(paths):
        cap = cv2.VideoCapture(path)
        video_id = os.path.basename(path).split('.')[0]
        stack_frame = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            resized_img, center, scale = preprocess(frame, model_input_size)
            stack_frame.append(resized_img)
        stack_frame = np.array(stack_frame, dtype=np.float32)

        # inference
        outputs = inference_batch_imgs(sess, stack_frame)
        # postprocessing
        keypoints, scores = postprocess(
            outputs, model_input_size, center, scale)
        results = np.concatenate((keypoints, scores[:, :, None]), axis=2)
        save_file = os.path.join(result_path, video_id + '.pkl')
        with open(save_file, mode='wb') as f:
            pickle.dump(results, f)

    print("Done")


if __name__ == '__main__':
    main()
