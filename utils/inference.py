import numpy as np
import onnxruntime as ort


def inference_batch_imgs(sess: ort.InferenceSession, imgs: np.ndarray) -> np.ndarray:
    """Inference RTMPose model.

    Args:
        sess (ort.InferenceSession): ONNXRuntime session.
        img (np.ndarray): Batch input image in shape.

    Returns:
        outputs (np.ndarray): Output of RTMPose model.
    """
    # build input
    inputs = imgs.transpose(0, 3, 1, 2)

    # build output
    sess_input = {sess.get_inputs()[0].name: inputs}
    sess_output = []
    for out in sess.get_outputs():
        sess_output.append(out.name)

    # run model
    outputs = sess.run(sess_output, sess_input)

    return outputs