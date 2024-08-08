ROOT = '.'
VIDEO_DEMO_PATH = r"../data/video_demo"

FASTTEXT_PRETRAINED_PATH = r'D:/Code/DATN/project/checkpoints/word-embedding/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'
WLASL_JSON_PATH = r'../data/wlasl_2000.json'
WLASL_CLASS_LIST_PATH = r'../data/wlasl_class_list.txt'

PE_ONNX_PATHS = [r"../onnx_model/rtmw-l+.onnx",
                 r"../onnx_model/rtmpose-l.onnx"]

MODEL_CONFIG_PATHS = [r'../checkpoints/ctr-gcn-27-rtmw_onehot_top1=53.93.yaml',
                      r"../checkpoints/ctr-gcn-27-rtml_onehot_top1=43.22.yaml",
                      r"../checkpoints/stgcnpp-31-rtmw_onehot_top1=54.66.yaml",
                      r"../checkpoints/stgcnpp-31-rtmw_nla_top1=56.46.yaml"]

WEIGHT_PATHS = [r"../checkpoints/ctr-gcn-27-rtmw_onehot_top1=53.93.pt",
                r"../checkpoints/ctr-gcn-27-rtml_onehot_top1=43.22.pt",
                r"../checkpoints/stgcnpp-31-rtmw_onehot_top1=54.66.pt",
                r"../checkpoints/stgcnpp-31-rtmw_nla_top1=56.46.pt"]

SAVE_VISUALIZE_DIR = '../visualize_tmp'
