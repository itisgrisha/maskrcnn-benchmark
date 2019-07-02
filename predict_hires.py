from glob import glob
from PIL import Image
from random import choice
import pandas as pd
from tqdm import tqdm
from time import time
import torch

from maskrcnn_benchmark.config import cfg
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

class Detector():
    def __init__(self, cfg_path, weights_path, input_shape=(608, 608)):
        cfg.merge_from_file(cfg_path)
        self._cfg = cfg.clone()
        self.model = build_detection_model(self._cfg)
        self.model.eval()
        self._device = 'cuda'
        self.model.to(self._device)
        self.shape = input_shape

        save_dir = cfg.OUTPUT_DIR
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        load_state_dict(self._model, checkpoint.pop("model"))

        self.transform = self._build_transform()


    def __call__(self, frame):
        return self.infer(frame)

    def infer(self, frame):

        transformed_frame = self._transforms(frame)
        image_list = to_image_list(transformed_frame, self._cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self._device)

        # compute predictions
        with torch.no_grad():
            predictions = self._model(image_list)
        predictions = [o.to('cpu') for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        # height, width = frame.shape[:-1]
        # prediction = prediction.resize((width, height))

        # keypoints = prediction.get_field('keypoints').keypoints.numpy()[..., :2]
        bboxes = prediction.bbox.numpy()
        labels = prediction.get_field('labels').numpy()
        scores = prediction.get_field('scores').numpy()

        return list(zip(bboxes, labels, scores))


    def _build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        if self._cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=self._cfg.INPUT.PIXEL_MEAN, std=self._cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                # T.Resize(self._cfg.INPUT.MIN_SIZE_TEST),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform


def get_crops(img, x_from, y_from, x_to, y_to, size):
    width, height = x_to - x_from, y_to - y_from

    x_count = width // size + 1
    step_x = max(int(0.5 + (width - size) / x_count), 1)
    x_to = x_from + width - size + 1

    y_count = height // size + 1
    step_y = max(int(0.5 + (height - size) / y_count), 1)
    y_to = y_from + height - size + 1
#     print(y_count, y_from ,y_to, step_y)

    crops = []
    shifts = []
    for y in range(y_from, y_to, step_y):
        for x in range(x_from, x_to, step_x):
            crops.append(img.crop((x, y, x+size, y+size)))
            shifts.append((x, y))

    return crops, shifts


path = '/data/ice/images/2018-03-07_1336_right/025520.jpg'
img = Image.open(path)
crops, shifts = get_crops(img, 0, 0, 2448, 1208, 608)

model = Detector('configs/efnet_retina.yaml', '/data/mask_ckpts/rc_608/model_final.pth')

tik = time()

for c in crops:
    predictions = model(c)

tok = time()

print("elapsed {:d}ms for {:d} crops".format(int(1000*(tok-tik))), len(crops))
