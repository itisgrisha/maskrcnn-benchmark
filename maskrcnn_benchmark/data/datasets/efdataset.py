from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box.boxlist_ops import remove_small_boxes
import maskrcnn_benchmark.transforms as T
from pathlib import Path
from collections import defaultdict
import imagesize
from PIL import Image

class RandomCrop:
    def __init__(self, w, h, s, min_area, min_visibility):
        self.w = w
        self.h = h
        self.s = s
        self.min_area = min_area
        self.min_visibility = min_visibility

    @staticmethod
    def _iom(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(min(boxBArea, boxAArea))
        return iou


    def __call__(self, example):
        image = example['image']
        boxes = example['boxes']
        labels = example['labels']
        im_w, im_h = image.size

        try:
            box = np.random.choice(boxes)
            from copy import deepcopy
            orig = deepcopy(box)

            if (box.right - box.left) > self.w:
                x_from = max(0, box.left-self.s)
                x_to = x_from + 1
            else:

                x_from = max(0, box.right - self.w)
                x_to = min(im_w - self.w, box.left) + 1
            x = np.random.randint(int(x_from), int(x_to))

            if (box.bottom - box.top) > self.h:
                y_from = max(0, box.top-self.s)
                y_to = y_from + 1
            else:
                y_from = max(0, box.bottom - self.h)
                y_to = min(im_h - self.h, box.top)  + 1
            y = np.random.randint(int(y_from), int(y_to))
        except:
            print('bbox:', box.left, box.top, box.right, box.bottom)
            print('im_h: {}, im_w: {}'.format(im_h, im_w))
            print('x:')
            print('x_from; max(0,',box.right - self.w)
            print('x_to: min',im_w - self.w, box.left)
            print(x_from, x_to)
            print('y:')
            print('y_from: max(0,', box.bottom - self.h)
            print('y_to: min', im_h - self.h, box.top   )
            print(y_from, y_to)
            sys.stdout.flush()
        crop = [x, y, x+self.w, y+self.h]

        image = image.crop(crop)

        new_boxes = []
        new_labels = []
        for box, label in zip(boxes, labels):
            iom = RandomCrop._iom(crop, [box.left, box.top, box.right, box.bottom])
            if iom < self.min_visibility:
                continue
            new_boxes.append(
                BoundingBox(
                    max(0, box.left - x),
                    max(0, box.top - y),
                    min(self.w - 1, box.right - x),
                    min(self.h - 1, box.bottom - y),
                    self.w,
                    self.h,
                    label
                )
            )
            new_labels.append(label)
        if len(new_boxes) == 0:
            print(new_boxes)
            box = orig
            print('bbox:', box.left, box.top, box.right, box.bottom)
            print('im_h: {}, im_w: {}'.format(im_h, im_w))
            print('x:')
            print('x_from; max(0,',box.right - self.w)
            print('x_to: min',im_w - self.w, box.left)
            print(x_from, x_to)
            print('y:')
            print('y_from: max(0,', box.bottom - self.h)
            print('y_to: min', im_h - self.h, box.top   )
            print(y_from, y_to)
            sys.stdout.flush()

        return {'image': image, 'boxes': new_boxes, 'labels': new_labels}


class IceDataset(object):
    def __init__(self, labels_path, images_path, transforms=None, train=True):
        self.class_map = {c: i for i, c in enumerate(['2.1', '2.4', '3.1', '3.24', '3.27', '4.1', '4.2', '5.19', '5.20', '8.22'], 1)}
        self.class_map = defaultdict(lambda: 0, self.class_map)
        self.labels_path = Path(labels_path)
        self.images_path = Path(images_path)
        self.tasks = []
        self.train = train
        for label in self.labels_path.glob("**/*.tsv"):
            df = pd.read_csv(label, sep='\t')
            if df.shape[0] > 0:
                task = '/'.join((label.parts[-2], label.stem))
                self.tasks.append(task)
        self.transforms = transforms

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        impath = self.images_path/(task+'.jpg')
        lpath = self.labels_path/(task+'.tsv')
        image = Image.open(impath)
        df = pd.read_csv(lpath, sep='\t')
        labels = []
        boxlist = []
        for i, row in df.iterrows():
            labels.append(self.class_map[row['class']])
            boxlist.append([row.xtl, row.ytl, row.xbr, row.ybr])

        if self.train:
            RandomCrop...

        labels = torch.tensor(labels)
        boxlist = BoxList(boxlist, image.size, mode="xyxy")
        boxlist.add_field("labels", labels)
        remove_small_boxes(boxlist, 10)
        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)

        return image, boxlist, idx

    def get_img_info(self, idx):
        return {"height": 1224, "width": 1224}
