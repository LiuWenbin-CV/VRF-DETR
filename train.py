import warnings

from win32comext.adsi.demos.scp import verbose

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-C2f-MSCF-GConv-AIFI-MSCF.yaml')
    model.load('weights/VRF-DETR.pt') # loading pretrain weights
    model.train(data='dataset/data_VisDrone.yaml',
                cache=True,
                imgsz=640,
                epochs=200,
                batch=8,
                workers=12,
                device='0',
                # freeze=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,18,19,20,22,23,25,26],
                # resume='runs/train/multiscale/weights/last.pt', # last.pt path
                project='runs/train',
                name='VisDrone_C2f-MSCF-GConv-AIFI-MSCF',
                verbose=False
                )