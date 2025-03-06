import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('weights/VRF-DETR.pt')
    model.val(data='dataset/data_VisDrone.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=1,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='VisDrone_C2f-MSCF-GConv-AIFI-MSCF',
              )

