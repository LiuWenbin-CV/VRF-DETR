import warnings

# from ultralytics.nn.extra_modules.mamba.test_mamba_module import model

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 调试看Tensor：.clone().detach().cpu().numpy()

if __name__ == '__main__':
    model = RTDETR('weights/VRF-DETR.pt') # select your model.pt path
    model.predict(source='example/example_result.jpg',
                  conf=0.4,
                  project='runs/detect',
                  name='VisDrone',
                  save=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  show_conf=True, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  save_conf=True,
                  save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )