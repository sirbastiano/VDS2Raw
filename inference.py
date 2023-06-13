import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
# Show the results
from mmcv.transforms import LoadImageFromFile, Compose, Resize
import cv2
import argparse
import os
from pathlib import Path
import json 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# create parser
parser = argparse.ArgumentParser(description='Inference on a single image')
parser.add_argument('--img_path', type=str, help='path to the image')
parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection')
# add device argument
parser.add_argument('--device', type=str, default='cpu', help='device used for inference: cpu or cuda:0')

def draw_bounding_boxes(image, bounding_boxes, scores=None, score_threshold=0.05, backend_args=None, savepath=None):
    """
    Draws bounding boxes on an image.

    Args:
        image (numpy.ndarray): The image to draw bounding boxes on.
        bounding_boxes (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        scores (list, optional): A list of scores for each bounding box. Defaults to None.
        score_threshold (float, optional): The minimum score required to draw a bounding box. Defaults to 0.05.
        backend_args (dict, optional): A dictionary of arguments to pass to the matplotlib backend. Defaults to None.
        savepath (str, optional): The path to save the image with bounding boxes. Defaults to None.

    Returns:
        None
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(1, **backend_args)

    # Display the image
    ax.imshow(image)

    # Add bounding boxes to the image
    for i, bbox in enumerate(bounding_boxes):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axes
        ax.add_patch(rect)
        ax.axis(False)
        # Add the score as text
        if scores is not None:
            score = scores[i]
            ax.text(x_max+5, y_max+5, f'Score: {score:.2f}',
                    color='white', fontsize=8, bbox=dict(facecolor='r', alpha=0.7))
    
    if savepath is not None:
        fig.savefig(savepath)
    # do not show the image
    plt.close(fig)
 
if __name__ == '__main__':
    args = parser.parse_args()
    # define dataloader 
    loader = LoadImageFromFile(to_float32=False, color_type='color', imdecode_backend='tifffile', backend_args=None)
    # Specify the path to model config and checkpoint file
    config_file = 'checkpoint/vfnet_r18_fpn_1x_vessel.py'
    checkpoint_file = 'checkpoint/epoch_239.pth'

    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=args.device)

    # Init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # The dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # Test a single image and show the results
    img_path = args.img_path
    base_name = Path(img_path).stem

    load = loader(results={'img_path': img_path})
    img = load['img']
    result = inference_detector(model, img)

    img = mmcv.imconvert(img, 'bgr', 'rgb')
    print('Inference completed. Saving image...')

    predictions = list(result.pred_instances.all_items())

    keyholder={}
    for item in predictions:
        keyholder[item[0]]=item[1]
        
    scores, boxes, labels = keyholder['scores'], keyholder['bboxes'], keyholder['labels']
    # write the scores and boxes to a json file:
    output_pred_path = Path('output_results') / (base_name + '_predictions.json')
    keyholder_numpy = {k: v.tolist() for k, v in keyholder.items()}
    with open(output_pred_path, 'w') as f:
        json.dump(keyholder_numpy, f)

    scores = list(scores.detach().cpu().numpy())
    boxes = list(boxes.detach().cpu().numpy())

    new_name = base_name + '_pred.png'
    savepath = Path('output_results') / new_name
    # Draw the bounding boxes on the image
    draw_bounding_boxes(img, boxes, scores = scores, backend_args=dict(figsize=(40, 40), dpi=100), savepath=savepath, score_threshold=args.threshold)
    print('Image saved.')