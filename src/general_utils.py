import cv2
import numpy as np

def crop(image : np.ndarray, crop : np.ndarray ) -> np.ndarray :
    return image[crop[1]:crop[1]+crop[3], crop[0]:crop[0] + crop[2]]

def preprocess(image : np.ndarray, **cfg) -> np.ndarray :
    result_image = image.copy()
    result_image = result_image.astype(np.float32)

    rows = cfg.get('rows', result_image.shape[0])
    cols = cfg.get('cols', result_image.shape[1])
    result_image = cv2.resize(result_image, (cols, rows), cv2.INTER_LINEAR)
    mean = result_image.mean()
    std  = result_image.std()
    result_image = (result_image - mean) / std
    
    return result_image

def draw_rectangle(image : np.ndarray , box : np.ndarray, text : str , **cfg) -> np.ndarray :
    thickness    = cfg.get('thickness', 1)
    color        = cfg.get('color', (0,255,0))
    font         = cfg.get('font', cv2.FONT_HERSHEY_SIMPLEX)
    font_scale   = cfg.get('font', 0.8)
    result_image = image.copy()

    cv2.rectangle(
        result_image, 
        (box[0], box[1]), 
        (box[0] + box[2], box[1] + box[3]),
        color=color,
        thickness=thickness,
    )

    cv2.putText(
        result_image, 
        text, 
        (box[0], box[1]), 
        font, font_scale, color, thickness)

    return result_image