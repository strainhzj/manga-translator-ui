from typing import Tuple, List
import numpy as np
import cv2
import math
import logging

from tqdm import tqdm
from shapely.geometry import Polygon
# from sklearn.mixture import BayesianGaussianMixture
# from functools import reduce
# from collections import defaultdict
# from scipy.optimize import linear_sum_assignment

from ..utils import Quadrilateral, image_resize, imwrite_unicode

COLOR_RANGE_SIGMA = 1.5 # how many stddev away is considered the same color

def save_rgb(fn, img, logger=None):
    if logger is None:
        logger = logging.getLogger('manga_translator')
    if len(img.shape) == 3 and img.shape[2] == 3:
        imwrite_unicode(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), logger)
    else:
        imwrite_unicode(fn, img, logger)

def area_overlap(x1, y1, w1, h1, x2, y2, w2, h2):  # returns None if rectangles don't intersect
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0
    
def extend_rect(x, y, w, h, max_x, max_y, extend_size):
    x1 = max(x - extend_size, 0)
    y1 = max(y - extend_size, 0)
    w1 = min(w + extend_size * 2, max_x - x1 - 1)
    h1 = min(h + extend_size * 2, max_y - y1 - 1)
    return x1, y1, w1, h1

def complete_mask_fill(text_lines: List[Tuple[int, int, int, int]]):
    for (x, y, w, h) in text_lines:
        final_mask = cv2.rectangle(final_mask, (x, y), (x + w, y + h), (255), -1)
    return final_mask

from pydensecrf.utils import compute_unary, unary_from_softmax
import pydensecrf.densecrf as dcrf

# 兼容不同版本的 pydensecrf
DIAG_KERNEL = getattr(dcrf, 'DIAG_KERNEL', 0)
NO_NORMALIZATION = getattr(dcrf, 'NO_NORMALIZATION', 0)

def refine_mask(rgbimg, rawmask):
    # Optimization: Early exit for empty or trivial masks
    if rawmask is None or rawmask.size == 0:
        return rawmask
    if np.max(rawmask) == 0: # Check if mask is completely black
        return rawmask
        
    # Optimization: Skip expensive CRF for very small regions (e.g. < 100 pixels)
    # The overhead of creating DenseCRF2D outweighs the benefit for tiny spots
    if rawmask.size < 100:
        return rawmask

    if len(rawmask.shape) == 2:
        rawmask = rawmask[:, :, None]
    
    # 复用数组，减少内存分配
    # Optimization: Assuming rawmask is 0 or 255. 
    # Pre-calculating probabilities without division if possible, but modern CPUs handle this fast enough.
    # We stick to standard float conversion but can optimize if needed.
    mask_softmax = np.empty((rawmask.shape[0], rawmask.shape[1], 2), dtype=np.float32)
    # Norm to 0-1
    float_mask = rawmask[:, :, 0].astype(np.float32) / 255.0
    mask_softmax[:, :, 0] = 1.0 - float_mask
    mask_softmax[:, :, 1] = float_mask
    
    n_classes = 2
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=3, kernel=DIAG_KERNEL,
                            normalization=NO_NORMALIZATION)

    d.addPairwiseBilateral(sxy=23, srgb=7, rgbim=rgbimg,
                        compat=20,
                        kernel=DIAG_KERNEL,
                        normalization=NO_NORMALIZATION)
    
    # Reverted to 5 steps as per user request to maintain quality.
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
    
    # 直接转换，避免额外复制
    return (res * 255).astype(np.uint8)

def complete_mask(img: np.ndarray, mask: np.ndarray, textlines: List[Quadrilateral], keep_threshold = 1e-2, dilation_offset = 0,kernel_size=3):
    bboxes = [txtln.aabb.xywh for txtln in textlines]
    polys = [Polygon(txtln.pts) for txtln in textlines]
    for (x, y, w, h) in bboxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0), 1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    M = len(textlines)
    
    # --- NEW DIAGNOSTIC LINES ---
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"--- MASK_REFINEMENT_DEBUG: Entering complete_mask ---")
    logger.debug(f"--- MASK_REFINEMENT_DEBUG: Number of textlines (M) = {M} ---")
    logger.debug(f"--- MASK_REFINEMENT_DEBUG: Number of connected components (num_labels) = {num_labels} ---")
    # --- END DIAGNOSTIC ---

    textline_ccs = [np.zeros_like(mask) for _ in range(M)]
    iinfo = np.iinfo(labels.dtype)
    textline_rects = np.full(shape = (M, 4), fill_value = [iinfo.max, iinfo.max, iinfo.min, iinfo.min], dtype = labels.dtype)
    ratio_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
    dist_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
    valid = False
    for label in range(1, num_labels):
        # skip area too small
        if stats[label, cv2.CC_STAT_AREA] <= 9:
            continue

        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w1 = stats[label, cv2.CC_STAT_WIDTH]
        h1 = stats[label, cv2.CC_STAT_HEIGHT]
        area1 = stats[label, cv2.CC_STAT_AREA]
        cc_pts = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])
        cc_poly = Polygon(cc_pts)

        for tl_idx in range(M):
            area2 = polys[tl_idx].area
            try:
                # 尝试计算交集，如果几何体无效则使用 buffer(0) 修复
                overlapping_area = polys[tl_idx].intersection(cc_poly).area
            except Exception as e:
                # 几何体无效时，尝试使用 buffer(0) 修复
                try:
                    fixed_poly = polys[tl_idx].buffer(0)
                    fixed_cc_poly = cc_poly.buffer(0)
                    overlapping_area = fixed_poly.intersection(fixed_cc_poly).area
                except Exception:
                    # 如果仍然失败，设置为 0
                    overlapping_area = 0
            
            ratio_mat[label, tl_idx] = overlapping_area / min(area1, area2)
            
            try:
                dist_mat[label, tl_idx] = polys[tl_idx].distance(cc_poly.centroid)
            except Exception:
                # 如果距离计算失败，使用一个大值
                dist_mat[label, tl_idx] = float('inf')
            # print(textlines[tl_idx].pts, cc_pts, '->', overlapping_area, min(area1, area2), '=', overlapping_area / min(area1, area2), '|', polys[tl_idx].distance(cc_poly))

        avg = np.argmax(ratio_mat[label])
        max_overlap = ratio_mat[label, avg]

        # If the best overlap for this component is essentially zero, discard it.
        # This handles components from a raw_mask for regions that have been deleted.
        if max_overlap < 0.1:
            continue
            
        # print(avg, 'overlap:', ratio_mat[label, avg], '<', keep_threshold)
        area2 = polys[avg].area
        if area1 >= area2:
            continue
        if ratio_mat[label, avg] <= keep_threshold:
            avg = np.argmin(dist_mat[label])
            area2 = polys[avg].area
            unit = max(min([textlines[avg].font_size, w1, h1]), 10)
            # print("unit", unit, textlines[avg].font_size, w1, h1)
            # if area1 < 0.4 * w1 * h1:
            #     # ccs is probably angled
            #     unit /= 2
            # if avg == 0:
            # print('no intersect', area1, '>=', area2, dist_mat[label, avg], '>=', 0.5 * unit)
            if dist_mat[label, avg] >= 0.5 * unit:
                # print(dist_mat[label])
                # print('CONTINUE')
                continue

        textline_ccs[avg][y1:y1+h1, x1:x1+w1][labels[y1:y1+h1, x1:x1+w1] == label] = 255
        # if avg == 0:
        # print(avg)
        # cv2.imshow('ccs', image_resize(textline_ccs[avg], height = 800))
        # cv2.waitKey(0)
        textline_rects[avg, 0] = min(textline_rects[avg, 0], x1)
        textline_rects[avg, 1] = min(textline_rects[avg, 1], y1)
        textline_rects[avg, 2] = max(textline_rects[avg, 2], x1 + w1)
        textline_rects[avg, 3] = max(textline_rects[avg, 3], y1 + h1)
        valid = True

    if not valid:
        return None
    
    # tblr to xywh
    textline_rects[:, 2] -= textline_rects[:, 0]
    textline_rects[:, 3] -= textline_rects[:, 1]
    
    final_mask = np.zeros_like(mask)
    img = cv2.bilateralFilter(img, 17, 80, 80)
    for i, cc in enumerate(tqdm(textline_ccs, '[mask]')):
        x1, y1, w1, h1 = textline_rects[i]
        text_size = min(w1, h1, textlines[i].font_size)
        x1, y1, w1, h1 = extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], int(text_size * 0.1))
        # TODO: Need to think of better way to determine dilate_size.
        dilate_size = max((int((text_size + dilation_offset) * 0.3) // 2) * 2 + 1, 3)
        # print(textlines[i].font_size, min(w1, h1), dilate_size)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        cc_region = np.ascontiguousarray(cc[y1: y1 + h1, x1: x1 + w1])
        if cc_region.size == 0:
            continue
        # cv2.imshow('cc before', image_resize(cc_region, height = 800))
        img_region = np.ascontiguousarray(img[y1: y1 + h1, x1: x1 + w1])
        # cv2.imshow('img', image_resize(img_region, height = 800))
        cc_region = refine_mask(img_region, cc_region)
        # cv2.imshow('cc after', image_resize(cc_region, height = 800))
        # cv2.waitKey(0)
        cc[y1: y1 + h1, x1: x1 + w1] = cc_region
        # cc = cv2.dilate(cc, kern)
        x2, y2, w2, h2 = extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], -(-dilate_size // 2))
        cc[y2:y2+h2, x2:x2+w2] = cv2.dilate(cc[y2:y2+h2, x2:x2+w2], kern)
        final_mask[y2:y2+h2, x2:x2+w2] = cv2.bitwise_or(final_mask[y2:y2+h2, x2:x2+w2], cc[y2:y2+h2, x2:x2+w2])
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # for (x, y, w, h) in text_lines:
    #     final_mask = cv2.rectangle(final_mask, (x, y), (x + w, y + h), (255), -1)
    return cv2.dilate(final_mask, kern)

def unsharp(image):
    gaussian_3 = cv2.GaussianBlur(image, (3, 3), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)