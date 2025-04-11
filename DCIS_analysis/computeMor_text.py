# coding=utf-8
"""
This script is adapted from the sc_MTOP project, specifically the WSIGraph.py script available at:
https://github.com/fuscc-deep-path/sc_MTOP/blob/master/WSIGraph.py
"""

from collections import defaultdict
from tracemalloc import start
from tqdm import tqdm
from skimage.measure import regionprops
from scipy import stats
from openslide import OpenSlide
import skimage.feature as skfeat
import cv2
import numpy as np
import json
import time
import os
import multiprocessing as mp
import pandas as pd
import pickle
import argparse

try:
    mp.set_start_method('spawn')
except:
    pass


def getRegionPropFromContour(contour, bbox, extention=2):
    (left, top), (right, bottom) = bbox
    height, width = bottom - top, right - left
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    image = np.zeros((height + extention * 2,
                      width + extention * 2),
                     dtype=np.uint8)
    contour[:, 0] = contour[:, 0] - left + extention 
    contour[:, 1] = contour[:, 1] - top + extention
    cv2.drawContours(image, [contour], 0, 1, -1)
    # TODO: check contour coords
    regionProp = regionprops(image)[0]
    return regionProp


def getCurvature(contour, n_size=5):
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

 
    samplekeep = np.zeros((len(contour)), dtype=np.bool_)
    samplekeep[0] = True
    flag = 0
    for i in range(1, len(contour)):
        if np.abs(contour[i] - contour[flag]).sum() > 2:
            flag = i
            samplekeep[flag] = True

    contour = contour[samplekeep]
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    
    ds = np.sqrt(np.sum(dxy ** 2, axis=1, keepdims=True))
    if np.any(ds == 0):
        return (None,) * 6
    ddxy = dxy / ds
    ds = (ds + np.roll(ds, shift=1)) / 2
    Cxy = np.diff(np.concatenate([ddxy, ddxy[0:1]], axis=0), axis=0) / ds
    Cxy = (Cxy + np.roll(Cxy, shift=1, axis=0)) / 2
    k = (ddxy[:, 1] * Cxy[:, 0] - ddxy[:, 0] * Cxy[:, 1]) / ((ddxy ** 2).sum(axis=1) ** (3 / 2))

    curvMean = k.mean()
    curvMin = k.min()
    curvMax = k.max()
    curvStd = k.std()

    n_protrusion = 0
    n_indentation = 0
    if n_size > len(k):
        n_size = len(k) // 2
    k_circle = np.concatenate([k[-n_size:], k, k[:n_size]], axis=0)
    for i in range(n_size, len(k_circle) - n_size):
        neighbor = k_circle[i - 5:i + 5]
        if k_circle[i] > 0:
            if k_circle[i] == neighbor.max():
                n_protrusion += 1
        elif k_circle[i] < 0:
            if k_circle[i] == neighbor.min():
                n_indentation += 1
    n_protrusion /= len(contour)
    n_indentation /= len(contour)

    return curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation


def SingleMorphFeatures(args):
    cellIds, contours, bboxes = args
    featuresDict = defaultdict(list)
    
    for cellId,contour, bbox in zip(cellIds,contours, bboxes):
        featuresDict['cellId'] += [cellId]
        regionProps = getRegionPropFromContour(contour, bbox)
        featuresDict['Area'] += [regionProps.area]
        featuresDict['AreaBbox'] += [regionProps.bbox_area]
        # featuresDict['AreaConvex'] += [regionProps.convex_area]
        # featuresDict['EquialentDiameter'] += [regionProps.equivalent_diameter]
        featuresDict['CellEccentricities'] += [regionProps.eccentricity]
        if regionProps.perimeter ==0: 
            featuresDict['Circularity'] +=[None]
        else:
            featuresDict['Circularity'] += [(4 * np.pi * regionProps.area) / (regionProps.perimeter ** 2)]
        if regionProps.minor_axis_length==0:
            featuresDict['Elongation'] += [None]
        else:
            featuresDict['Elongation'] += [regionProps.major_axis_length / regionProps.minor_axis_length]
        featuresDict['Extent'] += [regionProps.extent]
        # featuresDict['FeretDiameterMax'] += [regionProps.feret_diameter_max]
        featuresDict['MajorAxisLength'] += [regionProps.major_axis_length]
        featuresDict['MinorAxisLength'] += [regionProps.minor_axis_length]
        # featuresDict['Orientation'] += [regionProps.orientation]
        featuresDict['Perimeter'] += [regionProps.perimeter]
        featuresDict['Solidity'] += [regionProps.solidity]

        curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation = getCurvature(contour)
        featuresDict['CurvMean'] += [curvMean]
        featuresDict['CurvStd'] += [curvStd]
        featuresDict['CurvMax'] += [curvMax]
        featuresDict['CurvMin'] += [curvMin]
        # featuresDict['NProtrusion'] += [n_protrusion]
        # featuresDict['NIndentation'] += [n_indentation]

    return featuresDict


def getMorphFeatures(cellIds, contours, bboxes, desc, process_n=1):
    
    if process_n == 1:
        return SingleMorphFeatures([cellIds, contours, bboxes])
    else:
        featuresDict = defaultdict(list)
        vertex_len = len(cellIds)
        batch_size = vertex_len // 8
        for batch in range(0, vertex_len, batch_size):
            p_slice = [slice(batch + i, min(batch + batch_size, vertex_len), process_n) for i in range(process_n)]
            args = [[cellIds[slice_i], contours[slice_i], bboxes[slice_i]] for _, slice_i in enumerate(p_slice)]
            with mp.Pool(process_n) as p:
                ans = p.map(SingleMorphFeatures, args)
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    featuresDict[k] += v
    return featuresDict


def getCellImg(slidePtr, bbox, pad=2, level=0):
    bbox = np.array(bbox)
    bbox[0] = bbox[0] - pad
    bbox[1] = bbox[1] + pad
    cellImg = slidePtr.read_region(location=bbox[0] * 2 ** level, level=level, size=bbox[1] - bbox[0])
    cellImg = np.array(cv2.cvtColor(np.asarray(cellImg), cv2.COLOR_RGB2GRAY))
    return cellImg


def getCellMask(contour, bbox, pad=2, level=0):
    if level != 0:
        raise KeyError('Not support level now')
    (left, top), (right, bottom) = bbox
    height, width = bottom - top, right - left
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    cellMask = np.zeros((height + pad * 2,
                         width + pad * 2),
                        dtype=np.uint8)
    contour[:, 0] = contour[:, 0] - left + pad 
    contour[:, 1] = contour[:, 1] - top + pad
    cv2.drawContours(cellMask, [contour], 0, 1, -1)
    return cellMask


def mygreycoprops(P):
    # reference https://murphylab.web.cmu.edu/publications/boland/boland_node26.html
    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    Pxplusy = np.zeros((num_level + num_level2 - 1, num_dist, num_angle))
    Ixplusy = np.expand_dims(np.arange(num_level + num_level2 - 1), axis=(1, 2))
    P_flip = np.flip(P, axis=0)
    for i, offset in enumerate(range(num_level - 1, -num_level2, -1)):
        Pxplusy[i] = np.trace(P_flip, offset)
    SumAverage = np.sum(Ixplusy * Pxplusy, axis=0)
    Entropy = - np.sum(Pxplusy * np.log(Pxplusy + 1e-15), axis=0)
    SumVariance = np.sum((Ixplusy - Entropy) ** 2 * Pxplusy, axis=0)

    Ix = np.tile(np.arange(num_level).reshape(-1, 1, 1, 1), [1, num_level2, 1, 1])
    Average = np.sum(Ix * P, axis=(0, 1))
    Variance = np.sum((Ix - Average) ** 2 * P, axis=(0, 1))
    return SumAverage, Entropy, SumVariance, Average, Variance


def SingleGLCMFeatures(args):
    wsiPath, cellIds, contours, bboxes, pad, level = args
    slidePtr = OpenSlide(wsiPath)
    # Use wsipath as parameter because multiprocess can't use pointer like the object OpenSlide() as parameter
    featuresDict = defaultdict(list)
    
    for cellId,contour, bbox in zip(cellIds,contours, bboxes):
        featuresDict['cellId'] += [cellId]
        cellImg = getCellImg(slidePtr, bbox, pad, level)
        cellmask = getCellMask(contour, bbox, pad).astype(np.bool_)
        cellImg[~cellmask] = 0

        outMatrix = skfeat.graycomatrix(cellImg, [1], [0])
        outMatrix[0, :, ...] = 0
        outMatrix[:, 0, ...] = 0

        dissimilarity = skfeat.graycoprops(outMatrix, 'dissimilarity')[0][0]
        homogeneity = skfeat.graycoprops(outMatrix, 'homogeneity')[0][0]
        # energy = skfeat.greycoprops(outMatrix, 'energy')[0][0]
        ASM = skfeat.graycoprops(outMatrix, 'ASM')[0][0]
        contrast = skfeat.graycoprops(outMatrix, 'contrast')[0][0]
        correlation = skfeat.graycoprops(outMatrix, 'correlation')[0][0]
        SumAverage, Entropy, SumVariance, Average, Variance = mygreycoprops(outMatrix)

        featuresDict['ASM'] += [ASM]
        featuresDict['Contrast'] += [contrast]
        featuresDict['Correlation'] += [correlation]
        # featuresDict['Dissimilarity'] += [dissimilarity]
        featuresDict['Entropy'] += [Entropy[0][0]]
        featuresDict['Homogeneity'] += [homogeneity]
        # featuresDict['Energy'] += [energy] #Delete because similar with ASM
        # featuresDict['Average'] += [Average[0][0]]
        # featuresDict['Variance'] += [Variance[0][0]]
        # featuresDict['SumAverage'] += [SumAverage[0][0]]
        # featuresDict['SumVariance'] += [SumVariance[0][0]]

        featuresDict['IntensityMean'] += [cellImg[cellmask].mean()]
        featuresDict['IntensityStd'] += [cellImg[cellmask].std()]
        featuresDict['IntensityMax'] += [cellImg[cellmask].max().astype('int16')]
        featuresDict['IntensityMin'] += [cellImg[cellmask].min().astype('int16')]
        # featuresDict['IntensitySkewness'] += [stats.skew(cellImg.flatten())] # Plan to delete this feature
        # featuresDict['IntensityKurtosis'] += [stats.kurtosis(cellImg.flatten())] # Plan to delete this feature
    return featuresDict


def getGLCMFeatures(wsiPath, cellIds, contours, bboxes, pad=2, level=0, process_n=1):
    
    if process_n == 1:
        return SingleGLCMFeatures([wsiPath, cellIds, contours, bboxes, pad, level])
    else:
        featuresDict = defaultdict(list)
        vertex_len = len(cellIds)
        batch_size = vertex_len // 8
        for batch in range(0, vertex_len, batch_size):
            p_slice = [slice(batch + i, min(batch + batch_size, vertex_len), process_n) for i in range(process_n)]
            args = [[wsiPath, cellIds[slice_i], contours[slice_i], bboxes[slice_i], pad, level] for _, slice_i in enumerate(p_slice)]
            with mp.Pool(process_n) as p:
                ans = p.map(SingleGLCMFeatures, args)
            for q_info in ans:
                for k, v in zip(q_info.keys(), q_info.values()):
                    featuresDict[k] += v
    return featuresDict


def loadGeoJson(geojson_path):
    """
    GeoJson files exported from Qupath, the locations are in micrometer unit
    """
    with open(geojson_path,'r') as f:
        allobjects = json.load(f)
        contour_dict={}
        for j,feature in enumerate(allobjects['features']):
            if feature['properties']['objectType'] != 'detection':#could only be 'annotation' or 'detection'
                continue
            if feature['geometry']['type']!='Polygon':#very few cells are multipolygons, ignore
                continue
            cell_contour = feature['geometry']['coordinates'][0]#a list of n lists of length 2
            cell_id = feature['id']
            contour_dict[cell_id]=cell_contour
        return contour_dict
                


def main(geojson_dir, wsi_dir, out_dir, process_n=8,offset=[0,0]):
    morph_out_dir = os.path.join(out_dir,'morphology')
    texture_out_dir = os.path.join(out_dir,'texture')
    os.makedirs(morph_out_dir,exist_ok=True)
    os.makedirs(texture_out_dir,exist_ok=True)
    for i,fname in enumerate(os.listdir(geojson_dir)):
        print(f'start processing {fname}')
        geojson_path=os.path.join(geojson_dir, fname)
        if len(fname.split('.')[0].split('_'))==3:#biopsy format HE_{patientNum}_{slideId}.svs
            slideId = fname.split('.')[0].split('_')[-1]
        elif len(fname.split('.')[0].split('_'))==2:#exc44 format {slideId}_H&E.svs
            slideId = fname.split('.')[0].split('_')[-2]

        t0 = time.time()
        raw_contour_dict = loadGeoJson(geojson_path)
        print(f"{'loading geojson cost':#^40s}, {time.time() - t0:*^10.2f}")

        cellIds,bboxes, centroids, contours = [], [], [], []
        for cellId,contour in raw_contour_dict.items():
            contour = np.round(np.array(contour)).astype(np.int32)
            left, top = contour.min(0)
            right, bottom = contour.max(0)
            bbox = [[left + offset[0], top + offset[1]], [right + offset[0], bottom + offset[1]]]
            centroid = [(bbox[0][0]+bbox[1][0])/2,(bbox[0][1]+bbox[1][1])/2]
            cellIds.append(cellId)
            bboxes.append(bbox)  
            centroids.append(centroid)
            contours.append(contour)
        assert len(cellIds)==len(bboxes) == len(centroids) , 'The attribute of nodes (bboxes, centroids, types) must have same length'
        vertex_len = len(cellIds)
        if vertex_len == 0:
            print('No cells')
            continue
        print('Getting morph features')
        t1 = time.time()
        morphFeats = getMorphFeatures(cellIds, contours, bboxes, 'MorphFeatures', process_n=process_n)
        out_path = os.path.join(morph_out_dir,f'{slideId}.json')
        with open(out_path,'w') as f:
            json.dump(morphFeats,f)
        print(f"{'morph features cost':#^40s}, {time.time() - t1:*^10.2f}")
        
        print('Getting GLCM features')
        t2 = time.time()
        wsiPath = os.path.join(wsi_dir,f'{slideId}.svs')
        level=0
        GLCMFeats = getGLCMFeatures(wsiPath, cellIds, contours, bboxes, pad=2, level=level, process_n=process_n)
        out_path = os.path.join(texture_out_dir,f'{slideId}.pkl')
        with open(out_path,'wb') as f:
            pickle.dump(GLCMFeats,f)
        print(f"{'GLCM features cost':#^40s}, {time.time() - t2:*^10.2f}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process biopsy images.")
    parser.add_argument("--geojson_dir", type=str, required=True, help="Directory path for the geojson files.")
    parser.add_argument("--wsi_dir", type=str, required=True, help="Directory path for the Whole Slide Images (WSI).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory path for morphological&texture features.")
    parser.add_argument("--process_n", type=int, required=True, help="Pixel length.")
    
    args = parser.parse_args()

    main(args.geojson_dir, args.wsi_dir, args.out_dir,args.process_n)

"""
geojson_dir = '/mnt/data10/shared/yujie/DCIS/cellDet_data/starDist/HEcell_starDist_geojson_biopsy'
wsi_dir = '/mnt/data10/shared/yujie/DCIS/biopsy_img_svs/HE'
out_dir = '/mnt/data10/shared/yujie/DCIS/ANALYSIS/biopsy_features/'
process_n = 8
python computeMor_text.py --geojson_dir /mnt/data10/shared/yujie/DCIS/cellDet_data/starDist/HEcell_starDist_geojson_biopsy --wsi_dir /mnt/data10/shared/yujie/DCIS/biopsy_img_svs/HE --out_dir /mnt/data10/shared/yujie/DCIS/ANALYSIS/biopsy_features/ --process_n 8 &
python computeMor_text.py --geojson_dir /mnt/data10/shared/yujie/DCIS/cellDet_data/starDist/HEcell_starDist_geojson_exc44 --wsi_dir /mnt/data10/shared/yujie/DCIS/excision_batch4_5_Oct3/svs/HE --out_dir /mnt/data10/shared/yujie/DCIS/ANALYSIS/exc44_features/ --process_n 8 &

python computeMor_text.py --geojson_dir /mnt/data10/shared/yujie/new_DCIS/review_exc/HE_starDist_geojson --wsi_dir /mnt/data10/shared/yujie/DCIS/excision_batch4_5_Oct3/svs/HE --out_dir /home/yxiao/review_exc/HE_cellFeatures --process_n 8 &
"""