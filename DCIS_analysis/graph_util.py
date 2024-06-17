'''
ripley_k_iso() is inspired by https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/Ripley_K_demo.ipynb
'''

import numpy as np
import pandas as pd

import networkx as nx
import alphashape
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import pdist, squareform

from astropy.stats import RipleysKEstimator

from scipy.spatial import Voronoi, Delaunay, KDTree
from scipy.stats import kurtosis, skew


#########For IHC defined regions
def construct_graph_IHC(cells, threshold):
    G = nx.Graph()
    n = len(cells)
    # #include all the points and their lcoations
    # for i in range(n):
    #     G.add_node(i, pos=tuple(cells[i]))
    dist_matrix = squareform(pdist(cells, 'euclidean'))
    mask = dist_matrix < threshold
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                G.add_edge(i, j)
    return G
def find_connected_components(graph):
    return [list(component) for component in nx.connected_components(graph)]

def calculate_concave_hull(cells, components, alpha=0.01):#0 means convex, the larger the more discrete
    regions = []
    for component in components:
        points = [(cells[i][0], cells[i][1]) for i in component]
        
        try:
            concave_hull = alphashape.alphashape(points, alpha)
            regions.append(concave_hull)
        except Exception as e:
            print(f'number of points {len(points)}, cant get concave hull')
            try:
                regions.append(Polygon(points))
            except:
                print(f'number of points {len(points)}, cant get concave hull or polygon')
    
    return regions

def get_occupied_regions(cells, threshold, alpha=0.01):
    graph = construct_graph_IHC(cells, threshold)
    components = find_connected_components(graph)
    regions = calculate_concave_hull(cells, components, alpha)
    return regions


################# For HE graph features

def calculate_statistics(data):
    data_array = np.array(data)
    if data_array.size == 0:
        return {
            'mean': None,
            'sd': None,
            'min': None,
            'max': None,
            'kurtosis': None,
            'skewness': None
        }
    return {
        'mean': np.mean(data),
        'sd': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'kurtosis': kurtosis(data),
        'skewness': skew(data)
    }

def calculate_voronoi_features(points):
    try:
        vor = Voronoi(points)
        areas, perimeters, chord_lengths = [], [], []
        for region in vor.regions:
            if -1 not in region and region:
                polygon = Polygon([vor.vertices[i] for i in region if i >= 0])
                areas.append(polygon.area)
                perimeters.append(polygon.length)
                for i in range(len(region)-1):
                    start = vor.vertices[region[i]]
                    end = vor.vertices[region[(i + 1) % len(region)]]
                    chord_lengths.append(np.linalg.norm(end - start))

        return {
            **{f'voronoi_area_{k}': v for k, v in calculate_statistics(areas).items()},
            **{f'voronoi_perimeter_{k}': v for k, v in calculate_statistics(perimeters).items()},
            **{f'voronoi_chord_length_{k}': v for k, v in calculate_statistics(chord_lengths).items()}
        }
    except Exception as e:
        print(f"Error in calculate_voronoi_features: {e}")
        return {
            **{f'voronoi_area_{k}': None for k in ['mean','sd','min','max','kurtosis','skewness']},
            **{f'voronoi_perimeter_{k}': None for k in ['mean','sd','min','max','kurtosis','skewness']},
            **{f'voronoi_chord_length_{k}': None for k in ['mean','sd','min','max','kurtosis','skewness']}
        }
                         
def calculate_delaunay_features(points):
    try:
        tri = Delaunay(points)
        side_lengths, areas = [], []
        for simplex in tri.simplices:
            pts = points[simplex]
            a, b, c = np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[2] - pts[0])
            side_lengths.extend([a, b, c])
            s = (a + b + c) / 2
            area_term = s * (s - a) * (s - b) * (s - c)
            area = np.sqrt(area_term) if area_term >=0 else 0
            areas.append(area)

        return {
            **{f'delaunay_side_length_{k}': v for k, v in calculate_statistics(side_lengths).items()},
            **{f'delaunay_area_{k}': v for k, v in calculate_statistics(areas).items()}
        }
    except Exception as e:
        print(f"Error in calculate_delauney_features: {e}")
        return {
            **{f'delaunay_side_length_{k}': None for k in ['mean','sd','min','max','kurtosis','skewness']},
            **{f'delaunay_area_{k}': None for k in ['mean','sd','min','max','kurtosis','skewness']},
        }

def calculate_mst_features(points):
    try:
        tri = Delaunay(points)
        graph = nx.Graph()
        for simplex in tri.simplices:
            graph.add_edge(simplex[0], simplex[1], weight=np.linalg.norm(points[simplex[0]] - points[simplex[1]]))
            graph.add_edge(simplex[1], simplex[2], weight=np.linalg.norm(points[simplex[1]] - points[simplex[2]]))
            graph.add_edge(simplex[2], simplex[0], weight=np.linalg.norm(points[simplex[2]] - points[simplex[0]]))

        #MST of the delauney triangulation graph
        mst = nx.minimum_spanning_tree(graph)
        
        edge_lengths = [data['weight'] for _, _, data in mst.edges(data=True)]

        
        return {f'mst_edge_length_{k}': v for k, v in calculate_statistics(edge_lengths).items()}
    except Exception as e:
        print(f"Error in calculate_mst_features: {e}")
        return {
            **{f'mst_edge_length_{k}': None for k in ['mean','sd','min','max','kurtosis','skewness']},
        }
def calculate_nn_features(points, k):
    try:
        tree = KDTree(points)
        distances, _ = tree.query(points, k=k+1)
        return {f'nn_k{k}_{stat}': value for stat, value in calculate_statistics(distances[:, 1:].flatten()).items()}
    except Exception as e:
        print(f"Error in calculate_nn_features_{k}: {e}")
        return {
            **{f'nn_k{k}_{stat}': None for stat in ['mean','sd','min','max','kurtosis','skewness']},
        }
def calculate_H(points):
    r = np.linspace(5,50,10)
    try:
        
        K = ripley_k_astropy(points,r)
        H = np.sqrt(K)/r-r
        return {f'H_{int(ri)}': Hi for ri,Hi in zip(r,H)}
    except Exception as e:
        print(f"Error in calculate_H: {e}")
        return {f'H_{int(ri)}': None for ri in r}
def calculate_spat_features(points):
    cell_count = len(points)
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    area = (max_x - min_x) * (max_y - min_y)
    density = cell_count / area if area > 0 else 0
    features = {
        'Cell Count': cell_count,
        'Area': area,
        'Density': density
    }
    features.update(calculate_voronoi_features(points))
    features.update(calculate_delaunay_features(points))
    features.update(calculate_mst_features(points))
    features.update(calculate_nn_features(points, 3))
    features.update(calculate_nn_features(points, 5))
    features.update(calculate_nn_features(points, 7))
    features.update(calculate_H(points))
    return features



def ripley_k_astropy(points,radii):
    x_min,x_max,y_min,y_max = np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])
    area = (x_max - x_min)*(y_max - y_min)
    Kest = RipleysKEstimator(area=area, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)
    #radii = np.linspace(0,max(x_max-x_min,y_max-y_min),50).reshape(50,1)   
    return Kest(data=points, radii=radii, mode='ripley')#mode can be 'translation'/'ohser'/'var_width'
    #H(r) = sqrt(K)/r-r

def ripley_k_iso(points):
    x_min,x_max,y_min,y_max = np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])
    area = (x_max - x_min)*(y_max - y_min)

    radii = np.linspace(0,max(x_max-x_min,y_max-y_min),50).reshape(50,1)   
    npts = np.shape(points)[0]                                 # Number of events in A

    diff = np.zeros(shape = (npts*(npts-1)//2,2))               # Decomposed distances matrix
    k = 0
    for i in range(npts - 1):
        size = npts - i - 1
        diff[k:k + size] = abs(sample1[i] - sample1[i+1:])
        k += size

    ripley = np.zeros(len(radii))

    hor_dist = np.zeros(shape=(npts * (npts - 1)) // 2, dtype=np.double)
    ver_dist = np.zeros(shape=(npts * (npts - 1)) // 2, dtype=np.double)

    for k in range(npts - 1):                           # Finds horizontal and vertical distances from every event to nearest egde 
        min_hor_dist = min(x_max - sample1[k][0], sample1[k][0] - x_min)
        min_ver_dist = min(y_max - sample1[k][1], sample1[k][1] - y_min)
        start = (k * (2 * (npts - 1) - (k - 1))) // 2
        end = ((k + 1) * (2 * (npts - 1) - k)) // 2
        hor_dist[start: end] = min_hor_dist * np.ones(npts - 1 - k)
        ver_dist[start: end] = min_ver_dist * np.ones(npts - 1 - k)

        
    dist = np.hypot(diff[:, 0], diff[:, 1])
    dist_ind = dist <= np.hypot(hor_dist, ver_dist)     # True if distance between events is less than or equal to distance to edge

    w1 = (1 - (np.arccos(np.minimum(ver_dist, dist) / dist) + np.arccos(np.minimum(hor_dist, dist) / dist)) / np.pi)
    w2 = (3 / 4 - 0.5 * (np.arccos(ver_dist / dist * ~dist_ind) + np.arccos(hor_dist / dist * ~dist_ind)) / np.pi)

    weight = dist_ind * w1 + ~dist_ind * w2              # Weighting term

    for r in range(len(radii)):
        ripley[r] = ((dist < radii[r]) / weight).sum()   # Indicator function with weighting term

    ripley = area * 2. * ripley / (npts * (npts - 1))
    return radii,ripley