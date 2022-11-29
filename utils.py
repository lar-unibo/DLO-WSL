import numpy as np
import cv2
from scipy.interpolate import splprep, splev




def borders_from_center(points_center_3d, diameter=4/1000):

    points_1, points_2 = [], []
    for i, p in enumerate(points_center_3d):
        if i != points_center_3d.shape[0]-1:
            neigh_p = points_center_3d[i+1]
        else:
            neigh_p = points_center_3d[i-1]

        dir = neigh_p - p
        dir[2] = 0
        dir = dir / np.linalg.norm(dir)

        dir_orth = np.array([dir[1], -dir[0], 0])

        if i != points_center_3d.shape[0]-1:
            p1 = p + dir_orth * diameter/2
            p2 = p - dir_orth * diameter/2
        else:
            p2 = p + dir_orth * diameter/2
            p1 = p - dir_orth * diameter/2           

        points_1.append(p1)
        points_2.append(p2)
    
    points_1_arr = np.array(points_1)
    points_2_arr = np.array(points_2)
    return points_1_arr, points_2_arr


def draw_function(mask, arr_points, color, radius):
    points = np.array(arr_points)
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    c = [int(color[0]*255), int(color[1]*255), int(color[2]*255)]
    cv2.polylines(mask, [pts], False, c, radius)    

def draw_mask_poly_lines(arr_points, shape, radius, is_closed=False):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(arr_points)
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(mask, [pts], is_closed, 255, radius)    
    return mask

def draw_mask_poly_fill(arr_points, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array(arr_points)
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)    
    return mask

def get_xyz(px, py, depth, camera_matrix):

    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1] 
    cx = camera_matrix[0,2]
    cy = camera_matrix[1,2]

    x = depth * (px - cx) / fx
    y = depth * (py - cy) / fy
    z = depth

    return x, y, z


def get_pxpy(x, y, z, camera_matrix):

    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1] 
    cx = camera_matrix[0,2]
    cy = camera_matrix[1,2]

    px = (x * fx) / z + cx
    py = (y * fy) / z + cy
    return px, py


def compute_spline(points, k=3, smoothing=0.0, periodic=0, num_points=100):
    tck, u = splprep(np.array(points).T, u=None, k=k, s=smoothing, per=periodic)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.stack([x_new, y_new]).T

def compute_spline_2(points, k=3, smoothing=0.0, periodic=0, num_points=100):
    tck, u = splprep(np.array(points).T, u=None, k=k, s=smoothing, per=periodic)
    u_new = np.linspace(u.min(), u.max(), num_points)

    x_new, y_new = splev(u_new, tck, der=0)
    xp, yp = splev(u_new, tck, der=1)
    return np.stack([x_new, y_new]).T, np.stack([xp, yp]).T



def compute_spline_3D(points, k=3, smoothing=0.0, periodic=0, num_points=100):
    #points = np.unique(points, axis=0)
    tck, u = splprep(np.array(points).T, u=None, k=k, s=smoothing, per=periodic)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new, z_new = splev(u_new, tck, der=0)
    return np.stack([x_new, y_new, z_new]).T

def projection(camera_pose, points_3d, camera_model, use_distortion=False, keep_invalid=False):
    T = np.linalg.inv(camera_pose)
    tvec =np.array(T[0:3, 3])
    rvec, _ = cv2.Rodrigues(T[:3,:3]) 
    if use_distortion:
        point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, camera_model.K, camera_model.D)
    else:
        point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, camera_model.K, None)

    points_ref = []
    for p in point2d[0].squeeze():
        i, j = [round(p[1]), round(p[0])]
        if i < camera_model.H and i >= 0 and j < camera_model.W and j >= 0:
            points_ref.append(tuple([j,i]))
        else:
            if keep_invalid:
                points_ref.append(None)
    
    return np.array(points_ref)

def unproject(points, z_values, camera_model, use_distortion=True):

    if use_distortion:
        points_undistorted = np.array([])
        if len(points) > 0:
            points_undistorted = cv2.undistortPoints(points.astype(np.float64), camera_model.K, camera_model.D, P=camera_model.K)
        points_undistorted = np.squeeze(points_undistorted, axis=1)
    else:
        points_undistorted = points

    result = []
    for idx in range(points_undistorted.shape[0]):
        x,y,z = get_xyz(points_undistorted[idx,0], points_undistorted[idx,1], z_values[idx], camera_model.K)
        result.append([x, y, z])
    
    return np.array(result)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def distance_2D(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def index_point_to_list_min_distance(point, list_points):
    distances = [distance_2D(point, p) for p in list_points]
    return np.argmin(distances)



def distort(points_und, camera_model, camera_matrix_opt):
    pts_out = cv2.undistortPoints(np.array(points_und, dtype='float32'), camera_matrix_opt, None)
    pts_temp = cv2.convertPointsToHomogeneous(pts_out)
    pts_proj = cv2.projectPoints(pts_temp, np.array([0,0,0], dtype='float32'), np.array([0,0,0], dtype='float32'), camera_model.K, camera_model.D, pts_out)

    points_out = []
    for p in pts_proj[0].squeeze():
        i, j = [round(p[1]), round(p[0])]
        if i < camera_model.H and i >= 0 and j < camera_model.W and j >= 0:
            points_out.append(tuple([j,i]))
    
    return points_out

def undistort(img, camera_model):
    h, w = img.shape[:2]
    camera_matrix_refined, _ = cv2.getOptimalNewCameraMatrix(camera_model.K, camera_model.D, (w, h), 1, (w, h))
    return cv2.undistort(img, camera_model.K, camera_model.D, None, camera_matrix_refined), camera_matrix_refined

