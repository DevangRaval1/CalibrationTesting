import numpy as np
import cv2
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv
from scipy.linalg import expm, inv
from numpy import dot, eye
import os
import glob, ast
from numpy.linalg import inv, norm
from scipy.linalg import expm
import numpy as np
from numpy.linalg import inv, norm, svd

#use the opencv function to find the corners in the image with chessboard pattern
def find_corners(image):
    found, corners = cv2.findChessboardCorners(image, pattern_size)
    # cv2.drawChessboardCorners(image, pattern_size, corners, True)
    # cv2.imshow("Corners", image)
    # cv2.waitKey(0)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    if found:
        cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), term)
    return found, corners

#draw the coners not use in main function but to test if the image has detectable corners and if they are correct orientation
def draw_corners(image, corners):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(color_image, pattern_size, corners, True)
    cv2.imshow("Corners", color_image)
    cv2.waitKey(0)
    return color_image

# use the opencv function to get the pose of the object in the image
def get_object_pose(object_points, image_points, camera_matrix, dist_coeffs):
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    return rvec.flatten(), tvec.flatten()

# calibrate the camera using the chessboard pattern returns the camera matrix and the distortion coefficients for the camera
# calibrate() then solves for X in A_i X = X B_i.(Tsai method)
def calibrate_lens(image_list):
    img_points, obj_points = [], []
    h,w = 0, 0
    for img in image_list:
        h, w = img.shape[:2]
        found,corners = find_corners(img)
        if not found:
            raise Exception("chessboard calibrate_lens Failed to find corners in img")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
    camera_matrix = np.zeros((3,3))
    dist_coeffs = np.zeros(5)
    cv2.calibrateCamera(obj_points, img_points, (w,h), camera_matrix, dist_coeffs)
    return camera_matrix, dist_coeffs

# calculate the rotation vector from the rotation matrix R (SO(3) “matrix logarithm”)
def log(R, eps=1e-8):
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)

    if abs(theta) < eps:
        return np.zeros(3)        
    return np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) * theta / (2.0 * np.sin(theta))

# calculate the inverse square root of a matrix using SVD
def invsqrt(mat, eps=1e-8):
    U, s, Vt = np.linalg.svd(mat)
    s_inv = 1.0 / np.sqrt(np.clip(s, eps, None))
    return U @ np.diag(s_inv) @ Vt

# calibrate the robot using the object pose and the camera pose
#a is the robot pose to the chessboard and B is the object pose to the chessboard
# will return the rotation matrix and the translation vector
# the rotation matrix is the rotation from the robot to the camera and the translation vector is the translation from the robot to the camera
def calibrate(A, B):
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))

    Rx = dot(invsqrt(dot(M.T, M)), M.T)

    C = zeros((3*N, 3))
    d = zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - dot(Rx, tb)

    tx = dot(inv(dot(C.T, C)), dot(C.T, d))
    return Rx, tx.flatten()

# create skew-symmetric matrix from a vector
def hat(v):
    return [[   0, -v[2],  v[1]],
            [v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]

# create a transformation matrix from a rotation vector and a translation vector
def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm(hat(r))
    res[0:3, -1] = t
    return res

# known dataset path and loading
# img_dir  = r"C:\Users\draval\OneDrive - Bishop-Wisecarver Corporation\Desktop\FanucStuff\Code\VaccinePython\Project\dataset\image"
# pose_dir = r"C:\Users\draval\OneDrive - Bishop-Wisecarver Corporation\Desktop\FanucStuff\Code\VaccinePython\Project\dataset\pose"

# img_paths  = sorted(glob.glob(os.path.join(img_dir,  "image*.png")))
# pose_paths = sorted(glob.glob(os.path.join(pose_dir, "pose*.txt")))

# custom dataset path and loading
IMG_POSE_DIR = r"C:\Users\draval\OneDrive - Bishop-Wisecarver Corporation\Desktop\FanucStuff\Code\VaccinePython\Project\src\Calinration_UI_Testing\captures"

img_paths  = sorted(glob.glob(os.path.join(IMG_POSE_DIR,  "image*.png")))
pose_paths = sorted(glob.glob(os.path.join(IMG_POSE_DIR, "pose*.txt")))

assert len(img_paths) == len(pose_paths), "Count mismatch between images and poses!"

#load all the images into a list
img_list = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

#loading all the robot poses as a list of matrices
rob_pose_list = []
for path in pose_paths:
    with open(path, 'r') as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        # below 4 lines for known dataset
        # clean = line.strip().lstrip('[').rstrip(']').replace(',', '')
        # if not clean:
        #     continue
        # parts = clean.split()

        # below 3 lines for custom dataset 
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Expected 4 numbers/line in {path}, got {parts!r}")
        rows.append([float(x) for x in parts])
    mat = np.array(rows, dtype=float)
    if mat.shape != (4,4):
        raise ValueError(f"{path} parsed to {mat.shape}, expected (4,4)")
    rob_pose_list.append(mat)

#custom dataset chessboard pattern
pattern_size = (9, 7)
square_size = 20 # mm

#known dataset chessboard pattern
# pattern_size = (10, 6)
# square_size = 0.020 # m

#making the pattern points as a list of 3D points
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

corner_list = []
obj_pose_list = []

#get the camera matrix and the distortion coefficients using the function calibrate_lens
camera_matrix, dist_coeffs = calibrate_lens(img_list)

#loop through all the images and find the corners in the image and get the pose of the object in the image
for i, img in enumerate(img_list):
    found, corners = find_corners(img)
    corner_list.append(corners)
    if not found:
        raise Exception("Failed to find corners in img # %d" % i)
    rvec, tvec = get_object_pose(pattern_points, corners, camera_matrix, dist_coeffs)
    object_pose = tf_mat(rvec, tvec)
    obj_pose_list.append(object_pose)

# build relative transform from 1 pose to next
# A is the robot pose to the chessboard and B is the object pose to the chessboard
#neede for the calibration
A, B = [], []
for i in range(1,len(img_list)):
    p = rob_pose_list[i-1], obj_pose_list[i-1]
    n = rob_pose_list[i], obj_pose_list[i]
    A.append(dot(inv(p[0]), n[0]))
    B.append(dot(inv(p[1]), n[1]))

# make an idnetity matrix to store the rotation and translation gotten from the calibration
X = eye(4)
Rx, tx = calibrate(A, B)
X[0:3, 0:3] = Rx
X[0:3, -1] = tx

#display the transformation matrix from robot base to camera
# this should be the same for all the images
print("\nTransformation Matrix from robot base to camera: ")
print(X)

print("\nChecking the transformation matrix is true for all the images")
print("All the transformations should be quite similar/same\n")

for i in range(len(img_list)):
    rob = rob_pose_list[i]
    obj = obj_pose_list[i]
    tmp = dot(rob, dot(X, inv(obj)))
    print(tmp)

rob = rob_pose_list[0]
obj = obj_pose_list[0]
cam_pose = dot(dot(rob, X), inv(obj))

cam = {'rotation' : cam_pose[0:3, 0:3].tolist(),
       'translation' : cam_pose[0:3, -1].tolist(),
       'camera_matrix' : camera_matrix.tolist(),
       'dist_coeffs' : dist_coeffs.tolist()}

##########################################################################################################################
## Testing that all the matrix generated are correct
#This is done by removing 1 image and doing the calibration on rest of the images and then calculating the pose from the image and comparing it to the pose from the robot

#calculates the mean of the transformation matrices as did before
# this is done by summing the rotation matrices and then taking the svd of the sum
def mean_transform(T_list):
    R_sum = sum(T[:3,:3] for T in T_list)
    U, _, Vt = svd(R_sum)
    R_mean = U @ Vt
    t_mean = np.mean([T[:3,3] for T in T_list], axis=0)
    Tm = np.eye(4)
    Tm[:3,:3] = R_mean
    Tm[:3, 3] = t_mean
    return Tm

# calculation of the transformation matrix without 1 image then testing the transformation matrix on the image
def predict_gripper_pose(img_list, rob_pose_list, leave_idx):
    #removing the image index mentioned
    idxs_train = [i for i in range(len(img_list)) if i != leave_idx]
    #rest same as calibration before
    imgs_train  = [img_list[i]       for i in idxs_train]
    poses_train = [rob_pose_list[i]  for i in idxs_train]

    cam_mtx, dist = calibrate_lens(imgs_train)

    obj_train = []
    for img in imgs_train:
        found, corners = find_corners(img)
        if not found:
            raise RuntimeError("Corner not found in a train image")
        rvec, tvec = get_object_pose(pattern_points, corners, cam_mtx, dist)
        obj_train.append(tf_mat(rvec, tvec))

    A, B = [], []
    for j in range(1, len(idxs_train)):
        A.append(inv(poses_train[j-1]) @ poses_train[j])
        B.append(inv(obj_train[j-1])  @ obj_train[j])
    Rx, tx = calibrate(A, B)
    X = np.eye(4)
    X[:3,:3], X[:3,3] = Rx, tx

    T_base_obj_list = [poses_train[j] @ X @ inv(obj_train[j])
                       for j in range(len(idxs_train))]
    T_base_obj_mean = mean_transform(T_base_obj_list)

    #########
    #after getting the transformation matrix we need to test it on the image that was removed
    #load the image and the robot pose
    img_test  = img_list[leave_idx]
    rob_test  = rob_pose_list[leave_idx]
    #get the corners of chessboard in the image
    found, corners = find_corners(img_test)
    if not found:
        raise RuntimeError(f"No corners in test image #{leave_idx}")

    #get the object pose in the image
    rvec, tvec = get_object_pose(pattern_points, corners, cam_mtx, dist)
    obj_test = tf_mat(rvec, tvec)

    #predict the gripper pose in base frame for the held-out image
    g_pred = T_base_obj_mean @ obj_test @ inv(X)

    #get the error
    E = inv(rob_test) @ g_pred
    trans_err = norm(E[:3,3])
    rot_err   = np.arccos((np.trace(E[:3,:3]) - 1)/2)
    print("\n\n")
    print(f"Custom Calculations:\n")
    print(f"predicted_wrist_pose: \n{g_pred} \n")
    print(f"error_transform: \n{E}\n")
    print(f"translation_error_mm:\n {trans_err}\n")
    print(f"rotation_error_deg:\n {rot_err}\n")
    return {
        'predicted_wrist_pose': g_pred,
        'error_transform':      E,
        'translation_error_m':  trans_err,
        'rotation_error_rad':   rot_err
    }

result = predict_gripper_pose(img_list, rob_pose_list, leave_idx=2)