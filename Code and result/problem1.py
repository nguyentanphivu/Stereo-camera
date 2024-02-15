from __future__ import print_function
import numpy as np 
import cv2
from copy import deepcopy

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    # Function to write ply file
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

## Image size
height = 720
width = 1280

cut_height = 680
cut_width = 1240

# Threshold for warning
low_thresh = 0
upper_thresh = 100 #this value mean that our program will warning if the distance of the object is < 100 cm

# Calibrate system, the matrices are gotten from the previous assignment
intrinsicR = np.array([[1427.311685713886, 0.0, 646.2119594419198], [0.0, 1426.2572995156333, 334.9699403574512], [0.0, 0.0, 1.0]])
intrinsicL = np.array([[1420.735770244577, 0.0, 636.5024366111307], [0.0, 1417.8190134722824, 342.0848143866361], [0.0, 0.0, 1.0]])
distortR= np.array([[0.03213557067813336, 0.6343405835195601, -0.005779435303546466, 0.0018456241520351075, -0.5670881302057097]])
distortL = np.array([[0.03234090555243866, 0.8097846333391775, -0.0002400095276759211, 0.001973497542532829, -2.818388480312125]])
update_intrinsicR, _ = cv2.getOptimalNewCameraMatrix(intrinsicR, distortR, (width, height), 1, (width, height))
update_intrinsicL, _ = cv2.getOptimalNewCameraMatrix(intrinsicL, distortL, (width, height), 1, (width, height))

# Set ID for camera
camL_id = 0 # Left camera
camR_id = 1 # Right camera

camL = cv2.VideoCapture(camL_id)
camR = cv2.VideoCapture(camR_id)

def click_event(event, x, y, flags, params):
    # print the distance in cm when click on the object
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        print(disparity[y][x])
        print(disparity1[y][x])
        print(163000/disparity1[y][x], 'cm')
    
def nothing(x):
    pass

def warning(disparity):
    # Function to warn if the object is close to the cam   
    disparity = 163000/disparity
    mask = cv2.inRange(disparity, low_thresh, upper_thresh) #Thresholding
    mask2 = np.zeros_like(mask)
    
    # Check if a significantly large obstacle is present and filter out smaller noisy regions
    if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
    
        # Contour detection 
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Check if detected contour is significantly large (to avoid multiple tiny regions)
        if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
                
            cv2.drawContours(mask2, cnts, 0, (255), -1)

            return True, mask, mask2
        
        else:
            return False, mask, mask2
        
    else:
        return False, mask, mask2
    
# Creating tune
cv2.namedWindow('Tune',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tune',500,500)
cv2.createTrackbar('Tune Mode?', 'Tune', 1, 1, nothing)
cv2.createTrackbar('numDisparities','Tune',6,100,nothing)
cv2.createTrackbar('blockSize','Tune',8,100,nothing)
cv2.createTrackbar('preFilterType','Tune',0,100,nothing)
cv2.createTrackbar('preFilterSize','Tune',0,100,nothing)
cv2.createTrackbar('preFilterCap','Tune',0,100,nothing)
cv2.createTrackbar('textureThreshold','Tune',0,100,nothing)
cv2.createTrackbar('uniquenessRatio','Tune',0,100,nothing)
cv2.createTrackbar('speckleRange','Tune',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','Tune',0,100,nothing)
cv2.createTrackbar('disp12MaxDiff','Tune',100,200,nothing)
cv2.createTrackbar('minDisparity','Tune',35,200,nothing)
cv2.createTrackbar('setP1','Tune',0,100,nothing)
cv2.createTrackbar('setP2','Tune',0,100,nothing)
 
while True:
    
  # Read cameras
  retL, imgL= camL.read()
  retR, imgR= camR.read()
   
  # Condition to process when the cameras be captured
  if retL and retR:
    imgRGray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgLGray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

    # Undistortion
    imgLUndistort = cv2.undistort(imgLGray, intrinsicL, distortL, None, update_intrinsicL)
    imgRUndistort = cv2.undistort(imgRGray, intrinsicR, distortR, None, update_intrinsicR)

    tune = cv2.getTrackbarPos('Tune Mode?', 'Tune')
    
    # set parameters 
    if tune:
        # Update parameters based on tuning
        numDisparities = cv2.getTrackbarPos('numDisparities','Tune')*16
        blockSize = cv2.getTrackbarPos('blockSize','Tune')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','Tune')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','Tune')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','Tune')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','Tune')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','Tune')
        speckleRange = cv2.getTrackbarPos('speckleRange','Tune')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','Tune')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','Tune')
        minDisparity = cv2.getTrackbarPos('minDisparity','Tune')
        P1 = cv2.getTrackbarPos('setP1','Tune')
        P2 = cv2.getTrackbarPos('setP2','Tune')
    else:
        # Default setting
        window_size = 3
        minDisparity = 16
        num_disp = 112-minDisparity
        blockSize = 16
        preFilterType = 0
        preFilterSize = 0
        preFilterCap = 0
        textureThreshold = 0
        uniquenessRatio = 10
        speckleRange = 32
        speckleWindowSize = 100
        disp12MaxDiff = 1
        P1 = 8*3*window_size**2
        P2 = 32*3*window_size**2

    # Setting the updated parameters to computing disparity map
    if numDisparities == 0:
        numDisparities = 1
    if preFilterCap == 0:
        preFilterCap = 1
    # set stereo used SGBM
    window_size = 3
    stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        P1 = P1,
        P2 = P2,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )
    # Compute disparity by StereoBM algorithm
    disparity = stereo.compute(imgLUndistort, imgRUndistort)
    # The code produces a single channel image called CV_16S which contains a disparity map scaled by 16 
    # To use it properly, we need to convert it to CV_32F and reduce its scale by 16.
    disparity1 = deepcopy(disparity)
    disparity = disparity.astype(np.float32)
    disparityply = deepcopy(disparity)/16.0
    disparity = (disparity/16.0 - minDisparity)/numDisparities
    ret, contourMask, contourMask2 = warning(disparity1)

    if (ret):
        print("WARNING!!! Object is in range of: ", 'x' ,' cm') # we can modify the value of x according to the requirement
    else:
        pass

    # Show disparity map, image from the cam, and contour map
    cv2.imshow("disp", disparity)
    cv2.imshow("left", imgLUndistort)
    cv2.imshow("right", imgRUndistort)
    cv2.imshow("mask", contourMask)
    cv2.imshow("mask2", contourMask2)
 
    # save ply file with esc key
    if cv2.waitKey(1) == 27: # esc key
        cv2.imwrite('left.jpg', imgLUndistort)
        cv2.imwrite('right.jpg', imgRUndistort)
        cv2.imwrite('disparity.jpg', disparity*255)
        f = 0.8*width                        # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*width],
                    [0,-1, 0,  0.5*height], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
        points = cv2.reprojectImageTo3D(disparityply, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disparityply > disparityply.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = r'C:\Users\nguye\OneDrive - Fulbright University Vietnam\Computer Vision\Assignment 2\out.ply'
        write_ply(out_fn, out_points, out_colors)
        print('%s saved' % out_fn)


    cv2.setMouseCallback('disp', click_event)
  else:
    camL = cv2.VideoCapture(camL_id)
    camR = cv2.VideoCapture(camR_id)