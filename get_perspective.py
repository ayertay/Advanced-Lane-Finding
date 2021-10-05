import cv2
import numpy as np
import math
import scipy.spatial.distance

def get_perspective(img):
    (rows,cols,_) = img.shape
    #image center
    u0 = (cols)/2.0
    v0 = (rows)/2.0
    offset_x = 200
    offset_y = 0
    t_l = [cols*0.425, rows * 0.625]
    t_r = [cols*0.575, rows * 0.625]
    b_l = [cols*0.1, rows*0.9]
    b_r = [cols*0.9, rows*0.9]

    #widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(t_l,t_r)
    w2 = scipy.spatial.distance.euclidean(b_l,b_r)
    h1 = scipy.spatial.distance.euclidean(t_l,b_l)
    h2 = scipy.spatial.distance.euclidean(t_r,b_r)

    w = max(w1,w2)
    h = max(h1,h2)

    #visible aspect ratio
    ar_vis = float(w)/float(h)

    #make numpy arrays and append 1 for linear algebra
    m1 = np.float32((t_l[0],t_l[1],1))
    m2 = np.float32((t_r[0],t_r[1],1))
    m3 = np.float32((b_l[0],b_l[1],1))
    m4 = np.float32((b_r[0],b_r[1],1))

    #calculate the focal distance
    k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
    k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]
    #f = math.sqrt(np.abs( (1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))
    #A = np.array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

    #At = np.transpose(A)
    #Ati = np.linalg.inv(At)
    #Ai = np.linalg.inv(A)


    #calculate the real aspect ratio
    #ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))
    ar_real = math.sqrt((n21**2 + n22**2)/(n31**2 + n32**2))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    src = np.float32([t_l, t_r, b_l, b_r])

    #dst = np.float32([[0, offset_y], 
    #                  [maxWidth - 1, offset_y],
    #                  [0, maxHeight - offset_y],
    #                  [maxWidth - 1, maxHeight - offset_y]])

    #pts1 = np.float32(p)
    pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])

    M = cv2.getPerspectiveTransform(src,pts2)
    Minv = cv2.getPerspectiveTransform(pts2, src)
    warped = cv2.warpPerspective(img, M, (W,H), flags=cv2.INTER_LINEAR)
    return warped, M, Minv