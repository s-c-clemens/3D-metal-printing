# DLP Lens Correction
# Scott Clemens, Eric Everett
# PHY 432 Final Project

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from data import *

def import_design(file, height=912, width=1140):
    """ This function reads in a 250 px by 250 px monochrome design and
        pastes it onto a blank image, then translates that image into a
        numpy array of color values.
 
    parameters
    file : image file
        image to be converted to array

    returns
    image : numpy array
        the design that you want to print
    rows : int
        number of rows in the image
    columns : int
        number of columns in the image
    """

    design = cv.imread(file) # RGB 8-bit color
    
    rows, columns, _ = design.shape # height, width, number of colors (not needed)
    image = np.zeros((height, width, 3), np.uint8)
    # 1140 px wide; center is at 570-1 = 569
    # 912 px tall; center is at 456-1 = 455
    # 250 px wide/tall; half is 125 px
    # 569 -/+ 125 = 444:694
    # 455 -/+ 125 = 330:580
    image[330:580, 444:694] = design[0:rows, 0:columns]
    
    # Uncomment to display design and image
    #cv.imshow('Design', design)
    #cv.imshow('Image for DMD', image)

    cv.imwrite("Test_DMD.bmp", image)
    
    return image, rows, columns # np.array, int, int


def make_DMD(image, columns=912, rows=1140, x0=5.4, y0=5.4, px=10.8):
    """ This function creates the array of pixels for the DMD

    parameters
    image : numpy array
        the design that you want to print
    columns : int
        number of columns in the DMD
    rows : int
        number of rows in the DMD
    x0 : float
        starting location of first x pixel [um]
    y0 : float
        starting location of first y pixel [um]
    px : float
        pixel hypotenuse length [um]

    returns
    DMD : numpy array
        array of tuples that contain physical x-y pixel locations [um]
    """
    DMD = np.zeros([columns, rows, 2], dtype=np.float64)
    colors = np.zeros([columns, rows, 3], dtype=object)
    x, y = 0, 0
    for x in range(columns):
        
        xloc = x0 + x*px
        
        for y in range(rows):
            color = image[x, y, :]
            if y%2 != 0:
                # if y is odd
                xloc = 2*x0 + x*px
                yloc = y0 + y*y0
            else:
                # if y is even
                yloc = y0 + y*y0
                xloc = x0 + x*px
            
            DMD[x, y] = (xloc, yloc)
            colors[x,y,:] = color

    return DMD, colors # np.array


def angle_between_two_vec(vector1, vector2):
    """ This function computes the angle between two vectors
        from the equation of the dot product

    parameters
    vector1 : array
        vector 1
    vector2 : array
        vector 2

    return
    theta : float
        angle between the two vectors [radians]
    """
    theta = np.arccos( np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)) )
    return theta # float


def snells_law(n1, n2, theta1):
    """ This function computes the refracted angle
        using Snell's Law

    parameters
    n1 : float
        refractive index of medium 1
    n2 : float
        refractive index of medium 2
    theta1 : float [radians]
        angle from normal of medium 1

    return
    theta2 : float [radians]
        angle from normal of medium 2
    """
    #theta2 = (n1/n2) * np.arcsin(theta1)
    theta2 = np.arcsin( (n1/n2) * np.sin(theta1) )
    return theta2 # float [radians]


def z(array, R, L):
    """ This function determines the z-coordinate for a point on a surface of a sphere

    parameters
    array : array-like
        array of physical x-y coordinates of pixels
    R : float
        radius of lens [mm]
    L : float
        distance from lens [mm]

    return
    z : array-like
        z-coordinate of surface point on lens [mm]
    """
    sq = R**2 - array[:,:,0]**2 - array[:,:,1]**2
    z = L - np.sqrt(sq) + R

    return z # np.array [mm]


def normal(array):
    """ This function creates the normal vector array given an array
    
    parameters
    array : array-like
        the coordinates of pixels
    
    returns
    dn : array-like
        the normal of each point that contains each pixel
    """
    dn = 2 * array
    nabs = np.sqrt(dn[:,:,0]**2 + dn[:,:,1]**2 + dn[:,:,2]**2)
    
    for i in range(3):
        dn[:,:,i] /= nabs
    return dn # np.array


def angle_i(array, R, L):
    """ Thifs function determines the incident angle
        that a ray makes with the normal

    parameters
    array : array-like
        the array of pixel locations
    R : float
        radius of sphere 1
    L : float
        distance of sphere 1 from DLP

    returns
    angle : array-like
        array of angles
    n_input :
    """
    
    # convert units and offset for origin
    array /= 1000
    array[:,:,0] -= 4.9248
    array[:,:,1] -= 3.078

    zloc = z(array, R, L)
    
    # compute normal vector at given point
    n_input = np.zeros([912, 1140, 3], dtype=np.float64)
    n_input[:,:,0] = array[:,:,0]
    n_input[:,:,1] = array[:,:,1]
    n_input[:,:,2] = zloc
    n = normal(n_input)

    angle = np.zeros([912, 1140], dtype=np.float64)
    for x in range(912):
        for y in range(1140):
            n_point = n[x,y,:]
            z_point = np.array([0,0,zloc[x,y]])
            angle[x,y] = angle_between_two_vec(n_point, z_point)

    return angle, n_input, n # float


def angle_i2(array2, array1):
    """ This function determines the incident angle
        that a ray makes with the normal

    parameters
    array : array-like
        the array of pixel locations
    R : float
        radius of sphere 1
    L : float
        distance of sphere 1 from DLP

    returns
    angle : array-like
        array of angles
    n_input :
    """
    
    # compute normal vector at given point
    n = normal(array2)
    angle3 = np.zeros([912, 1140], dtype=np.float64)
    
    for x in range(912):
        for y in range(1140):
            n_point = n[x,y,:]
            point = array2[x,y,:] - array1[x,y,:]
            angle3[x,y] = angle_between_two_vec(n_point, point)

    return angle3, n


def r_p_vector(z_o, r1_points):
    """ This function calculates the r_p vector
        which goes from (0, 0, L) to the point at which
        a ray diffracts into lens 1

    parameters
    z_o : array-like
        initial z position
    r1_points : array-like
        the points that are on the surface of lens 1

    return
    r : array-like
        magnitude of 
    """
    z_o_vec = np.zeros_like(r1_points)
    z_o_vec[:,:,2] = z_o
    r = r1_points - z_o_vec
    return r


def law_of_sine(sideA, sideB, thetaA, z_o):
    """ This function computes the angle between the
        ray inside lens 1 and the vector pointing to the
        point of refraction.
        For more information see Step 7 in "Calculating the
        Refraction" on the GitHub wiki here:
        https://github.com/Py4Phy/final-2023-dlp_lens_correction/wiki/Calculating-the-Refraction
    parameters
    sideA : float
        length of side 1 [mm]
    sideB : float
        length of side 2 [mm]
    thetaA : float
        angle opposite side 1 [radians]
    return
    thetaC : float
        angle opposite side 3 [radians]
    sideC : float
        length of side 3 [mm]
    
    """
    sideA = sideA*np.ones_like(sideB)
    thetaB = np.arcsin( (sideB  / sideA) * np.sin(thetaA) )
    thetaCdeg = 180 - np.rad2deg(thetaA) - np.rad2deg(thetaB)
    thetaC = np.deg2rad(thetaCdeg)
    sideC = np.zeros_like(sideA)
    # now that we know 2 sides and 3 angles we can determine the third side
    for x in range(912):
        for y in range(1140):
            if thetaA[x, y] != 0.0:
                sideC[x,y] = sideA[x, y] * ( np.sin(thetaC[x, y]) / np.sin(thetaA[x, y]) )
            else:
                # This is probably the chief ray which is colinear to the optical axis
                # as a result, it will have an indicent angle of 0 (thetaA = 0)
                sideC[x,y] = z_o
    return thetaC, sideC # float [mm]


def cartesian_to_cylindrical(cartesian_positions):
    """ This function converts cartesian positions to cylindrical

    parameters
    cartesian_positions : numpy array
        array of (x, y, z) positions

    return
    cylindrical_positions : numpy array
        array of (r, phi, z) positions
    """
    x = cartesian_positions[:,:,0]
    y = cartesian_positions[:,:,1]

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    cylindrical_positions = np.zeros_like(cartesian_positions)
    cylindrical_positions[:,:,0] = r
    cylindrical_positions[:,:,1] = phi
    cylindrical_positions[:,:,2] = cartesian_positions[:,:,2] # z doesn't change
    
    return cylindrical_positions


def cylindrical_to_cartesian(cylindrical_positions):
    """ This function converts cylindrical positions to cartesian

    parameters
    cylindrical_positions : numpy array
        array of (r, phi, z) positions    

    return
    cartesian_positions : numpy array
        array of (x, y, z) positions
    """
    r = cylindrical_positions[:,:,0]
    phi = cylindrical_positions[:,:,1]

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    cartesian_positions = np.zeros_like(cylindrical_positions)
    cartesian_positions[:,:,0] = x
    cartesian_positions[:,:,1] = y
    cartesian_positions[:,:,2] = cylindrical_positions[:,:,2] # z doesn't change
    
    return cartesian_positions


def z_lens(incident_positions, r_p_array, thetaC, sideC):
    """ This function determines the z-coordinate for a point on a surface of a sphere

    parameters
    incident_positions : numpy array
        incident positions of rays that strike lens 1
    r_p_array : numpy array
        array of vectors that point to the incident positions onto lens 1
    thetaC : numpy array
        array of angles between r_p vector and radius
    sideC : numpy array
        ray lengths inside lens

    return
    positions2 : numpy array
        x, y, z coordinates of surface point on next lens
    """
    # convert incident positions to cylindrical coordinates
    cyl_positions = cartesian_to_cylindrical(incident_positions)
    
    # find the angle between the z-axis and r_p vector
    theta_rp = angle_between_two_vec(r_p_array, [0, 0, 1])

    #theta_ri = np.pi - theta_rp
    theta_rideg = 180 - np.rad2deg(theta_rp) 
    theta_ri = np.deg2rad(theta_rideg)
    
    # calculate the angle that the r vector makes at zloc with
    # thetaC = thetaR
    theta_d = thetaC - theta_ri 
    
    dz = sideC * np.sin(theta_d)
    dr = abs(sideC * np.cos(theta_d))
    
    cyl_positions2 = np.zeros_like(cyl_positions)
    cyl_positions2[:,:,0] = cyl_positions[:,:,0] - dr
    cyl_positions2[:,:,1] = cyl_positions[:,:,1] # phi doesn't change
    cyl_positions2[:,:,2] = cyl_positions[:,:,2] + dz

    # x2, y2 convert back to cartesian
    positions2 = cylindrical_to_cartesian(cyl_positions2)

    return positions2


def magnification(di, do, hi, ho):
    """ This function calculates theoretical magnification

    parameters
    di : float
        distance from DLP to center of lens
    do : float
        distance from lens to projected image
    hi : float
        height of image projected to lens
    ho : float
        height of image emitted through lens
    return
    M1 : float
        magnification using di and do
    M2 : float
        magnification using hi and ho
    error : float
        magnification error between M1 and M2
    """
    # calculate magnification both ways
    M1 = -do/di
    M2 = hi/ho
    error = abs( (M1 - M2)/((M1+M2)/2) )*100
    
    return M1, M2, error


def focal_length(n, R1, R2, d):
    """ This function calculates the theoretical focal length

    parameters
    n : float
        refractive index of the material
    R1 : float
        radius of curvature closest to light source
    R2 : float
        radius of curvature farthest from light source
    d : float
        thickness of lens along optical axis
    """
    f = 1 / ( (n-1)*((1/R1) - (1/R2) + ((n-1)*d / (n*R1*R2))) )
    return f


def doublet_focal_length(R1, R2, R3, n_air, n1, n2, tc1, tc2):
    ### Written by ChatGPT ###
    """Calculate the focal length of a lens doublet.

    Args:
        R1 (float): Radius of curvature of the first lens surface.
        R2 (float): Radius of curvature of the second lens surface.
        R3 (float): Radius of curvature of the third lens surface.
        n1 (float): Refractive index of the first lens material.
        n2 (float): Refractive index of the second lens material.
        tc1 (float): Thickness of the first lens, measured along the optical axis.
        tc2 (float): Thickness of the second lens, measured along the optical axis.

    Returns:
        float: The focal length of the doublet.
    """
    d = tc1 + tc2  # Total thickness of the doublet
    term1 = (n2/n_air - n1/n_air) * ((1/R1) - (1/R2))
    term2 = (n2/n2 - n1/n2) * ((1/R2) - (1/R3))
    term3 = ((n2/n_air - n1/n_air) * (tc1/n2)) / (R1*R2)
    term4 = ((n2/n2 - n1/n2) * (tc2/n1)) / (R2*R3)
    f = 1 / (n2/n_air - (term1 + term2 + term3 + term4))
    return f
