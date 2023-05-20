import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from data import *
from Functions import *

image, rows, columns = import_design("blank.bmp")
DMD, colors = make_DMD(image)

######### Variables ###########
###############################
# Don't change these; imports from data.py
R1 = lens['R1']
R2 = lens['R2']
R3 = lens['R3']
tc1 = lens['tc1']
tc2 = lens['tc2']
# You can change these
L = 10              # [mm] distance from DLP to lens
L2 = 19.363             # [mm] distance from lens to object

print('Wavelength Options: 365, 405, 436, 488, 707, 1064 [nm]')
wavelength = int(input('Choose a wavelength: '))

if wavelength == 365:
    nair = n_air['365']
    nlens1 = n_lens1['365']
    nlens2 = n_lens2['365']
elif wavelength == 405:
    nair = n_air['405']
    nlens1 = n_lens1['405']
    nlens2 = n_lens2['405']    
elif wavelength == 436:
    nair = n_air['436']
    nlens1 = n_lens1['436']
    nlens2 = n_lens2['436']
elif wavelength == 488:
    nair = n_air['488']
    nlens1 = n_lens1['488']
    nlens2 = n_lens2['488']    
elif wavelength == 707:
    nair = n_air['707']
    nlens1 = n_lens1['707']
    nlens2 = n_lens2['707']
elif wavelength == 1064:
    nair = n_air['1064']
    nlens1 = n_lens1['1064']
    nlens2 = n_lens2['1064']
else:
    raise NotImplemented


########## BARRIER 1 ##########
###############################

incident_angles, incident_positions, incident_normal = angle_i(DMD, R=R1, L=L)
refracted_angles = snells_law(n1=nair, n2=nlens1, theta1=incident_angles)


########## BARRIER 2 ##########
###############################

z_o = L + tc1 - R2
r_p = r_p_vector(z_o, incident_positions)
theta_p = np.zeros([912, 1140], dtype=np.float64)
for x in range(912):
    for y in range(1140):
        v1 = -incident_normal[x,y,:]
        v2 = r_p[x,y,:]
        theta_p[x,y] = angle_between_two_vec(v1, v2)

theta_r = theta_p + refracted_angles

mag_r_p = np.zeros([912, 1140])
for x in range(912):
    for y in range(1140):
        mag_r_p[x,y] = np.linalg.norm(r_p[x,y,:])

angles_two, ray_lengths = law_of_sine(R2, mag_r_p, theta_r, z_o=L+tc1)

z1  = z(DMD, R=R1, L=10)
x1 = DMD[:,:,0]
y1 = DMD[:,:,1]

positions2 = z_lens(incident_positions, r_p, theta_r, ray_lengths)

x2 = positions2[:,:,0]
y2 = positions2[:,:,1]
z2 = positions2[:,:,2]


########## BARRIER 3 ##########
###############################

incident_positions2 = positions2
incident_angles2, incident_normal2 = angle_i2(incident_positions2, incident_positions)
refracted_angles2 = snells_law(n1=nlens1, n2=nlens2, theta1=incident_angles2)

# z_o = L + tc1 - R1
z_o2 = L + tc1 + tc2 - R3
r_p2 = r_p_vector(z_o2, incident_positions2)
theta_p2 = np.zeros([912, 1140], dtype=np.float64)
for x in range(912):
    for y in range(1140):
        v1 = -incident_normal2[x,y,:]
        v2 = r_p2[x,y,:]
        theta_p2[x,y] = angle_between_two_vec(v1, v2)

theta_r2 = theta_p2 + refracted_angles2

mag_r_p2 = np.zeros([912, 1140])
for x in range(912):
    for y in range(1140):
        mag_r_p2[x,y] = np.linalg.norm(r_p2[x,y,:])

angles_three, ray_lengths2 = law_of_sine(R3, mag_r_p2, theta_r2, z_o=tc1+tc2+L)
positions3 = z_lens(incident_positions2, r_p2, theta_r2, ray_lengths2)

x3 = positions3[:,:,0]
y3 = positions3[:,:,1]
z3 = positions3[:,:,2]


########## FINAL IMAGE ##########
#################################

incident_positions3 = positions3
incident_angles3, incident_normal3 = angle_i2(incident_positions3, incident_positions2)
refracted_angles3 = snells_law(n1=nlens2, n2=nair, theta1=incident_angles3)

# z_o = L + tc1 + tc2 + L2
z_o3 = L + tc1 + tc2 + L2 - 10000
r_p3 = r_p_vector(z_o3, incident_positions3)
theta_p3 = np.zeros([912, 1140], dtype=np.float64)
for x in range(912):
    for y in range(1140):
        v1 = -incident_normal3[x,y,:]
        v2 = r_p3[x,y,:]
        theta_p3[x,y] = angle_between_two_vec(v1, v2)

theta_r3 = theta_p3 + refracted_angles3

mag_r_p3 = np.zeros([912, 1140])
for x in range(912):
    for y in range(1140):
        mag_r_p3[x,y] = np.linalg.norm(r_p3[x,y,:])

angles_four, ray_lengths3 = law_of_sine(10000, mag_r_p3, theta_r3, z_o=L+tc1+tc2+L2)
positions4 = z_lens(incident_positions3, r_p3, theta_r3, ray_lengths3)

x4 = positions4[:,:,0]
y4 = positions4[:,:,1]
z4 = positions4[:,:,2]


########## Calculations ##########
##################################

print(f"Information for L = {L} mm, L2 = {L2} mm, {wavelength} nm\n")

print("-----Points of Interest [mm]-----")
print("      L + tc1 - R2 =", round(L+tc1-R2, 2))
print("           L + tc1 =", L+tc1)
print("     L + tc1 + tc2 =", L+tc1+tc2)
print("            L + R1 =", L+R1)
print("L + tc1 + tc2 + L2 =", L+tc1+tc2+L2)

# Magnification
di = L + tc1/2
do = L2 + tc2/2
hi = DMD[455,1139,1] - DMD[455,0,1]
ho = positions4[455,1139,1] - positions4[455,0,1]

M1, M2, error = magnification(di, do, hi, ho)
print(f"\n-----Magnification-----")
print(f"\tM = -do/di: {round(M1,2)}")
print(f"\t M = hi/ho: {round(M2,2)}")
print(f"    percent error: {round(error,2)} %")

# Chromatic Aberration
abx = positions4[911,569,0] - positions4[455,569,0]
aby = positions4[455,1139,1] - positions4[455,569,1]
ogx = DMD[911,569,0] - DMD[455,569,0]
ogy = DMD[455,1139,1] - DMD[455,569,1]
ratiox = abx/ogx
ratioy = aby/ogy

print(f"\n-----Chromatic aberration {wavelength} nm-----")
print(f"\toriginal x = {round(ogx,3)} mm \tfinal image x = {round(abx,3)} mm")
print(f"\toriginal y = {round(ogy,3)} mm \tfinal image y = {round(aby,3)} mm")
print(f"\tfinal to original ratio in x = {round(ratiox, 3)} y = {round(ratioy, 3)}")

# Distortion Coefficient
kx = (DMD[:,:,0]/(positions4[:,:,0]**3)) - positions4[:,:,0]**(-2)
ky = (DMD[:,:,1]/(positions4[:,:,1]**3)) - positions4[:,:,1]**(-2)

# Theoretical Focal Point
theo_f1 = focal_length(nlens1, R1, -R2, tc1)
theo_f2 = focal_length(nlens2, -R2, -R3, tc2)
theo_df = doublet_focal_length(R1, -R2, -R3, nair, nlens1, nlens2, tc1, tc2)

# Empirical Focal Point
delta_x = positions4[911,569,0] - positions3[911,569,0]
delta_y = positions4[911,569,1] - positions3[911,569,1] # solved for focal point using both 
delta_z = positions4[911,569,2] - positions3[911,569,2] # the x and y values, but they're the
m_xz = delta_x/delta_z                                  # same so there's no need in printing both
m_yz = delta_y/delta_z
b_xz = positions4[911,569,0] - m_xz*positions4[911,569,2]
b_yz = positions4[911,569,1] - m_yz*positions4[911,569,2]
fp_xz = -b_xz/m_xz
fp_yz = -b_yz/m_yz
print(f"\n-----Focal Point-----")
##print(f"m = {m_xz},\tb = {b_xz}")
##print(f"m = {m_yz},\tb = {b_yz}")
##print(f"Focal Point: z = {fp_xz} mm")
print(f" Focal Point at z = {round(fp_yz, 3)} mm")
print(f"\t\tf = {round(fp_yz-25, 3)} mm")
print(f" Theoretical 1: f = {round(theo_f1, 3)} mm")
print(f" Theoretical 2: f = {round(theo_f2, 3)} mm")
print(f" Theoretical D: f = {round(theo_df, 3)} mm")

########## PLOTTING ##########
##############################

# draw each lens arc for reference
yy = np.arange(-12.7, 12.7, 0.01)
zz1 = L + R1 - np.sqrt(R1**2 - yy**2)
zz2 = L + tc1 - R2 + np.sqrt(R2**2 - yy**2)
zz3 = L + tc1 + tc2 - R3 + np.sqrt(R3**2 - yy**2)

### 3D plot of the lens
##plt.figure(1)
##ax = plt.axes(projection="3d")
##ax.scatter3D(x1,y1,z1, color='b')
##ax.scatter3D(x2,y2,z2, color='g')
##ax.scatter3D(x3,y3,z3, color='r')
##ax.scatter3D(x4,y4,z4, color='k')
##ax.set_xlabel("x [mm]")
##ax.set_ylabel("y [mm]")
##ax.set_zlabel("z [mm]")
##plt.title(f"3D View of image projection on lens\n$\lambda$={wavelength}nm")
##ax.set_zlim(0,60)
####ax.set_ylim(-12.7,12.7)
####ax.set_xlim(-12.7,12.7)
##ax.set_ylim(-6,6)
##ax.set_xlim(-6,6)
###ax.view_init(elev=30, azim=60, roll=90)
##plt.savefig(f"3D_{wavelength}nm.png")

# 2D plot of XZ plane
plt.figure(2)
plt.plot(z1, x1, marker='.', markersize=0.25, linestyle='None', color='b')
plt.plot(z2, x2, marker='.', markersize=0.25, linestyle='None', color='g')
plt.plot(z3, x3, marker='.', markersize=0.25, linestyle='None', color='r')
plt.plot(z4, x4, marker='.', markersize=0.25, linestyle='None', color='k')
plt.plot(zz1, yy, 'b', linewidth=0.5, alpha=0.25, label="Lens")
plt.plot(zz2, yy, 'b', linewidth=0.5, alpha=0.25)
plt.plot(zz3, yy, 'b', linewidth=0.5, alpha=0.25)
plt.title(f"XZ Plane\n$\lambda$={wavelength}nm")
plt.xlabel("z [mm]")
plt.ylabel("x [mm]")
plt.xlim([0, 60])
plt.ylim([-12.7, 12.7])
plt.legend()
plt.savefig(f"XZ_Plane_{wavelength}nm.png")

# 2D plot of YZ plane
plt.figure(3)
plt.plot(z1, y1, marker='.', markersize=0.25, linestyle='None', color='b')
plt.plot(z2, y2, marker='.', markersize=0.25, linestyle='None', color='g')
plt.plot(z3, y3, marker='.', markersize=0.25, linestyle='None', color='r')
plt.plot(z4, x4, marker='.', markersize=0.25, linestyle='None', color='k')
plt.plot(zz1, yy, 'b', linewidth=0.5, alpha=0.25, label="Lens")
plt.plot(zz2, yy, 'b', linewidth=0.5, alpha=0.25)
plt.plot(zz3, yy, 'b', linewidth=0.5, alpha=0.25)
plt.xlim([0, 60])
plt.ylim([-12.7, 12.7])
plt.title(f"YZ Plane\n$\lambda$={wavelength}nm")
plt.xlabel("z [mm]")
plt.ylabel("y [mm]")
plt.legend()
plt.savefig(f"YZ_Plane_{wavelength}nm.png")

### 2D plot of the images in XY (not XY plane)
##plt.figure(4)
##plt.plot(x1, y1, marker='.', markersize=0.25, linestyle='None', color='b', label="Lens 1")
##plt.plot(x2, y2, marker='.', markersize=0.25, linestyle='None', color='g', label="Lens 2")
##plt.plot(x3, y3, marker='.', markersize=0.25, linestyle='None', color='r', label="Lens 2")
##plt.title(f"Image projected onto each lens barrier\n$\lambda$={wavelength}nm L = {10} mm")
##plt.xlabel("x [mm]")
##plt.ylabel("y [mm]")
##plt.xlim([-13, 13])
##plt.ylim([-13, 13])
##plt.savefig(f"Image_Projection_for_$lambda$={wavelength}.png")

# 2D Ray tracing of the YZ plane
# Ray 1
p0y = DMD[455,1139,1]
p1y = incident_positions[455,1139,1]
p2y = positions2[455,1139,1]
p3y = positions3[455,1139,1]
p4y = positions4[455,1139,1]
py = np.array([p0y, p1y, p2y, p3y, p4y])
p0z = 0.0
p1z = incident_positions[455,1139,2]
p2z = positions2[455,1139,2]
p3z = positions3[455,1139,2]
p4z = positions4[455,1139,2]
pz = np.array([p0z, p1z, p2z, p3z, p4z])
# Ray 2
p0y2 = DMD[455,1000,1]
p1y2 = incident_positions[455,1000,1]
p2y2 = positions2[455,1000,1]
p3y2 = positions3[455,1000,1]
p4y2 = positions4[455,1000,1]
py2 = np.array([p0y2, p1y2, p2y2, p3y2, p4y2])
p0z2 = 0.0
p1z2 = incident_positions[455,1000,2]
p2z2 = positions2[455,1000,2]
p3z2 = positions3[455,1000,2]
p4z2 = positions4[455,1000,2]
pz2 = np.array([p0z2, p1z2, p2z2, p3z2, p4z2])
# Ray 3
p0y3 = DMD[455,861,1]
p1y3 = incident_positions[455,861,1]
p2y3 = positions2[455,861,1]
p3y3 = positions3[455,861,1]
p4y3 = positions4[455,861,1]
py3 = np.array([p0y3, p1y3, p2y3, p3y3, p4y3])
p0z3 = 0.0
p1z3 = incident_positions[455,861,2]
p2z3 = positions2[455,861,2]
p3z3 = positions3[455,861,2]
p4z3 = positions4[455,861,2]
pz3 = np.array([p0z3, p1z3, p2z3, p3z3, p4z3])

# plot the rays
plt.figure(5)
plt.plot(pz, py, linewidth=0.5, color='m')
plt.plot(pz2, py2, linewidth=0.5, color='m')
plt.plot(pz3, py3, linewidth=0.5, color='m')
plt.plot(zz1, yy, 'b', linewidth=0.5, alpha=0.25)
plt.plot(zz2, yy, 'b', linewidth=0.5, alpha=0.25)
plt.plot(zz3, yy, 'b', linewidth=0.5, alpha=0.25)
plt.vlines(fp_yz, -12.7, 12.7, linestyles='-.', color='k', alpha=0.25, label="Focal Point")
#plt.vlines(fp_xz, -12.7, 12.7, linestyles='--', color='k', alpha=0.25)
plt.xlim([0, 60])
plt.ylim([-12.7, 12.7])
plt.legend()
fp_round = round(fp_yz-25, 2)
plt.title(f"Ray Tracing on YZ Plane\n$\lambda$={wavelength}nm, f={fp_round}mm")
plt.xlabel("z [mm]")
plt.ylabel("y [mm]")
plt.savefig(f"Ray_Traving_YZ_{wavelength}nm.png")

# plotting the before and after image
plotting_image = np.zeros([912, 1140, 5], dtype=object)
plotting_image[:,:,:2] = DMD
plotting_image[:,:,2:] = colors
size = 1

x1 = np.zeros([912,1140], dtype=np.float64)
x2 = np.zeros([912,1140], dtype=np.float64)
y1 = np.zeros([912,1140], dtype=np.float64)
y2 = np.zeros([912,1140], dtype=np.float64)

for x in range(912):
    for y in range(1140):
        if sum(plotting_image[x,y,2:]) == 0:
            x1[x,y] = plotting_image[x,y,0]
            y1[x,y] = plotting_image[x,y,1]
        if sum(plotting_image[x,y,2:]) != 0:
            x2[x,y] = plotting_image[x,y,0]
            y2[x,y] = plotting_image[x,y,1]

plt.figure(6)
plt.scatter(x1,y1,color='k',s=size)
plt.scatter(x2,y2,color='w',s=size)
plt.title('Image Before Distortion\n')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.xlim(-1.5,1.5)
plt.ylim(-0.75,0.75)
plt.savefig(f"Image_before.png")

plotting_image[:,:,:2] = positions4[:,:,:2]

x3 = np.zeros([912,1140], dtype=np.float64)
x4 = np.zeros([912,1140], dtype=np.float64)
y3 = np.zeros([912,1140], dtype=np.float64)
y4 = np.zeros([912,1140], dtype=np.float64)

for x in range(912):
    for y in range(1140):
        if sum(plotting_image[x,y,2:]) == 0:
            x3[x,y] = plotting_image[x,y,0]
            y3[x,y] = plotting_image[x,y,1]
        if sum(plotting_image[x,y,2:]) != 0:
            x4[x,y] = plotting_image[x,y,0]
            y4[x,y] = plotting_image[x,y,1]

plt.figure(7)
plt.scatter(x3,y3,color='k',s=size)
plt.scatter(x4,y4,color='w',s=size)
plt.title(f'Image After Distortion\n$\lambda$={wavelength}nm')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.xlim(-1.5,1.5)
plt.ylim(-0.75,0.75)
plt.savefig(f"Image_after_distortion_{wavelength}nm.png")

plt.show()
