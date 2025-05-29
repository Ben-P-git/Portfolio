# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:59:49 2025

@author: BenPa

Plate spreading function that calculates a deformation seires across a series of interferograms using the okada dislocation plots. 
model 
Integrated usigng syinterferopy to operate this function

"""
#%% preamble
import numpy as np
import numpy.ma as ma
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
import pickle
import matplotlib.pyplot as plt
import math


import scipy 
from scipy.stats import gaussian_kde 
import numpy.ma as ma

sys.path.append(r"C:\Users\BenPa\Documents\Modules")
sys.path.append(r"C:\Users\BenPa\Documents\syinterferopy")


# link to the module syinterferopy: https://github.com/matthew-gaddes/SyInterferoPy
import syinterferopy
from syinterferopy.syinterferopy import deformation_wrapper

#%% impor the initial data 


with open("C:/Users/BenPa/onedrive#/OneDrive - University of Leeds/final year/IDE/octree/dem_mask_111D.pkl","rb") as f:
    dem = pickle.load(f)
    mask= pickle.load(f)
    lons_mg = pickle.load(f)
    lats_mg= pickle.load(f)
#%% define the initial plate velocity 



def latlon_midpoint_bearing_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Midpoint (simple average for small distances)
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    
    # Bearing calculation
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    bearing_deg = (bearing_deg + 360) % 360  # Normalize to 0-360°
    
    # Distance calculation (Haversine formula)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371  # Earth's radius in km
    distance_km = R * c
    
    return (mid_lat, mid_lon, bearing_deg, distance_km)




femri_lat, femri_lon, bearing_femri,distance_femri = latlon_midpoint_bearing_distance(65.3699963, -16.6220013, 65.6999993, -16.821998)
AS_lat, AS_lon, AS_bearing,distance_AS = latlon_midpoint_bearing_distance(65.0490019, -16.7709978,  65.3699963,-16.6220013)
AN_lat, AN_lon, AN_bearing,distance_AN = latlon_midpoint_bearing_distance(64.5500034, -17.4409985,  65.0490019,-16.7709978)

#%% plot the lines 

plt.plot([ -16.821998, -16.6220013],[65.6999993,65.3699963],c="green")
plt.plot([-16.7709978, -16.6220013],[65.0490019,65.3699963],c="red")
plt.plot([ -16.7709978, -17.4409985],[65.0490019, 64.5500034],c="blue")

#%%

rigid_block=np.zeros(dem.shape)
def latlon_to_cartesian(lat, lon):
    # Approximate conversion (for small regions)
    x = (lon - lons.min()) * 111.32 * np.cos(np.radians(lats.mean()))  # 111.32 km/° at equator
    y = (lat - lats.min()) * 111.32  # 111.32 km/°
    return x, y
def line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)
    return A, B, C
def classify_grid(lats, lons, line1, line2,line3,line1a,line2a,val,bearings,lon="lon"):
    x, y = latlon_to_cartesian(lats, lons)
    X, Y = np.meshgrid(x, y)  # Create 2D grid
    
    # Line 1: A1*X + B1*Y + C1
    A1, B1, C1 = line_equation(*line1)
    sign1 = np.sign(A1 * X + B1 * Y + C1)
    
    # Line 2: A2*X + B2*Y + C2
    A2, B2, C2 = line_equation(*line2)
    sign2 = np.sign(A2 * X + B2 * Y + C2)
    # Line 3: A2*X + B2*Y + C2
    A3, B3, C3 = line_equation(*line3)
    sign3 = np.sign(A3 * X + B3 * Y + C3)
    # horizontal lines to divide up the initial plate lines
    # line 4: X + Y + C2a 
    A2a, B2a, C2a = line_equation(*line2a)
    sign2a = np.sign(A2a * X + B2a * Y + C2a)
    # line 5: X + Y + C2a
    A1a, B1a, C1a = line_equation(*line1a)
    sign1a = np.sign(A1a * X + B1a * Y + C1a)
    
    # Combine classifications 
    regions = np.zeros_like(X, dtype=float)
    
    if lon== "lon":
        #plot the velocity across the longitude rather than the longitude
        regions[(sign1 < 0) & (sign1a > 0)] =val *np.abs(np.sin(np.deg2rad(bearings[0])))
        regions[(sign2 < 0) & (sign2a > 0) & (sign1a < 0)] = val*np.abs(np.sin(np.deg2rad(bearings[1])))
        regions[(sign3 < 0) & (sign2a < 0)] = val*np.abs(np.sin(np.deg2rad(bearings[2])))
        regions[(sign1 > 0) & (sign1a > 0)] = -val *np.abs(np.sin(np.deg2rad(bearings[0])))
        regions[(sign2 > 0) & (sign2a > 0) & (sign1a < 0)] = -val*np.abs(np.sin(np.deg2rad(bearings[1])))
        regions[(sign3 > 0) & (sign2a < 0)] = -val*np.abs(np.sin(np.deg2rad(bearings[2])))
            
    else:
        #velocity along the longitude rather than the latitude
        regions[(sign1 < 0) & (sign1a > 0)] =val *np.abs(np.cos(np.deg2rad(bearings[0])))
        regions[(sign2 < 0) & (sign2a > 0) & (sign1a < 0)] = val*np.abs(np.cos(np.deg2rad(bearings[1])))
        regions[(sign3 < 0) & (sign2a < 0)] = val*np.abs(np.cos(np.deg2rad(bearings[2])))
        regions[(sign1 > 0) & (sign1a > 0)] = -val *np.abs(np.cos(np.deg2rad(bearings[0])))
        regions[(sign2 > 0) & (sign2a > 0) & (sign1a < 0)] = -val*np.abs(np.cos(np.deg2rad(bearings[1])))
        regions[(sign3 > 0) & (sign2a < 0)] = -val*np.abs(np.cos(np.deg2rad(bearings[2])))
          
    return regions
import numpy as np

# Example grid
lats = np.linspace(np.min(lats_mg), np.max(lats_mg), 701)
lons = np.linspace(np.min(lons_mg), np.max(lons_mg), 2001)

# Example lines (lat, lon pairs)
line1 = [(65.3699963, -16.6220013),( 65.6999993, -16.821998)]  # Line 1
line1a = [(65.3699963,-15),(65.3699963,-17)]
line2 = [(65.0490019, -16.7709978),( 65.3699963,-16.6220013)]  # Line 2
line2a = [(65.0490019,-15),(65.0490019,-17)]
line3 = [(64.5500034, -17.4409985),(  65.0490019,-16.7709978)]  # Line 2



# Convert line points to Cartesian
line1_cart = [latlon_to_cartesian(np.array([p[0]]), np.array([p[1]])) for p in line1]
line2_cart = [latlon_to_cartesian(np.array([p[0]]), np.array([p[1]])) for p in line2]
line3_cart = [latlon_to_cartesian(np.array([p[0]]), np.array([p[1]])) for p in line3]
line1a_cart = [latlon_to_cartesian(np.array([p[0]]), np.array([p[1]])) for p in line1a]
line2a_cart = [latlon_to_cartesian(np.array([p[0]]), np.array([p[1]])) for p in line2a]
bearings=[bearing_femri,AS_bearing,AN_bearing]

bearings = [i+90 for i in bearings]
val =-0.0174/2 # the velocity of each value
# Classify the grid
regions_x = classify_grid(lats, lons, line1_cart, line2_cart,line3_cart,line1a_cart,line2a_cart,val,bearings,lon="lon")
regions_y = classify_grid(lats, lons, line1_cart, line2_cart,line3_cart,line1a_cart,line2a_cart,val,bearings,lon="lat")

#plot the longitude model model
plt.imshow(regions_x, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()])
plt.title("longitudinal movement")
plt.colorbar(label='Region')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
#plot the 
plt.figure()
plt.imshow(regions_y, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()])
plt.colorbar(label='Region')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


sat_az = 168.12
sat_inc = 39.1618

#%%

deg2rad=np.pi/180
los_x=-np.cos(sat_az*deg2rad)*np.cos(sat_inc*deg2rad)
los_y=-np.sin(sat_az*deg2rad)*np.cos(sat_inc*deg2rad)


plate_motion = (los_y*regions_y)+(los_x*regions_x)

plt.imshow(plate_motion, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()])
plt.colorbar(label='Region')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



    
#%%
    
FE_t = {"strike": bearing_femri,
               "dip"   : 90,
               "rake":0,
               "width" : 8000,
               "depth":8000,
               "length": distance_femri*1000,
               "slip":0,
               "opening":-0.0174
               }
AN_t = {"strike": AN_bearing,
               "dip"   : 90,
               "rake":0,

               "width" : 5000,
               "depth":5000,
               "length": distance_AN*1000,
               "slip":0,
               "opening":-0.0174
    }
AS_t = {"strike": AS_bearing,
               "dip"   : 90,
               "rake":0,

               "width": 5000,
               "depth":5000,
               "length": distance_AS*1000,
               "slip":0,
               "opening":-0.0174
    }
        



#%%


los_grid_FE_t, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, 
                                                   (femri_lon,femri_lat), 'custom',
                                                   dem,asc_or_desc = 'desc',incidence= 39.1618, **FE_t)


los_grid_AN_t, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, 
                                                   (AN_lon,AN_lat), 'custom',
                                                   dem,asc_or_desc = 'desc',incidence= 39.1618, **AN_t)


los_grid_AS_t, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, 
                                                   (AS_lon,AS_lat), 'custom',
                                                   dem,asc_or_desc = 'desc',incidence= 39.1618, **AS_t)

#%%

# add the grids of backslip models to gether to form the function
backslip = los_grid_FE_t+los_grid_AN_t+los_grid_AS_t
backslip_cent = backslip - np.nanmean(backslip)

plt.figure()
plt.imshow(backslip_cent)
plt.colorbar()

#%%
#create a residual model to show 
final_model = -backslip+np.flipud(plate_motion)
plt.figure()
plt.imshow(final_model)
plt.colorbar()
#%% import the cumulative reconstruction

import pickle
with open("C:/Users/BenPa/onedrive#/OneDrive - University of Leeds/final year/IDE/askja_data/cumulative_recon.pkl","rb") as f:
    cumulative= pickle.load(f)

IC = cumulative[3]
IC=IC/6.07

#%%

backslip_rescale =((final_model - np.min(final_model))/(np.max(final_model)-np.min(final_model)))
combined_rescaled = backslip_rescale-IC
full_one = final_model-IC
norm = np.sqrt((1/len(ma.compressed(full_one)))*np.sum(full_one**2))
#print(f"this is the norm or something maybe idk  {norm}")

fig,axs= plt.subplots(1,3,figsize = (15,20),dpi = 300)
all_densities = np.concatenate([
    final_model,IC
])
vmin, vmax = np.nanmin(all_densities), np.nanmax(all_densities)

im1=axs[0].imshow(final_model,"inferno",vmin=vmin,vmax=vmax)
axs[1].imshow(IC,"inferno",vmin=vmin,vmax=vmax)
axs[2].imshow(full_one,"inferno",vmin=vmin,vmax=vmax)
#set titles
axs[0].set_title("Model")

axs[1].set_title("Data")

axs[2].set_title("Residual")


#set ticks
ytick_pixel_n = np.linspace(0, combined_rescaled.shape[0]-1, 5).astype(int)                             # only plot every 10th?
xtick_pixel_n = np.linspace(0, combined_rescaled.shape[1]-1, 10).astype(int)                             # only plot every 10th?
axs[0].set_xticks(xtick_pixel_n, np.round(lons[xtick_pixel_n],2), rotation = 'vertical')               # update xticks to longitudues, 2dp for clarity.  
axs[1].set_xticks(xtick_pixel_n, np.round(lons[xtick_pixel_n],2), rotation = 'vertical')               # update xticks to longitudues, 2dp for clarity.  
axs[2].set_xticks(xtick_pixel_n, np.round(lons[xtick_pixel_n],2), rotation = 'vertical')               # update xticks to longitudues, 2dp for clarity.  
axs[0].set_ylabel("Latitude (deg)")

axs[1].set_xlabel("Longitude (deg)")
axs[0].set_yticks(ytick_pixel_n, np.round(lats[ytick_pixel_n],2)[::1])    
axs[1].set_yticks([])
axs[2].set_yticks([])
#set colorbar
#fig.tight_layout()
fig.colorbar(im1, ax=axs, orientation='horizontal', 
             pad=0.05, aspect=40, label='Displacement (m/year)')



#%%


density_points = np.vstack([IC.ravel(),backslip_rescale.ravel()])


points =np.random.choice(len(ma.compressed(IC)),size = 100000)

density_points = np.vstack([ma.compressed(IC)[points],ma.compressed(final_model)[points]])
kde = gaussian_kde(density_points)
densities = kde(density_points)

#%% Scatter the density points 
plt.figure()
plt.scatter(density_points[0,:],density_points[1,:],c=densities)
plt.xlabel("Indepent component relative Velocity")
plt.xlabel("Model relative Velocity")

