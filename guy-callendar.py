#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: guy-callendar.py
#------------------------------------------------------------------------------
# Version 0.1
# 6 April, 2023
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import nc_time_axis
import cftime
# Plotting libraries:
import matplotlib.pyplot as plt; plt.close('all')
import seaborn as sns; sns.set()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16
glosat_version = 'GloSAT.p04c.EBC.LEKnormals'

temp_pkl = '../glosat-py/OUT/1658-2022-input/df_temp_qc.pkl'
year_start, year_end = 1880, 1935
    
#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def extract_station_timeseries( df_temp, stationcode ):
    
    dx = df_temp[ df_temp.stationcode ==stationcode ].reset_index(drop=True)
    t_yearly = np.arange( dx.year.min(), dx.year.max() + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(dx, how='left', on='year')
    
    # TRIM: to start of Pandas datetime range
            
    df = df[df.year >= 1678].reset_index(drop=True)  
    
    # CONSTRUCT: timeseries
    
    ts_monthly = np.array( df.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    t_monthly = pd.date_range(start=str(df.year.min()), periods=len(ts_monthly), freq='MS')     

    df_station = pd.DataFrame({'Tg':ts_monthly}, index=t_monthly)
    df_station.index.name = 'datetime'
    
    return df_station

#----------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------

def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    return obs_sum / ones_out
 
def update_yearly_mean( ds, var):
    """
    update Pandas dataframe yearly mean with weighted yearly mean
    """

    dt = ds.to_xarray()
    dt = dt.rename( {'index':'time'} )    
    du = weighted_temporal_mean( dt, var ).values
    
    return du

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid defined by WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    
    Input
    -----
    lat: vector or latitudes in degrees      
    
    Output
    ------
    r: vector of radius in meters
    
    '''

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric (see equation 3-110 in WGS84)
    lat = np.deg2rad( lat )
    lat_gc = np.arctan( (1-e2) * np.tan(lat) )

    # radius equation (see equation 3-107 in WGS84)
    r = ( a * (1 - e2)**0.5 ) / ( 1 - (e2 * np.cos(lat_gc)**2) )**0.5 
        
    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell (in meters)
    Based on the function in https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    
    Input 
    -----
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output (Xarray)
    ------
    area: grid-cell area in square-meters with dimensions [lat,lon]    
    """

    xlon, ylat = np.meshgrid( lon, lat )
    R = earth_radius( ylat )
    dlat = np.deg2rad( np.gradient( ylat, axis=0) ) 
    dlon = np.deg2rad( np.gradient( xlon, axis=1) )
    dy = dlat * R
    dx = dlon * R * np.cos( np.deg2rad( ylat ) )
    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=[ "latitude", "longitude" ],
        coords={ "latitude": lat, "longitude": lon },
        attrs={ "long_name": "area_per_pixel", "description": "area per pixel", "units": "m^2",},
    )
    return xda

#------------------------------------------------------------------------------
# LOAD: GloSAT absolute temperature archive in pickled pandas dataframe format
#------------------------------------------------------------------------------

df_temp = pd.read_pickle (temp_pkl, compression='bz2')

#------------------------------------------------------------------------------
# SPLIT: into Tropics and N and S Temperate zones
#------------------------------------------------------------------------------

'''
The north temperate zone extends from the Tropic of Cancer (approximately 23.5° north latitude) 
to the Arctic Circle (approximately 66.5° north latitude). 
The south temperate zone extends from the Tropic of Capricorn (approximately 23.5° south latitude) 
to the Antarctic Circle (at approximately 66.5° south latitude).
'''

df_temp_tropics = df_temp[ (df_temp.stationlat <= 23.5) & (df_temp.stationlat >= -23.5 ) ]
df_temp_temperate_n = df_temp[ (df_temp.stationlat > 23.5) & (df_temp.stationlat <= 66.5 ) ]
df_temp_temperate_s = df_temp[ (df_temp.stationlat < -23.5) & (df_temp.stationlat >= -66.5 ) ]

#------------------------------------------------------------------------------
# MASK: 1880-1930 (inclusive) stations having full records
#------------------------------------------------------------------------------

mask_daterange_tropics = ( df_temp_tropics.year >= year_start ) & ( df_temp_tropics.year <= year_end )
mask_daterange_temperate_n = ( df_temp_temperate_n.year >= year_start ) & ( df_temp_temperate_n.year <= year_end )
mask_daterange_temperate_s = ( df_temp_temperate_s.year >= year_start ) & ( df_temp_temperate_s.year <= year_end )

df_temp_daterange_tropics = df_temp_tropics.copy()[mask_daterange_tropics.values] # --> 361 stations
df_temp_daterange_temperate_n = df_temp_temperate_n.copy()[mask_daterange_temperate_n.values] # --> 3043 stations
df_temp_daterange_temperate_s = df_temp_temperate_s.copy()[mask_daterange_temperate_s.values] # --> 266 stations

mask_normals_tropics = ( df_temp_daterange_tropics.year >= 1900 ) & ( df_temp_daterange_tropics.year <= 1930 )
df_temp_normals_tropics = df_temp_daterange_tropics.copy()[mask_normals_tropics.values]
mask_counts_normals_tropics = df_temp_normals_tropics.groupby('stationcode').count().year >= 30
mask_counts_full_tropics = df_temp_daterange_tropics.groupby('stationcode').count().year >= 50
candidates_normals_tropics = mask_counts_normals_tropics.index[ mask_counts_normals_tropics.values ] # --> 175 stations
candidates_full_tropics = mask_counts_full_tropics.index[ mask_counts_full_tropics.values ] # --> 59 stations

mask_normals_temperate_n = ( df_temp_daterange_temperate_n.year >= 1900 ) & ( df_temp_daterange_temperate_n.year <= 1930 )
df_temp_normals_temperate_n = df_temp_daterange_temperate_n.copy()[mask_normals_temperate_n.values]
mask_counts_normals_temperate_n = df_temp_normals_temperate_n.groupby('stationcode').count().year >= 30
mask_counts_full_temperate_n = df_temp_daterange_temperate_n.groupby('stationcode').count().year >= 50
candidates_normals_temperate_n = mask_counts_normals_temperate_n.index[ mask_counts_normals_temperate_n.values ] # --> 2284 stations
candidates_full_temperate_n = mask_counts_full_temperate_n.index[ mask_counts_full_temperate_n.values ] # --> 859 stations

mask_normals_temperate_s = ( df_temp_daterange_temperate_s.year >= 1900 ) & ( df_temp_daterange_temperate_s.year <= 1930 )
df_temp_normals_temperate_s = df_temp_daterange_temperate_s.copy()[mask_normals_temperate_s.values]
mask_counts_normals_temperate_s = df_temp_normals_temperate_s.groupby('stationcode').count().year >= 30
mask_counts_full_temperate_s = df_temp_daterange_temperate_s.groupby('stationcode').count().year >= 50
candidates_normals_temperate_s = mask_counts_normals_temperate_s.index[ mask_counts_normals_temperate_s.values ] # --> 115 stations
candidates_full_temperate_s = mask_counts_full_temperate_s.index[ mask_counts_full_temperate_s.values ] # --> 58 stations

#------------------------------------------------------------------------------
# DEFINE: list of 1880-1930 stations for each zone
#------------------------------------------------------------------------------

stationcodelist_tropics = np.array( candidates_full_tropics )
stationcodelist_temperate_n = np.array( candidates_full_temperate_n )
stationcodelist_temperate_s = np.array( candidates_full_temperate_s )

#------------------------------------------------------------------------------
# INITIALISE: dataframe with GloSAT date range index for each zone
#------------------------------------------------------------------------------

t_vec = pd.date_range( start=str(year_start), end=str(year_end+1), freq='MS' )[0:-1] 
ts = np.ones( [len(t_vec)] ) * np.nan
#df = pd.DataFrame({'Tg':ts}, index=t_vec)

df_tropics = pd.DataFrame(index=t_vec)
df_temperate_n = pd.DataFrame(index=t_vec)
df_temperate_s = pd.DataFrame(index=t_vec)
df_tropics.index.name = 'datetime'
df_temperate_n.index.name = 'datetime'
df_temperate_s.index.name = 'datetime'

#------------------------------------------------------------------------------
# EXTRACT: CRUTEM station timeseries for each zone
#------------------------------------------------------------------------------

for i in range(len(stationcodelist_tropics)):

    stationcode = stationcodelist_tropics[i]
    df_station = extract_station_timeseries( df_temp_tropics, stationcode ) 

    # ADD: to acquisition dataframe and rename columns

    df_tropics = df_tropics.merge(df_station, how='left', on='datetime'); df_tropics = df_tropics.rename(columns={'Tg':stationcode})

for i in range(len(stationcodelist_temperate_n)):

    stationcode = stationcodelist_temperate_n[i]
    df_station = extract_station_timeseries( df_temp_temperate_n, stationcode ) 

    # ADD: to acquisition dataframe and rename columns

    df_temperate_n = df_temperate_n.merge(df_station, how='left', on='datetime'); df_temperate_n = df_temperate_n.rename(columns={'Tg':stationcode})

for i in range(len(stationcodelist_temperate_s)):

    stationcode = stationcodelist_temperate_s[i]
    df_station = extract_station_timeseries( df_temp_temperate_s, stationcode ) 

    # ADD: to acquisition dataframe and rename columns

    df_temperate_s = df_temperate_s.merge(df_station, how='left', on='datetime'); df_temperate_s = df_temperate_s.rename(columns={'Tg':stationcode})

#------------------------------------------------------------------------------
# COMPUTE: 1900-1930 normals for each zone
#------------------------------------------------------------------------------

dg = []
for i in range(1,13):
    da = df_tropics[ df_tropics.index.month == i ]
    normal = da[ (da.index.year >= 1900) & (da.index.year <= 1930) ].mean()
    dg.append(normal)
dg = np.array( dg )

# MAP: normals onto time vec

dh_tropics = np.tile( dg.T, (year_end-year_start+1) ).T

dg = []
for i in range(1,13):
    da = df_temperate_n[ df_temperate_n.index.month == i ]
    normal = da[ (da.index.year >= 1900) & (da.index.year <= 1930) ].mean()
    dg.append(normal)
dg = np.array( dg )

# MAP: normals onto time vec

dh_temperate_n = np.tile( dg.T, (year_end-year_start+1) ).T

dg = []
for i in range(1,13):
    da = df_temperate_s[ df_temperate_s.index.month == i ]
    normal = da[ (da.index.year >= 1900) & (da.index.year <= 1930) ].mean()
    dg.append(normal)
dg = np.array( dg )

# MAP: normals onto time vec

dh_temperate_s = np.tile( dg.T, (year_end-year_start+1) ).T

#------------------------------------------------------------------------------
# COMPUTE: anomalies for each zone
#------------------------------------------------------------------------------

df_anom_tropics = df_tropics.copy() - dh_tropics
df_anom_temperate_n = df_temperate_n.copy() - dh_temperate_n
df_anom_temperate_s = df_temperate_s.copy() - dh_temperate_s

#------------------------------------------------------------------------------
# COMPUTE: GMST for each zone
#------------------------------------------------------------------------------

df_gmst_tropics = df_anom_tropics.mean(axis=1)
df_gmst_temperate_n = df_anom_temperate_n.mean(axis=1)
df_gmst_temperate_s = df_anom_temperate_s.mean(axis=1)

#------------------------------------------------------------------------------
# COMPUTE: yearly mean
#------------------------------------------------------------------------------
    
df_gmst_tropics_yearly = pd.DataFrame({'gmst':df_gmst_tropics.groupby(df_gmst_tropics.index.year).mean()})
df_gmst_tropics_yearly.index = [ pd.to_datetime( str(df_gmst_tropics_yearly.index[i])+'-01-01', format='%Y-%m-%d' ) for i in range(len(df_gmst_tropics_yearly)) ]

df_gmst_temperate_n_yearly = pd.DataFrame({'gmst':df_gmst_temperate_n.groupby(df_gmst_temperate_n.index.year).mean()})
df_gmst_temperate_n_yearly.index = [ pd.to_datetime( str(df_gmst_temperate_n_yearly.index[i])+'-01-01', format='%Y-%m-%d' ) for i in range(len(df_gmst_temperate_n_yearly)) ]

df_gmst_temperate_s_yearly = pd.DataFrame({'gmst':df_gmst_temperate_s.groupby(df_gmst_temperate_s.index.year).mean()})
df_gmst_temperate_s_yearly.index = [ pd.to_datetime( str(df_gmst_temperate_s_yearly.index[i])+'-01-01', format='%Y-%m-%d' ) for i in range(len(df_gmst_temperate_s_yearly)) ]

#------------------------------------------------------------------------------
# COMPUTE: yearly weighted mean for each zone
#------------------------------------------------------------------------------

#df_gmst_tropics_yearly['gmst'] = update_yearly_mean( df_gmst_tropics_yearly, 'gmst')
#df_gmst_temperate_n_yearly['gmst'] = update_yearly_mean( df_gmst_temperate_n_yearly, 'gmst')
#df_gmst_temperate_s_yearly['gmst'] = update_yearly_mean( df_gmst_temperate_s_yearly, 'gmst')

#------------------------------------------------------------------------------
# COMPUTE: area-weighted mean for each zone
#------------------------------------------------------------------------------

latstep = 0.5
lonstep = 0.5
lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
grid_cell_area = area_grid( lats, lons ) 
total_area = grid_cell_area.sum(['latitude','longitude'])
grid_cell_weights_total = grid_cell_area / total_area

ones = grid_cell_area * 0.0 + 1.0
mask_tropics = ( ones.latitude <= 23.5 ) & ( ones.latitude >= -23.5 )
mask_temperate_n = ( ones.latitude > 23.5 ) & ( ones.latitude <= 66.5 )
mask_temperate_s = ( ones.latitude < -23.5 ) & ( ones.latitude >= -66.5 )

masked_area_tropics = grid_cell_area[mask_tropics].sum(['latitude','longitude'])
masked_area_temperate_n = grid_cell_area[mask_temperate_n].sum(['latitude','longitude'])
masked_area_temperate_s = grid_cell_area[mask_temperate_s].sum(['latitude','longitude'])

grid_cell_weights_masked_tropics = grid_cell_area[mask_tropics] / masked_area_tropics
grid_cell_weights_masked_temperate_n = grid_cell_area[mask_temperate_n] / masked_area_temperate_n
grid_cell_weights_masked_temperate_s = grid_cell_area[mask_temperate_s] / masked_area_temperate_s

masked_fraction_tropics = np.array( masked_area_tropics / total_area )          # = 0.39949754
masked_fraction_temperate_n = np.array( masked_area_temperate_n / total_area )  # = 0.25894444
masked_fraction_temperate_s = np.array( masked_area_temperate_s / total_area )  # = 0.25894444

# CHECK: weight closure condition (sum to 1)

print( 'Sum of weights (lat) =', grid_cell_weights_masked_tropics.values.sum(axis=0).sum() ) # = 1
print( 'Sum of weights (lon) =', grid_cell_weights_masked_tropics.values.sum(axis=1).sum() ) # = 1

df_gmst_tropics_yearly_area_averaged = df_gmst_tropics_yearly.copy()
df_gmst_temperate_n_yearly_area_averaged = df_gmst_temperate_n_yearly.copy()
df_gmst_temperate_s_yearly_area_averaged = df_gmst_temperate_s_yearly.copy()

df_gmst_tropics_yearly_area_averaged['gmst'] = df_gmst_tropics_yearly_area_averaged['gmst'].values * masked_fraction_tropics
df_gmst_temperate_n_yearly_area_averaged['gmst'] = df_gmst_temperate_n_yearly_area_averaged['gmst'].values * masked_fraction_temperate_n
df_gmst_temperate_s_yearly_area_averaged['gmst'] = df_gmst_temperate_s_yearly_area_averaged['gmst'].values * masked_fraction_temperate_s

#------------------------------------------------------------------------------
# COMPUTE: GMST
#------------------------------------------------------------------------------

t_vec_yearly = df_gmst_tropics_yearly.index
gmst_mean = np.sum( [df_gmst_tropics_yearly_area_averaged['gmst'].values, df_gmst_temperate_n_yearly_area_averaged['gmst'].values, df_gmst_temperate_s_yearly_area_averaged['gmst'].values], axis=0 ) / np.sum( [ masked_fraction_tropics + masked_fraction_temperate_n + masked_fraction_temperate_s ] )
df_gmst_yearly = pd.DataFrame({'gmst':gmst_mean}, index=t_vec_yearly)

#------------------------------------------------------------------------------
# Guy Callendar GMST
#------------------------------------------------------------------------------

df_gmst_yearly_callendar = df_gmst_yearly.copy()
callendar = np.array( [  

-5,-10,-4,-10,-15,-19,-11,-9,-18,-9,
-9,-14,-18,-12,-5,-15,-6,-3,-2,0,
4,2,-12,-7,-11,-11,-5,-18,-14,-12,
-5,-3,-11,1,7,6,3,-7,0,-1,
6,15,2,6,2,16,14,14,13,2,
20,23,22,16,30,17    
    
    ] ) / 100.0
df_gmst_yearly_callendar['gmst'] = callendar.ravel()

#------------------------------------------------------------------------------
# PLOT: GMST timeseries
#------------------------------------------------------------------------------

titlestr = 'Guy Callendar (1938) modern reproduction with ' + glosat_version
figstr = 'guy-callendar-gmst.png'

fig, axs = plt.subplots(figsize=(15,10), nrows=5, ncols=1, sharex=True, sharey=True)

axs[0].plot( df_gmst_temperate_n_yearly.index, df_gmst_temperate_n_yearly.rolling(10, center=True).mean(), marker='.', lw=5, color='red', alpha=1, label='N Temperate Zone: ' + str(df_anom_temperate_n.shape[1]) + ' stations (10yr MA)' )
axs[0].fill_between( df_gmst_temperate_n_yearly.index, df_gmst_temperate_n_yearly.rolling(10, center=True).mean()['gmst'].values, y2=0, color='pink')
axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[0].tick_params(labelsize=10)    
axs[1].plot( df_gmst_tropics_yearly.index, df_gmst_tropics_yearly.rolling(10, center=True).mean(), marker='.', lw=5, color='green', alpha=1, label='Tropics: ' + str(df_anom_tropics.shape[1]) + ' stations (10yr MA)' )
axs[1].fill_between( df_gmst_tropics_yearly.index, df_gmst_tropics_yearly.rolling(10, center=True).mean()['gmst'].values, y2=0, color='lightgreen')
axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[1].tick_params(labelsize=10)    
axs[2].plot( df_gmst_temperate_s_yearly.index, df_gmst_temperate_s_yearly.rolling(10, center=True).mean(), marker='.', lw=5, color='blue', alpha=1, label='S Temperate Zone: ' + str(df_anom_temperate_s.shape[1]) + ' stations (10yr MA)' )
axs[2].fill_between( df_gmst_temperate_s_yearly.index, df_gmst_temperate_s_yearly.rolling(10, center=True).mean()['gmst'].values, y2=0, color='lightblue')
axs[2].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[2].tick_params(labelsize=10)    
axs[3].plot( df_gmst_yearly.index, df_gmst_yearly.rolling(10, center=True).mean(), lw=5, color='k', alpha=1, label='GMST (10yr MA): WGS84 area-averaged')
axs[3].plot( df_gmst_yearly_callendar.index, df_gmst_yearly_callendar.rolling(10, center=True).mean(), lw=2, ls='--', color='r', alpha=1, label='Callendar (1938) GMST (10yr MA)')
axs[3].fill_between( df_gmst_yearly.index, df_gmst_yearly.rolling(10, center=True).mean()['gmst'].values, y2=0, color='lightgrey')
axs[3].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[3].tick_params(labelsize=10)    
axs[4].plot( df_gmst_yearly.index, df_gmst_yearly, marker='.', lw=2, color='k', label='GMST (yearly mean)')
axs[4].plot( df_gmst_yearly_callendar.index, df_gmst_yearly_callendar, marker='o', lw=2, ls='--', color='r', label='Callendar (1938)')
axs[4].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[4].tick_params(labelsize=10)    

plt.yticks([-0.4,-0.2,0.0,0.2,0.4])

plt.xlabel('Time', fontsize=fontsize)
fig.supylabel(r'2m Temperature Anomaly (from 1900-1930), $^{\circ}$C', fontsize=fontsize)
fig.suptitle(titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')    

#------------------------------------------------------------------------------
print('** END')

