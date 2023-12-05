import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import scipy
from matplotlib_scalebar.scalebar import ScaleBar
import background_removal
from skimage.measure import profile_line


#contains the following commands: 

# ReadSNVMfile (filename) -> red file into file, metadata
# ExtractData (file, datatype = 'odmr:mTesla') -> Extract datatype from file
# AdjustDataToQS3 (data) -> adjust directions to fit to QS3 plotting
# PlotData(data, metadata, minval=None,maxval=None,datalabel='',zscaling=1,backgroundcorr=None)
#   -> Plot data, backgroundcorr subtracts mean value and sets min to zero in x or y direction, possible values: 'x', 'y' and None
# MeanCorrY(data) -> subtract mean of data for every y line
# MeanCorrX(data) -> analogous for x dir
# SetMinZero(data) -> subtract minimum from data


def ReadSNVMfile (filename):
    file = h5py.File(filename+'/scalarData.h5') #read file from hdf5
    metadata = json.load(open(filename+'/imageMeta.json')) #read corresponding metadata from json file
    # px_shape = metadata['resolution'] #get shape in pxs
    # actualshape = metadata['rect']['size'] #get shape in m
    # samplename = metadata['sampleName'] #get sample name
    return (file, metadata)

def ExtractData (file, datatype = 'odmr:mTesla'):
    data = file['data'][datatype]
    return data


def AdjustDataToQS3 (data):
    dataadjusted = data[:,::-1].transpose()
    return dataadjusted

def MeanCorrY(data):
    data_corr = np.zeros(np.shape(data))
    for i in range(len(data[0,:])): #y dir
                data_corr[:,i] = data[:,i] - np.nanmean(data[:,i])
    return data_corr

def MeanCorrX(data):
    data_corr = np.zeros(np.shape(data))
    for i in range(len(data[:,0])): #x dir
                data_corr[i,:] = data[i,:] - np.nanmean(data[i,:])
    return data_corr

def SetMinZero(data):
    data_corr = np.zeros(np.shape(data))
    data_min = np.nanmin(data)
    data_corr = data-data_min
    return data_corr

def PlotData (data, metadata, minval=None,maxval=None,datalabel='',zscaling=1,backgroundcorr=None,cmap='magma',actualshape = None):
    
    fig, ax = plt.subplots()
    
    #background correction -> remove mean value
    data_corr = np.zeros(np.shape(data))
    if backgroundcorr is not None:
        if backgroundcorr == 'y':
            data_corr = MeanCorrY(data)
        if backgroundcorr =='x':
            data_corr = MeanCorrX(data)
        #background correction -> subtract minimum
        data_corr = SetMinZero(data_corr)
        data_corr = data_corr*zscaling
    else:
        data_corr = data
        data_corr = data_corr*zscaling
    im = ax.imshow(data_corr,cmap=cmap,vmin=minval,vmax=maxval)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.set_title(samplename)
    
    px_shape = metadata['resolution'] #get shape in pxs
    if actualshape == None:
        actualshape = metadata['rect']['size'] #get shape in m
    else:
        actualshape = actualshape
    scalebar = ScaleBar(actualshape[0]/px_shape[0], length_fraction=0.5, 
                    width_fraction=0.02, box_alpha=0.5,frameon=True,box_color='w',location='upper right')
    ax.add_artist(scalebar)

    cb=fig.colorbar(im)
    cb.set_label(datalabel)
    fig.tight_layout()
    
    return fig, ax, im, cb, scalebar
    
    #plt.show()
    
def ExtractLinecut(data, metadata, x0 ,y0 ,x1 ,y1, num=150, method='map_coordinates', line_width = 1):
    
    x0_linecut,y0_linecut = x0, y0
    x1_linecut,y1_linecut = x1, y1
    
    if method == 'map_coordinates':
        
        i=0

        angle_rad_linecut = -1*np.arctan2((y1_linecut-y0_linecut),(x1_linecut-x0_linecut))
        angle_deg_linecut = angle_rad_linecut*360/(2*np.pi)
        
        x_step_width = np.sin(angle_rad_linecut)
        y_step_width = np.cos(angle_rad_linecut)
        
        x0_linecut_shifted = x0_linecut-(line_width-1)/2.*x_step_width
        y0_linecut_shifted = y0_linecut-(line_width-1)/2.*y_step_width
        x1_linecut_shifted = x1_linecut-(line_width-1)/2.*x_step_width
        y1_linecut_shifted = y1_linecut-(line_width-1)/2.*y_step_width

        z_linecut_array = [[]]*line_width
        
        for linecut_pos in range(line_width):
            
            x0_temp = x0_linecut_shifted+i*x_step_width
            y0_temp = y0_linecut_shifted+i*y_step_width
            x1_temp = x1_linecut_shifted+i*x_step_width
            y1_temp = y1_linecut_shifted+i*y_step_width
            
            x_linecut, y_linecut = np.linspace(x0_temp,x1_temp,num), np.linspace(y0_temp,y1_temp,num)
            
            z_linecut_map_coordinates = scipy.ndimage.map_coordinates(np.transpose(data),np.vstack((x_linecut,y_linecut)))
            z_linecut_array[i] = z_linecut_map_coordinates
            
            i+=1

        #print(np.shape(B_stray_linecut_array))
        z_linecut = np.average(z_linecut_array,axis=0)
        
        return z_linecut, z_linecut_array
    
    if method == 'profile_line':
        z_linecut = profile_line(np.transpose(data),(x0_linecut,y0_linecut),(x1_linecut,y1_linecut), 
                                           linewidth=line_width)
        return z_linecut
    
