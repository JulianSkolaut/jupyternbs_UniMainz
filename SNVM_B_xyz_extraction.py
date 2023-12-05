import plot_SNVM_data as pSd
import numpy as np
import matplotlib.pyplot as plt

############################

# contains the following commands:


# expand_Data(data, nr_expansions=1) -> returns data put next to each other in a square of nr_expansions by nr_expansions

# apply_Hann_window(data, metadata,nr_expansions=1,return_freq=False) -> apply 2D Hanning window to data, return_freq additionally returns frequency axis (data is return[0], freq is return[1])

# get_kvectors1D(data, metadata) -> returns 1D k-vectors and additional infos (kx,ky,px_y,px_y,dx,dy) (px is pixel size and dx or dy is length per pixel) based on pixel and actual size of data

# get_Kvectors2D(data, metadata, eps = 0.001) -> returns 2D K-vectors as (Kx,Ky,K) with added eps(ilon) to avoid division by zero

# get_eta(K,dz=0) -> returns eta (see Degen script) with additional z-offset dz

# get_NV_unit_vectors(theta,phi) -> returns unit vectors of NV center (projection between r,theta,phi and x,y,z) as (ex,ey,ez)

# transform_BFT_to_components(Bft,Kx,Ky,K,eta,ex,ey,ez) -> returns stray field components of K-space BFT as (BxFT,ByFT,BzFT,KNV)

# crop_expansion(data_expanded,nr_expansions=1) -> crops images transformed back into real space to original size, depending on nr_expansions

# project_BIP_onto_angle(Bx,By,proj_ang) -> returns stray field projected into IP direction along angle proj_ang 


############################

def expand_Data(data, nr_expansions=1):
    data_expanded = nr_expansions*[data]
    data_expanded = nr_expansions*[data_expanded]
    data_expanded = np.block(data_expanded)
    
    return data_expanded

def apply_Hann_window(data, metadata, nr_expansions=1,return_freq=False):
    size = metadata['rect']['size']
    resolution = metadata['resolution']
    freq_x = np.fft.fftfreq(nr_expansions*resolution[0],d=size[0]/nr_expansions*resolution[0])
    freq_y = np.fft.fftfreq(nr_expansions*resolution[1],d=size[1]/nr_expansions*resolution[1])
    #print(freq)
    window_x = np.hanning(freq_x.size)
    window_y = np.hanning(freq_y.size)
    window_2d = np.sqrt(np.outer(window_y,window_x))
    #print(freq.size,window.size,np.shape(window_2d))
    data_hann = window_2d*data
    
    if return_freq==True:
        return data_hann, freq
    else:
        return data_hann

def get_kvectors1D(data, metadata):
    #define 1D k vectors
    px_x = np.shape(data)[0]
    px_y = np.shape(data)[1]
    actualshape = metadata['rect']['size'] #get shape in m
    #print(actualshape[0]/px_x)
    dx = actualshape[0]/px_x #length per pixel in x dir realspace
    dy = actualshape[1]/px_y #length per pixel in y dir realspace
    if np.mod(px_x,2)==0:
        kx = 2*np.pi*np.linspace(-px_x/2,px_x/2-1,px_x-1)/(dx*px_x)
        # print(kx,np.shape(kx))
    else: 
        kx = 2*np.pi*np.linspace(-(px_x-1)/2,(px_x-1)/2,px_x)/(dx*px_x)
        # print(kx,np.shape(kx))

    if np.mod(px_y,2)==0:
        ky = 2*np.pi*np.linspace(-px_y/2,px_y/2-1,px_y-1)/(dy*px_y)
        # print(ky,np.shape(ky))
    else: 
        ky = 2*np.pi*np.linspace(-(px_y-1)/2,(px_y-1)/2,px_y)/(dy*px_y)
        # print(ky,np.shape(ky))
        
    #kx = np.flip(kx)
    #ky = np.flip(ky)
    
    return kx, ky, px_x, px_y, dx, dy

def get_Kvectors2D(data, metadata, eps = 0.001, actualshape = None):
    
    #define 1D k vectors
    px_x = np.shape(data)[0]
    px_y = np.shape(data)[1]
    if actualshape==None:
        actualshape = metadata['rect']['size'] #get shape in m
    else:
        actualshape = actualshape
    #print(actualshape[0]/px_x)
    dx = actualshape[0]/px_x #length per pixel in x dir realspace
    dy = actualshape[1]/px_y #length per pixel in y dir realspace
    
    ############ deprecated (used in Christian Degen's script for some reason) ################
    if np.mod(px_x,2)==0:
        kx = 2*np.pi*np.linspace(-px_x/2,px_x/2-1,px_x-1)/(dx*px_x)
        # print(kx,np.shape(kx))
    else: 
        kx = 2*np.pi*np.linspace(-(px_x-1)/2,(px_x-1)/2,px_x)/(dx*px_x)
        # print(kx,np.shape(kx))

    if np.mod(px_y,2)==0:
        ky = 2*np.pi*np.linspace(-px_y/2,px_y/2-1,px_y-1)/(dy*px_y)
        # print(ky,np.shape(ky))
    else: 
        ky = 2*np.pi*np.linspace(-(px_y-1)/2,(px_y-1)/2,px_y)/(dy*px_y)
        # print(ky,np.shape(ky))
        
    ###########################################################################################
    
    kx = 2*np.pi*np.linspace(-px_x/2,px_x/2,px_x)/(dx*px_x)
    ky = 2*np.pi*np.linspace(-px_y/2,px_y/2,px_y)/(dy*px_y)
    
    #kx = np.flip(kx)
    #ky = np.flip(ky)
    
    #define 2D k vectors
    # add eps to avoid division by zero
    Kx = np.outer(np.ones(px_y),kx) + eps
    Ky = np.outer(ky,np.ones(px_x)) + eps
    #print(np.shape(kx), np.shape(ky), np.shape(Kx), np.shape(Ky))
    K = np.sqrt(Kx**2+Ky**2)
    #print(Kx,Ky,K,np.shape(K))
    
    return Kx.transpose(), Ky.transpose(), K.transpose()

def get_eta(K,dz=0):
    # dz is an optional shift in z
    eta = np.exp(-K*dz)
    
    return eta

def get_NV_unit_vectors(theta,phi):
    # define probe parameters 
    
    theta_rad = theta*2*np.pi/360
    phi_rad = phi*2*np.pi/360
    
    ex = np.sin(theta_rad)*np.cos(phi_rad)
    ey = np.sin(theta_rad)*np.sin(phi_rad)
    ez = np.cos(theta_rad)

    return ex,ey,ez

def transform_BFT_to_components(Bft,Kx,Ky,K,eta,ex,ey,ez):
    # transform Bft to components 
    KNV = (1j*ex*Kx + 1j*ey*Ky - ez*K)
    Bxft = 1j*(Kx/KNV)*Bft[:,:]*eta
    Byft = 1j*(Ky/KNV)*Bft[:,:]*eta
    Bzft = -(K/KNV)*Bft[:,:]*eta
    
    return Bxft, Byft, Bzft, KNV

def crop_expansion(data_expanded,nr_expansions=1):
    data = np.zeros((int(np.shape(data_expanded)[0]/nr_expansions),int(np.shape(data_expanded)[1]/nr_expansions)))
    data = data_expanded[:int(np.shape(data_expanded)[0]/nr_expansions),:int(np.shape(data_expanded)[1]/nr_expansions)]
    
    return data

def project_BIP_onto_angle(Bx,By,proj_ang):
    proj_ang_rad = proj_ang*2*np.pi/360
    B_IP = np.sqrt(Bx**2+By**2)
    angle_B_IP = np.arctan2(By,Bx)
    angle_B_proj_B_IP = proj_ang_rad-angle_B_IP
    #B_proj = np.sin(angle_B_proj_B_IP)*B_IP
    B_proj = np.sin(angle_B_proj_B_IP)*B_IP
    #print(proj_ang_rad, B_IP, angle_B_IP, angle_B_proj_B_IP)
    

    return B_proj

def extract_Bxyz(data, metadata, theta, phi, eps = 0.001, nr_expansions=1, dz=0, apply_hann=False, actualshape = None):

    data_expanded = expand_Data(data,nr_expansions=nr_expansions)
    
    Bft = np.fft.fftshift(np.fft.fft2(data_expanded))
    
    pSd.PlotData(np.real(Bft), metadata,datalabel='BNV_expanded_FT_real')
    pSd.PlotData(np.imag(Bft), metadata,datalabel='BNV_expanded_FT_imag')
    
    if apply_hann:
        Bft = apply_Hann_window(Bft,metadata,nr_expansions=nr_expansions)
    
    kx, ky, px_x, px_y, dx, dy = get_kvectors1D(Bft,metadata)
    
    if actualshape == None:
        Kx, Ky, K = get_Kvectors2D(Bft,metadata)
    else:
        Kx, Ky, K = get_Kvectors2D(Bft,metadata,actualshape=actualshape)
    
    eta = get_eta(K,dz=dz)
    
    ex,ey,ez = get_NV_unit_vectors(theta,phi)
    
    #Bxft, Byft, Bzft, KNV = transform_BFT_to_components(Bft,Kx,Ky,K,eta,ex,ey,ez)
    Bxft, Byft, Bzft, KNV = transform_BFT_to_components(Bft,Kx,Ky,K,eta,ex,ey,ez)
    
    print(np.min(np.real(Bxft)),np.min(np.real(Byft)),np.min(np.real(Bzft)))
    
    pSd.PlotData(np.real(Bxft), metadata,datalabel='Bx_expanded_FT')
    pSd.PlotData(np.real(Byft), metadata,datalabel='By_expanded_FT')
    pSd.PlotData(np.real(Bzft), metadata,datalabel='Bz_expanded_FT')
    
    Bx_expanded = np.real(np.fft.ifft2(np.fft.ifftshift(Bxft)))
    By_expanded = np.real(np.fft.ifft2(np.fft.ifftshift(Byft)))
    Bz_expanded = np.real(np.fft.ifft2(np.fft.ifftshift(Bzft)))
    
    pSd.PlotData(Bx_expanded, metadata,datalabel='Bx_expanded')
    pSd.PlotData(By_expanded, metadata,datalabel='By_expanded')
    pSd.PlotData(Bz_expanded, metadata,datalabel='Bz_expanded')
    
    Bx = crop_expansion(Bx_expanded,nr_expansions=nr_expansions)
    By = crop_expansion(By_expanded,nr_expansions=nr_expansions)
    Bz = crop_expansion(Bz_expanded,nr_expansions=nr_expansions)
    
    return Bx, By, Bz

def extract_BIP_Projected_onto_Angle(data, metadata, theta, phi, proj_ang, eps = 0.001, nr_expansions=1, dz=0, apply_hann=False):
    
    Bx, By, Bz = extract_Bxyz(data, metadata, theta, phi, eps = eps, 
                              nr_expansions=nr_expansions, dz=dz, apply_hann=apply_hann)
    
    B_proj = project_BIP_onto_angle(Bx,By,proj_ang)
    
    return B_proj
    

def Bxyz_toBNV_toBxyz(datastack, metadata, theta, phi, eps=0.001, nr_expansions=1, dz=0, apply_hann=False, num_output=False, actualshape = None):
    
    data_x = datastack[0]
    data_y = datastack[1]
    data_z = datastack[2]
    
    pSd.PlotData(data_x, metadata,datalabel='data_x', actualshape = actualshape)
    pSd.PlotData(data_y, metadata,datalabel='data_y', actualshape = actualshape)
    pSd.PlotData(data_z, metadata,datalabel='data_z', actualshape = actualshape)
    
    
    ex, ey, ez = get_NV_unit_vectors(theta,phi)
    
    B_NV = data_x*ex + data_y*ey  + data_z*ez
    
    Bx, By, Bz = extract_Bxyz(B_NV, metadata, theta, phi, 
                              eps = eps, nr_expansions=nr_expansions, dz=dz, apply_hann=apply_hann, actualshape = actualshape)

    pSd.PlotData(B_NV, metadata,datalabel='B_NV from datainput', actualshape = actualshape)
    pSd.PlotData(Bx, metadata,datalabel='Bx', actualshape = actualshape)
    pSd.PlotData(By, metadata,datalabel='By', actualshape = actualshape)
    pSd.PlotData(Bz, metadata,datalabel='Bz', actualshape = actualshape)
    
    if num_output==True:
        print('Shape of datastack: ',np.shape(datastack))
        print('NV unit vector components: ', ex,ey,ez)
        print('Means of data_x,y,z: ',np.mean(data_x),np.mean(data_y),np.mean(data_z))
        print('Mean of B_NV from datainput: ',np.mean(B_NV))
        print('Means of B_x,y,z: ',np.mean(Bx),np.mean(By),np.mean(Bz))
        print('data_x:'+' \n',data_x)
        print('data_y:'+' \n',data_y)
        print('data_z:'+' \n',data_z)
        print('B_NV from datainput:'+' \n',B_NV)
        print('Bx:'+' \n',Bx)
        print('By:'+' \n',By)
        print('Bz:'+' \n',Bz)
    
    return(B_NV,Bx,By,Bz)
    







