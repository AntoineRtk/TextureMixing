import numpy as np
import matplotlib.pyplot as plt 
import scipy.special as spec
import numpy as np

# Utils
def ft(x):
    return np.fft.fft2(x, axes=(0, 1), norm='ortho')
def ift(x):
    return np.fft.ifft2(x, axes=(0, 1), norm='ortho').real
def iftc(x):
    return np.fft.ifft2(x, axes=(0, 1), norm='ortho')
def shift(x):
    return np.fft.fftshift(x, axes=(0, 1))

# Filters
def low_filter(r):
    return np.complex64((r<=np.pi/4) + (r>np.pi/4)*(r<np.pi/2)*np.cos(np.pi/2*(np.log(4*r/np.pi)/np.log(2))))

def high_filter(r):
    return np.complex64((r>=np.pi/2) + (r>np.pi/4)*(r<np.pi/2)*np.cos(np.pi/2*(np.log(2*r/np.pi)/np.log(2))))

def steer_filter(t,k,n):
    alpha = 2**(n-1)*spec.factorial(n-1)/(n*spec.factorial(2*(n-1)))**(0.5);
    return np.complex64((alpha*np.cos(t-np.pi*k/n)**(n-1))*\
            (np.abs(np.mod(t+np.pi-np.pi*k/n,2*np.pi)-np.pi)<np.pi/2))

def mask(t,t0):
    return np.complex64(2.0*(np.abs(np.mod(t-t0,2*np.pi)-np.pi)<np.pi/2) \
    + 1.0*(np.abs(np.mod(t-t0,2*np.pi)-np.pi)==np.pi/2))


# Pyramid decomposition
def build_pyr(img, Ns, Np, upSamp=0, cplx=1, freq=0):
    """
    Pyramid decomposition
    
    Parameters:
    - img (n x n x (3) numpy.ndarray): The image
    - Ns (int): The number of scales
    - Np (int): The number of orientations
    - upSamp (boolean): Up sample the image
    - cplx (boolean): Return the IFT in complex or real
    - freq (boolean): Return the IFT of the images
    
    Returns:
    - (Np x 1 numpy.ndarray): Low-pass images
    - (2 + Np * Ns x 1 numpy.ndarray): Low-pass and bandpass oriented images
    """
    
    M = img.shape[0]
    N = img.shape[1]
    
    # Fourier transform
    Fimg = ft(img)
    
    # Low-pass and high-pass subbands
    imgL = np.empty(Np+1, dtype=object)
    imgH = np.empty(Np*Ns+2, dtype=object)
    
    # Fourier domain
    Lx = np.concatenate((np.linspace(0,N//2-1,N//2),np.linspace(-N//2,-1,N//2)))
    Ly = np.concatenate((np.linspace(0,M//2-1,M//2),np.linspace(-M//2,-1,M//2)))
    lx = np.concatenate((np.linspace(0,N//4-1,N//4),np.linspace(-N//4,-1,N//4)))
    ly = np.concatenate((np.linspace(0,M//4-1,M//4),np.linspace(-M//4,-1,M//4)))
    
    X,Y = np.meshgrid(Lx,Ly)
    x,y = np.meshgrid(lx,ly)
    
    # Polar coordinates
    R = np.sqrt((X*2*np.pi/N)**2 + (Y*2*np.pi/M)**2)
    R[0,0] = 10**(-16)
    T = np.arctan2(Y, X)
    
    # If this is a 3 channel image
    if Fimg.ndim == 3:
        R = R[:, :, np.newaxis]
        T = T[:, :, np.newaxis]
    
    # Getting low-pass and high-pass filter
    L0 = low_filter(R/2.0)
    H0 = high_filter(R/2.0)
    L = low_filter(R)
    H = high_filter(R)
    
    # Applying the first low-pass filter
    low = L0 * Fimg
    imgL[0] = low
    
    # Applying the first high-pass filter
    imgH[0] = H0 * Fimg
    
    # Building the pyramid
    for i in range(Np):
        # Filtering 
        if i == 0:
            for k in range(Ns):
                imgH[1 + Ns*i + k] = 2.0 * steer_filter(T, k, Ns) * H * low
        else:
            if upSamp == 0:
                # Applying bandpass-oriented filters
                for k in range(Ns):
                    imgH[1 + Ns*i + k] = 2.0 * steer_filter(T, k, Ns) * H * low
            else:
                # Subsampling
                extx = np.int32((Fimg.shape[1] - N) // 2)
                exty = np.int32((Fimg.shape[0] - M) // 2)
                
                # Applying bandpass-oriented filters
                for k in range(Ns):
                    if Fimg.ndim == 3:
                        imgH[1 + Ns*i + k] = 2**i * shift(np.pad(shift(2.0 * steer_filter(T, k, Ns) * H * low),
                                                                      pad_width=((exty, exty), (extx, extx), (0,0)),
                                                                      mode='constant'))
                    else:
                        imgH[1 + Ns*i + k] = 2**i * shift(np.pad(shift(2.0 * steer_filter(T, k, Ns) * H * low),
                                                                      pad_width=((exty, exty), (extx, extx)),
                                                                      mode='constant'))
                        
        # Low pass filter the image for the next scale
        low = L * low
        if upSamp == 0:    
            if i < Np-1:
                # Subsampling
                low = low[np.int64(y), np.int64(x)] / 2.0
            imgL[1+i] = low
        else:
            # Change coordinates
            low = low[np.int64(y), np.int64(x)] / 2.0
            extx = np.int32((Fimg.shape[1] - N//2) // 2)
            exty = np.int32((Fimg.shape[0] - M//2) // 2)
            if Fimg.ndim == 3:
                imgL[1+i] = 2**(i+1) * shift(np.pad(shift(low),
                                                    pad_width=((exty, exty), (extx, extx), (0,0)),
                                                    mode='constant'))
            else:
                imgL[1+i] = 2**(i+1) * shift(np.pad(shift(low),
                                                    pad_width=((exty, exty), (extx, extx)),
                                                    mode='constant'))
        
        M = M // 2
        N = N // 2
    
        # Updating Fourier coordinates
        Lx = np.concatenate((np.linspace(0, N//2-1, N//2), np.linspace(-N//2, -1, N//2)))
        Ly = np.concatenate((np.linspace(0, M//2-1, M//2), np.linspace(-M//2, -1, M//2)))
        lx = np.concatenate((np.linspace(0, N//4-1, N//4), np.linspace(-N//4, -1, N//4)))
        ly = np.concatenate((np.linspace(0, M//4-1, M//4), np.linspace(-M//4, -1, M//4)))
        
        X, Y = np.meshgrid(Lx, Ly)
        x, y = np.meshgrid(lx, ly)
    
        # Polar coordinates
        R = np.sqrt((X*2*np.pi/N)**2 + (Y*2*np.pi/M)**2)
        R[0, 0] = 10**(-16)
        T = np.arctan2(Y, X)
    
    
        if Fimg.ndim == 3:
            R = R[:, :, np.newaxis]
            T = T[:, :, np.newaxis]
    
        # Low-pass and high-pass filter for the next iteration
        L = low_filter(R)
        H = high_filter(R)
    
    # Final low-pass filter image
    imgH[Np*Ns+1] = imgL[Np]
    
    # Inverse Fourier transform
    if freq == 0:
        for i in range(Np+1):
            imgL[i] = ift(imgL[i])
        for i in range(Ns*Np+2): 
            # Transformation inverse complexe si cplx=1
            if cplx == 0:
                imgH[i] = iftc(imgH[i]).real
            else:
                imgH[i] = iftc(imgH[i])
    
    return imgL, imgH

   
# Pyramid reconstruction    
def collapse_pyr(imH, Ns, Np, downSamp=0, freq=0):
    """
    Reconstruct the image from the pyramid representation
    
    Parameters:
    - imH (2 + Np * Ns x 1 numpy.ndarray): Pyramid representation containing high-pass and low-pass images
    - Ns (int): The number of scales
    - Np (int): The number of orientations
    - downSamp (boolean): Down sample the image
    - freq (boolean): Input is in the frequency domain
    
    Returns:
    - (n x n x (3) numpy.ndarray): Reconstructed image
    """
    
    # If the input is in spatial domain, convert to frequency domain
    if freq==0:
        imgH = np.empty(Np*Ns+2, dtype=object)
        for i in range(Ns*Np+2): 
            imgH[i] = ft(imH[i].real)
    else:
        imgH = np.copy(imH)
        
    imgHH = np.copy(imgH)
    
    # Down sample the image
    if downSamp==1:
        N = imgH[0].shape[1]
        M = imgH[0].shape[0]
        for i in range(1,Np):
            extx = np.int32((N-N//2**i)//2)
            exty = np.int32((M-M//2**i)//2)
            for k in range(Ns):
                imgHH[1+Ns*i+k] = shift(shift(imgH[1+Ns*i+k])[exty:exty+M//2**i,extx:extx+N//2**i])/2**i
                
            imgHH[Np*Ns+1] = shift(shift(imgH[Np*Ns+1])[exty:exty+M//2**i,extx:extx+N//2**i])/2**i
        
    N = imgHH[Np*Ns+1].shape[1]
    M = imgHH[Np*Ns+1].shape[0]
    Lx = np.concatenate((np.linspace(0,N//2-1,N//2),np.linspace(-N//2,-1,N//2)))
    Ly = np.concatenate((np.linspace(0,M//2-1,M//2),np.linspace(-M//2,-1,M//2)))
    
    X,Y = np.meshgrid(Lx,Ly)
    
    R = np.sqrt((X*2*np.pi/N)**2+(Y*2*np.pi/M)**2)
    R[0,0] = 10**(-16)
    T = np.arctan2(Y,X)
     
    if imgHH[Np*Ns+1].ndim==3:
        R = R[:,:,np.newaxis]
        T = T[:,:,np.newaxis]

    L = low_filter(R)
    H = high_filter(R)

    imgF = L*imgHH[Np*Ns+1]
    
    # Reconstruct the image
    for i in range(Np):
        for k in range(Ns):
            imgF = imgF + (steer_filter(T+np.pi,Ns-1.0-k,Ns)+steer_filter(T,Ns-1.0-k,Ns))*H*\
            np.fft.fft2(np.real(np.fft.ifft2(imgHH[Np*Ns-Ns*i-k],axes=(0,1),norm='ortho'))\
                        ,axes=(0,1) , norm='ortho')  
            
                         
        if i<Np-1:
            if imgHH[Np*Ns+1].ndim==3:
                imgFF = np.zeros((2*M,2*N,imgF.shape[2]),dtype=np.complex64)
                for j in range(imgF.shape[2]):
                    imgFF[:,:,j] = shift(np.pad(shift(imgF[:,:,j]), ((M//2,M//2),(N//2,N//2)), 'constant' ))
                imgF = imgFF
            else:
                imgF = 2*shift(np.pad(shift(imgF), ((M//2,M//2),(N//2,N//2)), 'constant' ))
            

            N = 2*N
            M = 2*M
    
            Lx = np.concatenate((np.linspace(0,N//2-1,N//2),np.linspace(-N//2,-1,N//2)))
            Ly = np.concatenate((np.linspace(0,M//2-1,M//2),np.linspace(-M//2,-1,M//2)))
            X,Y = np.meshgrid(Lx,Ly)
    
            R = np.sqrt((X*2*np.pi/N)**2+(Y*2*np.pi/M)**2)
            R[0,0] = 10**(-16)
            T = np.arctan2(Y,X)
            
            if imgHH[Np*Ns+1].ndim==3:
                R = R[:,:,np.newaxis]
                T = T[:,:,np.newaxis]
    
            L = low_filter(R)
            H = high_filter(R)
    
            imgF = L*imgF
        
    # Inverse Fourier transform to obtain the reconstructed image
    imgF = ift(low_filter(R/2.0)*imgF + high_filter(R/2.0)*imgHH[0]).real

    return imgF