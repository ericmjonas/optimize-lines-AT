import numpy as np
import numba
from numba import float32

@numba.autojit("f4(f4, f4, i4, f4, f4, f4[:])", nopython=True,
                locals=dict(c=float32, x=float32, v=float32))
def norm_dens_step(xstart, xstep, xn, mu, var, out):
    c = 1.0/np.sqrt(var*2*np.pi)

    x = xstart
    for i in xrange(xn):
        v =  -(x-mu)**2 / (2*var)
        out[i] = c* np.exp(v )
        x += xstep
    return c

@numba.autojit(nopython=True)
def compute_gaussian_mean(pos, lamb, posz, uvar):
    return pos - lamb * posz * uvar

@numba.autojit(nopython=True)
def compute_gaussian_var(z, sigma, sigma_0):
    C = (z * sigma + sigma_0)**2.0
    return C


@numba.autojit(nopython=True)
def compute_scale(posz,
                  sigma, lamb, P,
                  diffusor_depth, sigma_0):
    """
    compute xy location, xy var, and c0 scale factor of gaussian at this ux/uy
    for a point source at posx, posy, posz
    
    """
    
    z = -(posz - diffusor_depth)
    C = 1./(z * sigma + sigma_0)
    b = z**2.0 / (z**2.0 + P)
    c = C**2.0 / (2*np.pi)


    var = compute_gaussian_var(z, sigma, sigma_0) # 1.0/C
    front_of_exp = b * c
    
    c0_scale = front_of_exp * var * 2 * np.pi
    return c0_scale

def get_field(sc, img):
    """
    Use the scene config to reshape the image field to be
    uy, ux : picks an image 
    y, x : an image that can be plotted
    """
    UX_N = sc['ux'][2]
    UY_N = sc['uy'][2]
    X_N = sc['x'][2]
    Y_N = sc['y'][2]


    #b = np.reshape(img, (UY_N, UX_N, Y_N, X_N))
    assert img.shape == (UY_N, UX_N, Y_N, X_N)
    return img


@numba.jit(nopython=True)
def create_uy_grid(ystart, ystep, yn,
                   var, y_grid_points, LAMB, tgt_z, uy,
                   ykern):

    for yg_i in xrange(len(y_grid_points)):
        yg  = y_grid_points[yg_i]
        mu = yg - LAMB * tgt_z * uy
        norm_dens_step(ystart, ystep, yn,
                       mu, var, ykern[yg_i, :] )
        
@numba.jit(nopython=True,  locals=dict(xg=float32, mu=float32, x=float32,
                                       v=float32, c=float32, ux=float32, 
                                       mushift=float32))
def create_ux_grid(uxstart, uxstep, uxn,
                   x_grid_points, LAMB, tgt_z,
                   xstart, xstep, xn, var, xkern):

    c = 1.0/np.sqrt(var*2*np.pi)

    ux = uxstart

    for tgt_ux_i in range(uxn):
        mushift = LAMB * tgt_z * ux
        for xg_i in range(len(x_grid_points)):
            xg = x_grid_points[xg_i]
            
            mu = xg - mushift
            
            x = xstart
            for i in xrange(xn):
                v =  -(x-mu)**2 / (2*var)
                xkern[tgt_ux_i, xg_i, i] = c* np.exp(v )
                x += xstep
        ux += uxstep 

@profile    
def compute_ATx(ac, sc, data):

    x_grid_points = ac['x']
    y_grid_points = ac['y']


    SIGMA = sc['sigma']
    LAMB = sc['lambda']
    PATCH_P = sc['patch_p']
    DIFFUSOR_DEPTH = sc['diffusor_depth']
    SIGMA_0 = sc['sigma_0']
    total_field = np.zeros((len(ac['z']),
                            len(y_grid_points),
                            len(x_grid_points)), dtype=np.float32)

    ykern = np.zeros((len(y_grid_points), sc['y'][2]), dtype=np.float32)

    xkern = np.zeros((sc['ux'][2], len(x_grid_points), sc['x'][2]), dtype=np.float32)
    print xkern.shape, ykern.shape
    yconv = np.zeros((len(y_grid_points), sc['x'][2]), dtype=np.float32)
    xconv = np.zeros((len(y_grid_points), len(x_grid_points)), dtype=np.float32)                    
    for tgt_z_i, tgt_z in enumerate(ac['z']):
        """
        For each z in the atom's z grid
        """
        
        tgt_z = ac['z'][tgt_z_i]

        # the variance of the gaussian only depends on the z, not on
        # any other atoms or properties of the image field
        var = compute_gaussian_var(-(tgt_z - DIFFUSOR_DEPTH), SIGMA, SIGMA_0)

        # comptue the ux set of kernels to apply
        create_ux_grid(sc['ux'][0], sc['ux'][1], sc['ux'][2],
                       x_grid_points, LAMB, tgt_z,
                       sc['x'][0], sc['x'][1], sc['x'][2],
                       var, xkern)
        
                
        for tgt_uy_i in range(sc['uy'][2]):
            uy = sc['uy'][0] + sc['uy'][1] * tgt_uy_i

            # compute the UY set of kernels to apply
            create_uy_grid(sc['y'][0], sc['y'][1], sc['y'][2],
                           var, y_grid_points, LAMB, tgt_z, uy,
                           ykern)
            
            for tgt_ux_i in range(sc['ux'][2]):

                ux = sc['ux'][0] + sc['ux'][1] * tgt_ux_i

                img = get_field(sc, data)[tgt_uy_i, tgt_ux_i]

                yconv = np.dot(ykern, img)
                
                xconv = np.dot(yconv, xkern[tgt_ux_i].T)

                c0 = compute_scale(tgt_z, 
                                   sc['sigma'], sc['lambda'], sc['patch_p'],
                                   sc['diffusor_depth'], sc['sigma_0'])
                total_field[tgt_z_i] += xconv * c0
    return total_field.flatten()

