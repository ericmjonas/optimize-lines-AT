import numpy as np
import numba
import time

def create_iter_scene_config(X_N=100, Y_N=100, uxn=5) :
    """
    From herbert's notes
    everything in um
    """


    UX_N = uxn
    DATA_Y_POINT = 0.0

    diffusor_depth = 1000.0
    return  {'x' : (-150.0, 300.0/X_N, X_N), 
             'y' : (-150.0, 300.0/Y_N, 1.0, Y_N), 
             'ux' : (-0.18, 0.36/UX_N, UX_N), 
             'uy' : (-0.18, 0.36/UX_N, UX_N), 
             'iter' : True,
             'sigma' : 0.00712,
             'lambda' : 0.5,
             'patch_p' : (0.62 * diffusor_depth) **2,
             'data_y_point' : DATA_Y_POINT, 
             'diffusor_depth' : diffusor_depth,
             'sigma_0' : 1.0} 


def create_real_atom_config(sc, XGRID_N=30, Y_GRID=30, ZGRID_N=20):

    MIN_X = -150
    MAX_X = 150


    MIN_Z = -900
    MAX_Z = 900

    X_GRID = np.linspace(MIN_X, MAX_X, XGRID_N).astype(np.float32)
    Y_GRID = np.linspace(MIN_X, MAX_X, XGRID_N).astype(np.float32)
    Z_GRID = np.linspace(MIN_Z, MAX_Z, ZGRID_N).astype(np.float32)

    return {'x': X_GRID,
            'y' : Y_GRID,
            'z' : Z_GRID}    




@numba.autojit(nopython=True)
def f4_iter(xmin, xstep, xn, 
            ymin, ystep, yn,
            uxmin, uxstep, uxn,
            uymin, uystep, uyn,
            posx, posy, posz,
            sigma, lamb, P,
            diffusor_depth, sigma_0, yout):
    """
    Updated version of the forward model

    posz is as a function of the focal plane, and thus is positive and negative

    """

    z = -(posz - diffusor_depth)
    C = 1./(z * sigma + sigma_0)
    b = z**2.0 / (z**2.0 + P)
    c = C**2.0 / (2*np.pi)
    
    


    pos = 0

    uy = uymin
    for uy_i in xrange(uyn):
        uy_term = posy  - lamb *posz*uy

        ux = uxmin
        for ux_i in xrange(uxn):
            ux_term = posx  -lamb *posz*ux

            y = ymin
            for y_i in xrange(yn):
                av_y = (y  - uy_term)**2

                x = xmin
                for x_i in xrange(xn):

                    av_x = (x  - ux_term)**2
                    t = -(C**2)/2 * (av_x + av_y)

                    yout[pos] = np.exp(t) * b * c


                    pos += 1
                    x += xstep
                    
                y += ystep
            ux += uxstep
        uy += uystep


def get_atom_points(atom_config):
    if 'xyz' in atom_config: # if atoms are flattened
        return atom_config['xyz']
    
    zv, yv, xv = np.meshgrid(atom_config['z'], atom_config['y'], atom_config['x'], indexing='ij')
    return (np.c_[xv.flatten(), yv.flatten(), zv.flatten()]).astype(np.float32)


def render_points_count(sc):
    if 'iter' in sc:
        tot = 1
        for v in ['x', 'y', 'ux', 'uy']:
            tot *= sc[v][2]
        return tot
    else:
        
        return len(sc['x']) * len(sc['y']) * len(sc['ux']) * len(sc['uy'])



def atoms_count(ac):
    if 'xyz' in ac:
        return len(ac['xyz'])
    return len(ac['x']) * len(ac['y']) * len(ac['z'])


def create_atom_matrix_iter(atom_config, scene_config,
                            IM_BATCH=400, verbose=True):
            
    """
    Create the matrix of atoms. 

    FIXME Being smart should instead let us use
    a better solver and explicit computation of the forward model instead
    of explicitly blowing this out. 

    """

    IMAGE_PIX = render_points_count(scene_config)
    ATOM_N = atoms_count(atom_config)

    SIGMA = scene_config['sigma']
    LAMBDA = scene_config['lambda']
    PATCH_P = scene_config['patch_p']
    DIFFUSOR_DEPTH = scene_config['diffusor_depth']
    SIGMA_0 = scene_config['sigma_0']
    print "THE MATRIX WILL BE", (ATOM_N, IMAGE_PIX)
    A2 = np.zeros(shape=(ATOM_N, IMAGE_PIX),
                dtype=np.float32 )


    imdata_batch = np.zeros((IM_BATCH, IMAGE_PIX), dtype=np.float32)
    print "creating buffer of size", IM_BATCH*IMAGE_PIX*8/1e6, "MB"

    #render_points = exp.get_render_points(scene_config).astype(np.float32)
    # assume the transformer has already been set up for this scale of data

    atom_points = get_atom_points(atom_config).astype(np.float32)
    batchpos = 0
    sc = scene_config
    t1 = time.time()
    for pi in xrange(ATOM_N):
        if (pi % IM_BATCH) == 0 and verbose:
            t2 = time.time()
            delta = t2 - t1
            est_remaining = (ATOM_N - pi)*(delta/IM_BATCH)/60
            print "Working on ", pi, "of", ATOM_N, "took", delta, "secs", est_remaining, "mins remaining"

            t1 = time.time()

        f4_iter(sc['x'][0], sc['x'][1], sc['x'][2],
                sc['y'][0], sc['y'][1], sc['y'][2],
                sc['ux'][0], sc['ux'][1], sc['ux'][2],
                sc['uy'][0], sc['uy'][1], sc['uy'][2],
                atom_points[pi][0],
                atom_points[pi][1],
                atom_points[pi][2],
                SIGMA, LAMBDA, PATCH_P,
                DIFFUSOR_DEPTH, SIGMA_0, imdata_batch[batchpos]) 
            
            
        batchpos += 1

        if (batchpos == IM_BATCH) or (pi == (ATOM_N-1)):

            reduced1 = imdata_batch[:batchpos]
            
            A2[pi-batchpos+1:pi+1] = reduced1
            batchpos = 0
        
        
    A2 = A2.T
    print "THE ATOMIC MATRIX SIZE IS", A2.shape
    return A2
