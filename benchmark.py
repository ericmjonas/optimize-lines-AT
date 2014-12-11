import numpy as np
import time
import axfunc
import util


sc = util.create_iter_scene_config(512, 512,  50)
ac = util.create_real_atom_config(sc, XGRID_N=400, Y_GRID=400,  ZGRID_N=100)

image = np.random.rand(sc['uy'][2],
                       sc['ux'][2],
                       sc['y'][2],
                       sc['x'][2]).astype(np.float32)

t2 = time.time()
y_nc = axfunc.compute_ATx(ac, sc, image)
t3 = time.time()
print "on the fly", t3-t2
