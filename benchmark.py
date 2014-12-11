import numpy as np
import time
import axfunc
import util


#sc = util.create_iter_scene_config(512, 1, 32, 1)
#ac = util.create_real_atom_config(sc, XGRID_N=512, YGRID_N=1,  ZGRID_N=128)

# 1-D Y 47 sec
#sc = util.create_iter_scene_config(1, 400, 1, 50)
#ac = util.create_real_atom_config(sc, XGRID_N=1, YGRID_N=400,  ZGRID_N=400)

# 1-D X 50 sec
sc = util.create_iter_scene_config(400, 1, 400, 1)
ac = util.create_real_atom_config(sc, XGRID_N=400, YGRID_N=1,  ZGRID_N=400)

image = np.random.rand(sc['uy'][2],
                       sc['ux'][2],
                       sc['y'][2],
                       sc['x'][2]).astype(np.float32)

t2 = time.time()
y_nc = axfunc.compute_ATx(ac, sc, image)
t3 = time.time()
print "on the fly", t3-t2
