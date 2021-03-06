
Operator Optimization
---------------------
I have a linear operator that I'd love to optimize via SIMD, memory-alignment, intrinsics, numeric approximation, and the like



Goals
=======
I need this operator, "compute_ATx", to be faster, but I don't have the time to optimize
it myself. I've tried to provide everything needed, including some benchmarking code,
code to validate it against "ground truth", and an overview.

What I'd like is a python-callable function with an identical signature. My target
OS is linux and OSX. I'm currently using (and highly recommend you use) the
Anaconda python distribution (available for free from Continuum). I make extensive
use of their free open-source "numba" jit engine, but it isn't very transparent
or that well-documented, so it's hard to know where I can improve.

Building and interoperating with python is always a nightmare, so I highly
recommend using cython to wrap c code. I'm happy to use a modern GCC (say, 4.9) and
target AVX instructions. Any large dependencies on, say, MKL or OpenCL would
really need to be justified by a significant performance improvement.

I also recommend using the kernprof.py script for line-by-line profiling
at the python level. It's the best thing I've seen thus far.



Code overview
==============


I have a grid of images. The image grid is UY x UX, and each image is YxX. These
parameters are set in the "scene config" dictionary.

Then I have a collection of atoms or test points, set by the "atom
config" dictionary.

Each atom is generating a gaussian in each image. The exact location of the gaussian
in each image is determined by the atom's (x, y, z) in real space, as well
as which image in the UY x UX grid we are rendering.

Now, I have data consisting of an image set. I need to compute the inner
product of "the image set generated by atom i" with the image.

If you're familiar with linear inverse problems, I have a true set of
points x, a linear operator A, and image data y. I am trying to compute x = A^Ty
as part of a much-larger linear inverse systems solver.

The problem is, of course, this is all quite intensive and slow, even using
the python scientific computing libs coupled with the numab JIT engine.

Because my kernel is separable, computing this dot product can be
segmented into the independent X and Y parts, and we can apply them independently.


code
------
axfunc.py: the main functions
util.py : helper functions
validate.py : an example validation script to compare the explicit forward model with the operator
