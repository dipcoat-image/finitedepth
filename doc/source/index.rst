.. DipCoatImage-FiniteDepth documentation master file, created by
   sphinx-quickstart on Sat Mar  2 19:36:13 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DipCoatImage-FiniteDepth's documentation!
====================================================

.. plot::
   :context: reset

   from finitedepth import Reference, Substrate, CoatingLayer, get_sample_path
   import cv2, matplotlib.pyplot as plt
   img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
   _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
   subst = Substrate(ref)
   img = cv2.imread(get_sample_path("coat.png"), cv2.IMREAD_GRAYSCALE)
   _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   coat = CoatingLayer(bin, subst)
   images = (
      ref.draw(templateThickness=2, substrateThickness=2),
      cv2.cvtColor(coat.image, cv2.COLOR_GRAY2RGB),
      coat.draw(layer_color=(69, 132, 182)),
   )
   fig, axes = plt.subplots(1, 3, figsize=(9, 3))
   for img, ax in zip(images, axes):
      for sp in ax.spines:
         ax.spines[sp].set_visible(False)
      ax.xaxis.set_visible(False)
      ax.yaxis.set_visible(False)
      ax.imshow(img)
   fig.tight_layout()

DipCoatImage-FiniteDepth is a Python package to visualize and analyze the
coating layer profile from dip coating with finite depth.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   tutorial
   plugin
   reference/index
   development

Citation
--------

If you use this package in a scientific publication, please cite the following paper::

   @article{song2023measuring,
   title={Measuring coating layer shape in arbitrary geometry},
   author={Song, Jisoo and Yu, Dongkeun and Jo, Euihyun and Nam, Jaewook},
   journal={Physics of Fluids},
   volume={35},
   number={12},
   year={2023},
   publisher={AIP Publishing}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
