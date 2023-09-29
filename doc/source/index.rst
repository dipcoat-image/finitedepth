==================================================
Welcome to DipCoatImage-FiniteDepth documentation!
==================================================

.. currentmodule:: dipcoatimage.finitedepth

.. plot::

   import cv2, numpy as np, matplotlib.pyplot as plt
   from dipcoatimage.finitedepth import *
   data = dict(
      ref_path=get_data_path("ref3.png"),
      coat_path=get_data_path("coat3.mp4"),
      reference=dict(
         templateROI=(13, 10, 1246, 200),
         substrateROI=(100, 100, 1200, 500),
      ),
      coatinglayer=dict(
         deco_options=dict(
            layer=dict(
               facecolor=(69, 132, 182),
               linewidth=0,
            )
         )
      )
   )
   config = data_converter.structure(data, Config)
   coat = config.construct_coatinglayer(0, False)
   img = coat.draw()

   img[np.where(img == (0, 0, 0))[:-1]] = (100, 100, 100)

   mask = (coat.image.astype(bool) ^ coat.extract_layer())
   k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
   edge_mask = cv2.erode(mask.astype(np.uint8), k).astype(bool) ^ mask
   img[edge_mask] = (255, 255, 255)

   plt.figure(figsize=(4, 4))
   plt.axis("off")
   plt.imshow(img)
   plt.tight_layout()

DipCoatImage-FiniteDepth is a Python package to visualize and analyze the
coating layer profile in *finite depth dip coating* process.

The advantages of DipCoatImage-FiniteDepth are:

* Provides simple image analysis scheme.
* Good extensibility.
* Written in pure Python.

Contents
========

.. toctree::
   :maxdepth: 2

   intro
   tutorial/index
   guide/index
   reference
   explanation/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
