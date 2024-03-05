.. _background:
.. currentmodule:: finitedepth

Background
==========

.. figure:: _static/finite-depth-dip-coating.jpg
   :align: center

   The four stages of dip coating with finite immersion depth.

When the substrate is dipped into a bath with relatively small immersion
depth, the coating layer is confined to a narrow range on the bottom part
of the substrate.
DipCoatImage-FiniteDepth extracts and analyzes the coating layer profile
in this system using image analysis.

Fundamentals
------------

.. plot::
   :context: reset
   :caption: Reference image (left) and target image (right).
      Red box and green box in the reference image are *template region*
      and *substrate region*, respectively.

   from finitedepth import Reference, get_sample_path
   import cv2, matplotlib.pyplot as plt
   img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
   _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
   ref_img = ref.draw(templateThickness=2, substrateThickness=2)
   img = cv2.imread(get_sample_path("coat.png"), cv2.IMREAD_GRAYSCALE)
   _, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   images = (
      ref.draw(templateThickness=2, substrateThickness=2),
      cv2.cvtColor(bin, cv2.COLOR_GRAY2RGB),
   )
   fig, axes = plt.subplots(1, 2, figsize=(8, 4))
   for img, ax in zip(images, axes):
      for sp in ax.spines:
         ax.spines[sp].set_visible(False)
      ax.xaxis.set_visible(False)
      ax.yaxis.set_visible(False)
      ax.imshow(img)
   fig.tight_layout()

Image analysis is performed based on two types of images; *reference image* and
*target image*. Reference image is an image of substrate before coating, and
target image is an image after coating.

The basic scheme of analysis is:

#. Select substrate region and template region from the reference image.
#. Locate the substrate region in the target image by template matching of
   the template region.
#. Remove the substrate pixels from the target image to acquire coating layer
   region.
#. Find the coating layer profile and measure its shape.

For this purpose, DipCoatImage-FiniteDepth defines three types of objects:

- Reference (:class:`ReferenceBase`): Store the reference image and ROIs for
  two regions.
- Substrate (:class:`SubstrateBase`): Analyze the shape of the uncoated
  substrate.
- Coating layer (:class:`CoatingLayerBase`): Acquire the coating layer profile
  and measure its shape using the shape information of the uncoated substrate.

Refer to :ref:`module-reference` for more information about these objects.

See Also
--------

For more detailed explanation, check the paper in :ref:`citation`.
