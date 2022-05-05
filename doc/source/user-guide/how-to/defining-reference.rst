========================
Defining reference class
========================

.. currentmodule:: dipcoatimage.finitedepth.reference

.. plot::
   :context: reset
   :align: center
   :caption: Original reference image (left) and binarized reference image (right).

   import cv2
   import matplotlib.pyplot as plt
   from dipcoatimage.finitedepth import get_samples_path
   gray = cv2.imread(get_samples_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
   _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   _, axs = plt.subplots(1, 2, figsize=(6, 3))
   axs[0].imshow(gray, cmap="gray")
   axs[0].axis("off")
   axs[1].imshow(binary, cmap="gray")
   axs[1].axis("off")
   plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05)
   plt.show()

Reference class provides boilerplate to analyze the coated substrate image, and sometimes you may want to modify it.
DipcoatImage-FiniteDepth provides base class which should be inherited to define new reference class.

In this guide, a new class :class:`BinaryReference` will be defined from abstract base class.
It receives grayscale image and binarizes it to define template image and substrate image.
This is useful when your original image has bad contrast.

Importing parent class
======================

:class:`SubstrateReferenceBase` is an abstract base class for all concrete reference classes.
We import and directly subclass it to show how abstract members are implemented.

.. plot::
   :include-source:
   :context: reset

   >>> from dipcoatimage.finitedepth import SubstrateReferenceBase

Note that it is totally fine (and often better) to inherit already-implemented concrete class, e.g. :class:`SubstrateReference`, instead.

Defining parameter class
========================

Additional parameters for the reference class are passed as dataclass.
This is done by defining a frozen dataclass, and later assigning it to :attr:`Parameters` class attribute of new reference class.

Here, we define :class:`ThresholdParameters` which holds arguments that are passed to :func:`cv2.threshold` for binarization.

.. plot::
   :include-source:
   :context: close-figs

   >>> import dataclasses
   >>> @dataclasses.dataclass(frozen=True)
   ... class ThresholdParameters:
   ...     thresh: int
   ...     maxval: int
   ...     type: int

Defining drawing options
========================

Options to visualize the reference instance are passed as dataclass as well.
This is done by defining a dataclass, and later assigning it to :attr:`DrawOptions` class attribute of new reference class.

We will allow two modes to visualize :class:`BinaryReference` instance - as original grayscale image and as binarized image.
For this, we define :class:`DrawMode` enum and :class:`BinaryDrawOptions` dataclass to enclose it.

.. plot::
   :include-source:
   :context: close-figs

   >>> import enum
   >>> class DrawMode(enum.Enum):
   ...     ORIGINAL = "ORIGINAL"
   ...     BINARY = "BINARY"
   >>> @dataclasses.dataclass
   ... class BinaryDrawOptions:
   ...     mode: DrawMode = DrawMode.ORIGINAL

Defining reference class
========================

Now, we define :class:`BinaryReference` with abstract base class and dataclasses.
Full code will be shown first, and each line will be explained in subsections.

.. plot::
   :include-source:
   :context: close-figs

   >>> import cv2
   >>> class BinaryReference(SubstrateReferenceBase):
   ...     __slots__ = ("_binary",)
   ...     Parameters = ThresholdParameters
   ...     DrawOptions = BinaryDrawOptions
   ...     def binary_image(self):
   ...         if not hasattr(self, "_binary"):
   ...             args = dataclasses.asdict(self.parameters)
   ...             _, self._binary = cv2.threshold(self.image, **args)
   ...         return self._binary
   ...     @property
   ...     def template_image(self):
   ...         x0, y0, x1, y1 = self.templateROI
   ...         return self.binary_image()[y0:y1, x0:x1]
   ...     @property
   ...     def substrate_image(self):
   ...         x0, y0, x1, y1 = self.substrateROI
   ...         return self.binary_image()[y0:y1, x0:x1]
   ...     def examine(self):
   ...         if self.binary_image() is None:
   ...             return TypeError("Binarization failed.")
   ...     def draw(self):
   ...         if self.draw_options.mode == DrawMode.ORIGINAL:
   ...             img = self.image
   ...         else:
   ...             img = self.binary_image()
   ...         return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

Slots
-----

.. code-block:: python

   __slots__ = ("_binary",)

Reference class use slots for better performance.
:class:`BinaryReference` caches the binarized image by storing the result in attribute, so we define slot for it.

Parameters and DrawOptions
--------------------------

.. code-block:: python

   Parameters = ThresholdParameters
   DrawOptions = BinaryDrawOptions

As described above, we assign our dataclass types to reserved class attributes.

Binarization
------------

.. code-block:: python

   def binary_image(self):
       if not hasattr(self, "_binary"):
           args = dataclasses.asdict(self.parameters)
           _, self._binary = cv2.threshold(self.image, **args)
       return self._binary
   @property
   def template_image(self):
       x0, y0, x1, y1 = self.templateROI
       return self.binary_image()[y0:y1, x0:x1]
   @property
   def substrate_image(self):
       x0, y0, x1, y1 = self.substrateROI
       return self.binary_image()[y0:y1, x0:x1]

Once binarization is done, the result is cached to :attr:`_binary` attribute.
Template image and substrate image are cropped from binarized image.

Verification
------------

.. code-block:: python

   def examine(self):
       if self.binary_image() is None:
           return TypeError("Binarization failed.")

:func:`cv2.threshold` returns :obj:`None` if binarization fails.
We implement :meth:`examine` to detect this for verification.

Visualization
-------------

.. code-block:: python

   def draw(self):
       if self.draw_options.mode == DrawMode.ORIGINAL:
           img = self.image
       else:
           img = self.binary_image()
       return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

:meth:`draw` is implemented to check the draw option and choose the image type.
Note that the return image must be in RGB.

Result
======

We construct a reference with Otsu's binarization, and draw with original image.

.. plot::
   :include-source:
   :context: close-figs

   >>> from dipcoatimage.finitedepth import get_samples_path
   >>> ref_img = cv2.imread(get_samples_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
   >>> tempROI = (100, 20, 1150, 160)
   >>> substROI = (50, 10, 1200, 650)
   >>> params = ThresholdParameters(0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   >>> ref = BinaryReference(ref_img, tempROI, substROI, parameters=params)
   >>> import matplotlib.pyplot as plt #doctest: +SKIP
   >>> plt.imshow(ref.draw()) #doctest: +SKIP

Template image is binarized.

.. plot::
   :include-source:
   :context: close-figs

   >>> plt.imshow(ref.template_image, cmap="gray") #doctest: +SKIP

Substrate image is binarized as well.

.. plot::
   :include-source:
   :context: close-figs

   >>> plt.imshow(ref.substrate_image, cmap="gray") #doctest: +SKIP

We can change the option to visualize with binarized image.

.. plot::
   :include-source:
   :context: close-figs

   >>> ref.draw_options.mode = DrawMode.BINARY
   >>> plt.imshow(ref.draw()) #doctest: +SKIP

We can also verify the instance using either :meth:`valid` or :meth:`verify`.

   >>> import numpy as np
   >>> empty_img = np.empty((0, 0), dtype=np.uint8)
   >>> invalid_ref = BinaryReference(empty_img, tempROI, substROI, parameters=params)
   >>> invalid_ref.valid()
   False
   >>> invalid_ref.verify()
   Traceback (most recent call last):
     ...
   TypeError: Binarization failed.

Exercise
========

In this guide, :class:`BinaryReference` does not visualize ROI boxes in order to keep the document simple.
Try implement your own class with this feature.

Hint: subclass :class:`SubstrateReference` with mixing the parameters introduced here.
