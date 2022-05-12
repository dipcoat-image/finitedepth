========================
Defining substrate class
========================

.. currentmodule:: dipcoatimage.finitedepth

.. plot::
   :context: reset
   :align: center
   :caption: Substrate with circular cross section.

   >>> import cv2
   >>> import matplotlib.pyplot as plt #doctest: +SKIP
   >>> from dipcoatimage.finitedepth import get_samples_path
   >>> path = get_samples_path("ref2.png")
   >>> img = cv2.imread(path)
   >>> plt.imshow(img) #doctest: +SKIP

Substrate class analyzes the substrate geometry from bare substrate image to comprehend the coating layer over it.
For example, :class:`RectSubstrate` detects the vertices of rectangular cross section, allowing the division of coating layer region.

To analyze the substrate with particular geometry, a new class dedicated to it needs to be defined.
DipcoatImage-FiniteDepth provides base class which should be inherited to define new substrate class.

In this guide, a new class :class:`CircularSubstrate` will be defined from abstract base class.
It receives grayscale image and detects the center and radius of the substrate with circular cross section.

Importing parent class
======================

:class:`SubstrateBase` is an abstract base class for all concrete substrate classes.
We import and directly subclass it to show how abstract members are implemented.

.. plot::
   :include-source:
   :context: reset

   >>> from dipcoatimage.finitedepth import SubstrateBase, SubstrateError

Defining parameter class
========================

Additional parameters for the substrate class are passed as dataclass.
This is done by defining a frozen dataclass, and later assigning it to :attr:`Parameters` class attribute of new substrate class.

Here, we define :class:`HoughCirclesParameters` which holds arguments that are passed to :func:`cv2.HoughCircles`.

.. plot::
   :include-source:
   :context: close-figs

   >>> import dataclasses
   >>> @dataclasses.dataclass(frozen=True)
   ... class HoughCirclesParameters:
   ...     method: int
   ...     dp: float
   ...     minDist: float
   ...     param1: float = 100
   ...     param2: float = 100
   ...     minRadius: int = 0
   ...     maxRadius: int = 0

Defining drawing options
========================

Options to visualize the substrate instance are passed as dataclass as well.
This is done by defining a dataclass, and later assigning it to :attr:`DrawOptions` class attribute of new substrate class.

:class:`CircularSubstrate` will draw a circle around the circular substrate for visualization.
We define a dataclass to control its color.

.. plot::
   :include-source:
   :context: close-figs

   >>> @dataclasses.dataclass
   ... class CircleDrawOptions:
   ...     color: tuple = (0, 0, 255)

Defining substrate class
========================

Now, we define :class:`CircularSubstrate` with abstract base class and dataclasses.
Full code will be shown first, and each line will be explained in subsections.

.. plot::
   :include-source:
   :context: close-figs

   >>> import cv2
   >>> import numpy as np
   >>> class CircularSubstrate(SubstrateBase):
   ...     __slots__ = ("_circles",)
   ...     Parameters = HoughCirclesParameters
   ...     DrawOptions = CircleDrawOptions
   ...     def circles(self):
   ...         if not hasattr(self, "_circles"):
   ...             args = dataclasses.asdict(self.parameters)
   ...             circles = cv2.HoughCircles(self.image(), **args)
   ...             if circles is None:
   ...                 circles = np.empty((0, 0, 3))
   ...             self._circles = circles.astype(np.uint16)
   ...         return self._circles
   ...     def examine(self):
   ...         ret = None
   ...         if len(self.circles()) == 0:
   ...             ret = SubstrateError("Hough circle transformation failed.")
   ...         return ret
   ...     def draw(self):
   ...         ret = cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)
   ...         color = self.draw_options.color
   ...         thickness = 3
   ...         for ((x, y, r),) in self.circles()[:1]:
   ...             cv2.circle(ret, (x, y), r, color, thickness)
   ...         return ret

Slots
-----

.. code-block:: python

   __slots__ = ("_circles",)

Substrate class use slots for better performance.
:class:`CircularSubstrate` caches the Hough transformation result by storing it in attribute, so we define slot for it.

Parameters and DrawOptions
--------------------------

.. code-block:: python

   Parameters = HoughCirclesParameters
   DrawOptions = CircleDrawOptions

As described above, we assign our dataclass types to reserved class attributes.

Hough transformation
--------------------

.. code-block:: python

   def circles(self):
       if not hasattr(self, "_circles"):
           args = dataclasses.asdict(self.parameters)
           circles = cv2.HoughCircles(self.image(), **args)
           if circles is None:
               circles = np.empty((0, 0, 3))
           self._circles = circles.astype(np.uint16)
       return self._circles

Once Hough transformation is done, the result is cached to :attr:`_circles` attribute.

Verification
------------

.. code-block:: python

   def examine(self):
       ret = None
       if len(self.circles()) == 0:
           ret = SubstrateError("Hough circle transformation failed.")
       return ret

We implement :meth:`examine` to detect if no circle is found.

Visualization
-------------

.. code-block:: python

   def draw(self):
       ret = cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)
       color = self.draw_options.color
       thickness = 3
       for ((x, y, r),) in self.circles()[:1]:
           cv2.circle(ret, (x, y), r, color, thickness)
       return ret

:meth:`draw` is implemented to visualize the detected circles.
Note that the return image must be in RGB.

Result
======

We first construct a reference with grayscale image.

.. plot::
   :include-source:
   :context: close-figs

   >>> from dipcoatimage.finitedepth import get_samples_path, SubstrateReference
   >>> ref_img = cv2.imread(get_samples_path("ref2.png"), cv2.IMREAD_GRAYSCALE)
   >>> ref = SubstrateReference(ref_img, substrateROI=(400, 100, 1000, 600))
   >>> import matplotlib.pyplot as plt #doctest: +SKIP
   >>> plt.imshow(ref.draw()) #doctest: +SKIP

Then, construct the substrate instance.

.. plot::
   :include-source:
   :context: close-figs

   >>> params = HoughCirclesParameters(cv2.HOUGH_GRADIENT, dp=1, minDist=20,
   ...                                 param1=50, param2=30)
   >>> subst = CircularSubstrate(ref, parameters=params)
   >>> plt.imshow(subst.draw()) #doctest: +SKIP

We can change the option to draw the circle with different color.

.. plot::
   :include-source:
   :context: close-figs

   >>> subst.draw_options.color = (255, 0, 0)
   >>> plt.imshow(subst.draw()) #doctest: +SKIP

We can also verify the instance using either :meth:`valid` or :meth:`verify`.

   >>> params = HoughCirclesParameters(cv2.HOUGH_GRADIENT, dp=1, minDist=20,
   ...                                 param1=5000, param2=3000)
   >>> invalid_subst = CircularSubstrate(ref, parameters=params)
   >>> invalid_subst.valid()
   False
   >>> invalid_subst.verify()
   Traceback (most recent call last):
     ...
   SubstrateError: Hough circle transformation failed.

See Also
========

:doc:`dataclass-design` describes advanced dataclass design for :class:`Parameters` and :class:`DrawOptions`.

:doc:`../explanation/generic-typing` describes how to define robust class with type annoataions.
