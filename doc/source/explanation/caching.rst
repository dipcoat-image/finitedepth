=======
Caching
=======

When caching the methods of classes which stores image data, do not memoize it
in external container.
It will increase the reference counter of the instance, resulting huge memory
loss. Instead, cache the result in private attributes.

Caching the entire array is generally discouraged. In many cases, array is a
raw data which is processed to return expensive results with small sizes.
A good example is a binary image (raw data) and its contours (processed data).
Cache the processed data, not the raw one.

Abstract base classes cache the method only when its method is used by other
base classes (or itself). API methods that are used by the concrete
implementations are not cached by the base class. If the result needs to be
cached, it should be implemented in the concrete class itself. For example,
:meth:`SubstrateBase.contours` is cached because it is used by
:class:`CoatingLayerBase`. However :meth:`CoatingLayerBase.interfaces` is not
cached because no other base class needs access to it.
