.. _caching:

Caching
=======

Caching of methods **MUST** be done in the instance itself, not in
an external container.

This is because external caching increases the reference count
of *self*, preventing it from being garbage collected on time.
Our objects usually consume large memory space because of image
arrays; if the objects stacks up in external cache, the program will
soon crash.

It is generally OK to use external caching in experiment class and analysis
class, as they need relatively small memory space. Of course, this does not
hold if your implementations require large memory space.
