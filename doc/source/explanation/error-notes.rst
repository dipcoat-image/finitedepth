Common errors
=============

Documentation
-------------

``'Sphinx' object has no attribute 'add_javascript'`` error may raise
when building the document. Run ``pip install -U sphinx-tabs`` then retry.

.. notes::

    This error is caused by document packages depending on incompatible
    versions of ``docutils``. Future update can resolve this issue.

Testing
-------

``doc8`` may fail if document is not built, because some files are dynamically
generated when running sphinx. Build the document first and run test later.
