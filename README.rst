M2Stitch
========

Microscope image tile registration and stage coordinate correction.

Features
--------

M2Stitch is a Python package for registering and aligning multi-tile microscope images. It performs tile registration to update stage coordinates based on image content, without creating a full stitched image.

Key features:

* **Tile Registration**: Aligns overlapping microscope images using phase correlation
* **Stage Coordinate Updates**: Calculates corrected stage positions based on image registration
* **Pixel Size Calculation**: Determines the pixel size in microns by comparing stage and pixel positions
* **Flexible Input**: Supports reading TIFF images and CSV coordinate files
* **Channel Selection**: Can process specific fluorescence channels


Requirements
------------

* Python 3.8+


Installation
------------

You can install *M2Stitch* via pip:

.. code:: console

   $ pip install m2stitch


Usage
-----

Command Line
^^^^^^^^^^^^

Register tiles and update stage coordinates:

.. code:: console

   $ m2stitch /path/to/images /path/to/coordinates.csv

With options:

.. code:: console

   $ m2stitch /path/to/images /path/to/coordinates.csv \
       --output-csv updated_coordinates.csv \
       --channel-pattern "*405_nm_Ex.tiff" \
       --ncc-threshold 0.3

Python API
^^^^^^^^^^

.. code:: python

   from m2stitch import register_and_update_coordinates

   # Register tiles and update coordinates
   updated_coords = register_and_update_coordinates(
       image_directory="/path/to/images",
       csv_path="/path/to/coordinates.csv",
       channel_pattern="*405_nm_Ex.tiff",
       ncc_threshold=0.5
   )


License
-------

Distributed under the terms of the `BSD 3-Clause license`_.

.. _BSD 3-Clause license: https://opensource.org/licenses/BSD-3-Clause


Credits
-------

This program is based on the MIST_ stitching algorithm. The original paper is `here`_.

.. _MIST: https://pages.nist.gov/MIST
.. _here: https://github.com/USNISTGOV/MIST/wiki/assets/mist-algorithm-documentation.pdf
