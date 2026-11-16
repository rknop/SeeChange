********
Overview
********

SeeChange is designed to be used as a pipeline and archiving tool for imaging sky surveys, primarily for the La Silla Schmidt Southern Survey (LS4).

SeeChange consists of a main pipeline that takes raw images and produces a few data products:

* Calibrated images (after preprocessing such as bias, flat, etc).  Encapsulated by the ``Image`` class in ``models/image.py``.

* Source catalogs (after source detection and photometry).  Encapsulated by the ``SourceList`` class in ``models/source_list.py``.

* Point Spread Function (PSF) for the image.  Encapsulated by the ``PSF`` class in ``models/psf.py``.

* Calibrations of the sources to external catalogs: astrometric calibration in the form of a World Coordinate System (WCS) and a photometric calibration in the form of a magnitude zero point (ZP).

* Difference images.  These are also ``Image`` objects, and are tracked in the same table, but they have some addititional fields set.  (This includes, but is not limited to, ``is_sub`` which is true.)

* Source catalogs from the difference images (dubbed "detections" as we are mostly interested in transients); these are also ``SourceList`` objects.

* Cutouts around the sources detected in the difference images, along with the corresponding image cutouts from the reference and the newly acquired images.  Encapsulated by the ``Cutouts`` class in ``models/cutouts.py``.

* Measurements on those cutouts, including the photometric flux, the shapes, and some metrics that indicate if the source is astronomical or an artefact (using analytical cuts).  Encapsulated by the ``MeasurementSet`` and ``Mesurements`` classes in ``models/measurements.py``.

* "Deep Scores", a Real/Bogus score based on a ML algorithm running inferrence on the cutouts (where a high value means it's likely an astrophysical transient, and a low value means it's likely an artifact).  Encapsulated by the ``DeepScoreSet`` and ``DeepScores`` classes in ``models/deepscore.py``.

* Alerts: AVRO alerts generated as the pipeline disocvers transients, and which are sent to external kafka servers.  Becasue these alerts contain information that is already stored in our database, we don't store them explicitly.

Additional pipelines for making bias and flat frames, producing deep coadded references, and performing forced photometry on difference images are described separately.

To get set up with an instance of SeeChange, either to run the pipeline, or to use it to find and examine the data produced preivously by the pipeline, see :doc:`setup`.

To understand the data model and data storage model, see :doc:`data_storage`.

For information about running the pipeline, either to discover new transients, or to access data products produced previously by the pipeline, see :doc:`usage`.

For a more technical description of what the pipeline does, see :doc:`pipeline`.

To read more about developing and contributing to the project, see :doc:`development`.

