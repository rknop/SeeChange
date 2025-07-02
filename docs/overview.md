## Overview

SeeChange is designed to be used as a pipeline and archiving tool for imaging sky surveys, primarily for the La Silla Schmidt Southern Survey (LS4).

SeeChange consists of a main pipeline that takes raw images and produces a few data products:
 - Calibrated images (after preprocessing such as bias, flat, etc).

 - Source catalogs (after source detection and photometry).

 - Point Spread Function (PSF) for the image.

 - Calibrations of the sources to external catalogs: astrometric calibration in the form of a World Coordinate System (WCS) and a photometric calibration in the form of a magnitude zero point (ZP).

 - Difference images.

 - Source catalogs from the difference images
   (dubbed "detections" as we are mostly interested in transients).

 - Cutouts around the sources detected in the difference images, along with the corresponding image cutouts from the reference and the newly acquired images.

 - Measurements on those cutouts, including the photometric flux, the shapes, and some metrics that indicate if the source is astronomical or an artefact (using analytical cuts).

Additional pipelines for making bias frames, flat frames, and to produce deep coadded references are described separately.  The coadd pipeline is described in :doc:`references`.

Data is saved into a database, on the local filesystem, and on a data archive (usually this is a remote server).  Saving and loading data from these locations is described in :doc:`data_storage`.  Optionally, alerts may be sent to brokers on the net for newly discovered difference imaging sources that pass configurable cuts.

To read more about developing and contributing to the project, see :doc:`contribution` and :doc:`testing`.  To set up an instance of the pipeline to run on a local machine or on a remote server, see :doc:`setup`.

### Project architecture

To actually set up and run the pipeline, there are a number of things you need in place.  See :doc:`setup`.

The project is organized around two main subdirectories, `models` and `pipeline`.  `models` contains the objects that represent the data products, most of which are mapped (using SQL Alchemy) to a PostgreSQL database. `pipeline` contains the code that runs the analysis and processing required to produce the data products.

Additional folders include:

 - `alembic`: for migrating the database.

 - `data`: for storing local data, either from the tests or from the actual pipeline (not to be confused with the long term storing of data on the "archive").  (You probably don't really want to store data here, as your code is likely to be checked out on a filesystem that's different from the most efficient large-file storage system on your machine.  The actual data storage location is configured in a YAML file.)

 - `devshell`: docker definitions for running the pipeline in a local dockerized environment.

 - `docs`: documentation.

 - `extern`: external packages that are used by SeeChange, including the `nersc-upload-connector` package that is used to connect the archive.

 - `improc`: image processing code that is used by the pipeline, generally manipulating images in ways that are not specific to a single point in the pipeline (e.g., image alignment or inpainting).  (There is some scope seepage between `improc` and `pipeline`.)

 - `tests`: tests for the pipeline (more on that below).

 - `utils`: generic utility functions that are used by the pipeline.

 - `webap`: A web application for managing the pipeline (the "conductor" part of the webap) and browsing what exposures, images, and detections are in the database.

The actual source code for the pipeline is found in `pipeline`, `models`, `improc`, and `utils`.  Notable files in the `pipeline` folder include:

 - `data_store.py` (described below)

 - `top_level.py` : defines the Pipeline object, which is the object that runs the pipeline on a single image (i.e. a single chip of an exposure)

 - `exposure_processor.py` : The command-line utility to run to run the pipline for an entire exposure (i.e. all chips).  However, this depends on the exposure being known by the Conductor.  (TODO: document.)

 - `pipeline_exposure_launcher.py` : A command-line utility that sits and asks the conductor for exposures to process.  When the conductor assigns one, it uses the code in `exposure_processor.py` to process it.

Notable files in the `models` directory innclude:

 - `base.py` : contains the basic layer of the databse definition, as well as tools for database communication (e.g. `Psycopg2Connection` and `SmartSession`), along with several useful mixin classes.

 - `instrument.py` : a base class for defining instruments.  As an example of a fully defined instrument, see `decam.py`

### Pipeline segments (processes)

The pipeline is broken into several "processes" that take one data product and produce the next.  The idea is that the pipeline can start and stop at any of these junctions and can still be started up from that point with the existing data products; `top_level.py` is implemented this way.  Here is a list of the processes and their data products (including the object classes that track them):

 - **preprocessing**: dark, bias, flat, fringe corrections, etc. For large, segmented focal planes,
   will also segment the input raw data into "sections" that usually correspond to individual CCDs.
   This process takes an `Exposure` object and produces `Image` objects, one for each section/CCD.

 - **extraction**: find the sources in the pre-processed image, measure their PSF, cross-match them
   for astrometric and photometric calibration.
   This process takes an `Image` object and produces a `SourceList` and a 'PSF'.

 - **astrocal**: create a solution from pixel position to RA/Dec on the sky by matching objects in the image to the Gaia DR3 catalog.  This process takes a `SourceList` and produces a `WorldCoordinates.`

 - **photocal**: Use `SourceList` and the Gaia DR3 catalog to find a zeropoint `zp` defined so that `m=-2.5*log10(flux)+zp`, where `flux` is the full-psf flux of a star in ADU in the image.  As this is a discovery pipeline, not a lightcurve-building pipeline, we don't do really careful photmetric calibration, and you shouldn't expect this to be better than a couple of percent.  (In particular, it doesn't try to do anything to deal with objects having different SED from the effective average SED used to find the zeropoint.  It also doesn't handle any spatial variation of the zeropoint on the image (though ideally preprocessing would have taken that out).)  Produces a `ZeroPoint` object.

 - **subtraction**: taking a reference image of the same part of the sky (usually a deep coadd) and subtracting it from the "new" image (the one being processed by the pipeline).  Different algorithms can be used to match the PSFs of the new and reference image (we currently implement ZOGY and Alard/Lupton, and hope to add SFFT later).  Uses an `Image`,`SourceList`, `WorldCoordinates`, and `ZeroPoint` object for the "new" (also called "search" or "science") image, and a second set of the same types of objects for the "ref" (also called "reference" or "template") image, all of which are identified by a `Reference` object.  It produdes new `Image`, `WorldCoordinates`, and `ZeroPoint` objects for the difference image.  (By default, the `WorldCoordinates` and `ZeroPoint` are just inherited directly from the science image, as the reference image is warped to the science image and the difference image is scaled to have the same zeropoint as the science image.)

 - **detection**: finding the sources in the difference image.  This process uses the difference `Image` object and produces a `SourceList` object.  This new source list is different from the previous one, as it contains information only on variable or transient sources, and does not map the constant sources in the field.

 - **cutting**: this part of the pipeline identifies the area of the image around each source that was detected in the previous process step, and the corresponding area in the reference and the new image, and saves the list of those stamps as a `Cutouts` object.  There are three stamps for each detection: new, ref, and sub.

 - **measuring**: this part of the pipeline measures the fluxes and shapes of the sources in the cutouts. It uses a set of analytical cuts to distinguish between astronomical sources and artefacts.  This process uses `Cutouts` object to produce a `MeasurementSet` object and an associated set of `Measurements` objects, one for each source.  Measurements that pass cuts are saved to the database.  (There are two different configurable thresholds, one for which measurements are saved, one for which measurements are not marked as bad.  If the two threshold sets are the same, then there will be no measurements marked bad that are saved.)  Measurements are linked to existing Objects; if no Object is close enough, a new Object is created.

 - **scoring**: this part of the pipeline assigns to each measurement a deep learning/machine learning score based on a given algorithm and parameters. In addition to a column for the score, the resulting deepscores contain a JSONB column which can contain additional relevant information.  This process uses the list of `Measurements` objects to product a `DeepScoreSet` object and an associated list of `Deepscore` objects, one for each measurement.

 - **alerting**: send out alerts on measurements that passed the analytical cuts, and that have a configurable minimum deepscore from the scoring step.

More details on each step in the pipeline and their parameters can be found in `docs/pipeline.md`.


### The `DataStore` object

Every time we run the pipeline, objects need to be generated and pulled from the database.  The `DataStore` object is generated by the first process that is called in the pipeline, and is then passed from one process to another.  The datastore contains all the data products for the "new" image being processed, and also data products for the reference image (that are produced earlier by the reference pipeline).  DataStore is defined in `pipeline/data_store.py`.

(TODO: document uses of the datastore for finding things?  It's not really that convenient given provenances; the webap will be a far easier way to do it.  Perhaps we should implement something that lets a DataStore load its provenance tree from a provtag, and then come back and add back documentation here about using DataStore to find things— Issue #507.)

### Configuring the pipeline

Each part of the pipeline (each process) is conducted using a dedicated object.  In practice, you will not instantiate and run these objects individually, but would rather instantiate a single `Pipeline` object (defined in `pipeline/top_level.py`).  That `Pipeline` object then creates all of hte individual process objects described below, which are stored as attributes of the `Pipeline` object.  (In real practice, you won't even directly instantiate a `Pipeline` object, but you will just run something like `exposure_processor.py` or `pipeline_exposure_launcher.py`.  However, if you're writing something lower level, you might use a `Pipeline` object directly.)

There are a few ways to configure any object in the pipeline.  By default, the pipeline will use the config system defined in `util/config.py` to read a configuration YAML file.  An example file with documentation in comments (and the default location of this file) can be found as `default_config.yaml` in the directory of the project.  The `SEECHANGE_CONFIG` environment variable allows you to change the location of the config file that will be read when you run the code.  (We recommend that you don't actually edit the `default_config.yaml` file directly.  Either copy it and edit it to make your own config, or, better, leave it as is, but then create your own `local_overrides.yaml` file to update any config options that need to be modified from what's in the default file.)  The list below shows the blocks that can be found in the config yaml file that define the parameters for a given process.

You can also configure all the various process objects of a `Pipeline` by passing arguments to the `Pipeline` constructor.  The `Pipeline` constructor takes each of the following keyword arguments; the value must be a dictionary that configures the appropriate kind of object.  Each of the keys below also represents a block found in the config yaml file, which sets default values that override the default values in the python code.  For regular usage, you probably do not want to configure the pipeline by passing arguments to the object constructor, but use a config file.

 - `pipeline`: Set the parameters of the the top-level pipeline, setting the parameters of the `Pipeline` object itself.

 - `preprocessing`: Set the parmeters of the  the `Preprocessor` object defined in `pipeline/preprocessing.py`.  This produces the `Image` object that is the base of everything that follows.

 - `extraction`: Set the parmeters for the `Detector` object defined in `pipeline/detection.py`, which produces the `SourceList` and `PSF` objects.  Note that this also includes the parmeters for background subtraction (which will be run using a `Backgrouder` object defined in `pipeline/backgrounding.py`.

 - `astrocal`: Set the parameters for the `AstroCalibrator` object defined in `pipeline/astro_cal.py`, which produces the `WorldCoordinates` object.

 - `photocal`: Set the parameters for the `PhotCalibrator` object defined in `pipeline/photo_cal.py`, which to produces the `ZeroPoint` object.

 - subtraction: Set the parameters for the `Subtractor` object defined in `pipeline/subtraction.py`, which produces a  second `Image` object with the difference image (and possibly some other associated objects).

 - detection: Set the parameters for a second `Detector` object, which produces a second `SourceList` object, this time from the difference image.

 - `cutting`: Set the parameters for the `Cutter` object defined in `pipeline/cutting.py`, which produces the `Cutouts` object.

 - `measuring`: Set the parameters for the `Measurer` object defined in `pipeline/measuring.py`, which produces a `MeasurementSet` object and an associated list of `Measurements` objects.

 - `scoring`: Set the parameters for the `Scorer` object defined in `pipeline/scoring.py`, which produces the `DeepScoreSet` object and an associated list of `Deepscore` objects.

 - `fakeinjection` : Set the parameters for the `FakeInjector` object.  When fake injection happens, a whole second thread of the pipeline runs from subtraction through scoring, this time starting with the image with the injected fakes.

An example of passing dictionaries to the `Pipeline` constructor to override the parameters defined in the yaml config file (which themselves override the defaults in the code):

```python
from pipeline.top_level import Pipeline
p = Pipeline(
   pipeline={pipeline_par1': pl_val1, 'pipeline_par2': pl_val2},
   preprocessing={'preprocessing_par1': pp_val1, 'preprocessing_par2': pp_val2},
   extraction={'sources_par1: sr_val1, 'sources_par2': sr_val2},
   ...
)
```

If only a single object needs to be initialized, pass the parameters directly to the object's constructor.  Note that in this case the parmeters from the config file will *not* be used; only when you instantiate a top-level pipeline are the values in the config file automatically used to configure your process object.

```python
from pipeline.preprocessing import Preprocessor
pp = Preprocessor(
   preprocessing_par1=pp_value1,
   preprocessing_par2=pp_value2
)
```

If you do want to use the configuration file to configure an individual processing step's object, you can use the config system to pass that:

```python
from util.config import Config
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
cfg = Config.get()
pp = Preprocessor( **cfg.value('preprocessing') )
ex = Detector( **cfg.value('extraction') )
ac = AstroCalibrator( **cfg.value('astrocal') )
```

Finally, after all objects are initialized with their parameters, a user (e.g., in an interactive session) can modify any of the parameters using the `pars` attribute of the object.

```python
pp.pars['preprocessing_par1'] = new_pp_value1
# or
pp.pars.preprocessing_par1 = new_pp_value1
```

To get a list of all the parameters that can be modified, their descriptions and default values, use

```python
pp.pars.show_pars()
```

The definition of the base `Parameters` object is at `pipeline/parameters.py`, but each process class has a dedicated `Parameters` subclass where the parameters are defined in the `__init__` method.

Additional information on using config files and the `Config` class
can be found (eventually) at :doc:`configuration`.

.. _overview-provenance:
### Versioning using the `Provenance` model

Each of the output data products is stamped with a `Provenance` object.  This object tracks the code version, the parameters chosen for that processing step, and the provenances of the data products used to produce this product (the "upstreams").  The `Provenance` object is defined in `models/provenance.py`.

Users interacting with the database outside of the main pipeline are likely want to use provenance tags; see :doc:`versioning`.

The `Provenance` object is initialized with the following inputs:

 - `process`: the name of the process that produced this data product ('preprocessing', 'subtraction', etc.).

 - `code_version` : the CodeVersion object for the code used to produce data products with this provenance.

 - `code_version_id`: (maybe given in place of `code_version): the opaque UUID of the appropriate code version object.

 - `parameters`: a dictionary of parameters that were used to produce this data product.

 - `upstreams`: a list of `Provenance` objects that were used to produce this data product.

The `code_version` is a `CodeVersion` object, defined also in `models/provenance.py`.  Each process has its own code version, so that upstream provenances do *not* need to change when code is modified for a downstream version.  Code versions are stored in the database, so provenances that point to code versions that are older than what's in the current running code still have something to point to.  code versions use semantic versioning.  The current semantic version in the code for each process is defined in the class variable `CodeVersion.CODE_VERSION_DICT`.  A provenance is only considered different if the major or minor parts of the code version change.  If only the third number (the patch part) of the semantic verison changes, this does not cause the provenance to change.  (On consequence of this is that any time you make code changes that would change the output of a process, you must bump at least the minor version for that process.)

The parameters dictionary should include only "critical" parameters, as defined by the `__init__` method of the specific process object, and should not include auxiliary parameters like verbosity or number of processors.  Only parameters that affect the product values are included.

The upstreams are other `Provenance` objects defined for the data products that are an input to the current processing step.  The flowchart of the different process steps is defined in `pipeline/datastore.py::UPSTREAM_STEPS`.  E.g., the upstreams for the `subtraction` object are `['preprocessing', 'extraction', 'referencing']`.  `referencing` is a special case; its upstream is replaced by the provenances of the reference's `preprocessing` and `extraction` steps.

When a `Provenance` object has all the required inputs, it will produce a hash identifier that is unique to that combination of inputs.  So, when given all those inputs, a user (or a datastore) can create a `Provenance` object and then query the database for the data product that has that provenance.

Additional details on data versioning can be found at :doc:`versioning`.

### Database schema

The database structure is defined using SQLAlchemy's object-relational model (ORM).  Each table is mapped to a python-side object; to find the PostgreSQL table that corresponds to a given class, look at the `__tablename__` class variable of that class (which usually shows up as the very first thing in the class definition).  The definitions of the models can be found in the various files in the `models` subdirectory.

The following classes define models and database tables, each associated with a data product.  Usually, the file in which a given model's definition can be found is obvious, but if not, a quick search for `class <modelname>` in the `models` subdirectory should suffice.

 - `KnownExposure`: an exposure that exists out there, somewhere, that we might want to process.  Except for the reference in this table, the exposure is not loaded into the database, nor is the data from that exposure generally available immediately to the pipeline.  This table contains enough information so that the appropriate subclass of `Instrument` is able to download the exposure.  The Conductor web applicaton maintains the `knownexposures` table (which is the table backing the `KnownExposure` class).  The first step of a pipeline is downloading an image pointed to by a `KnownExposure` and making an `Exposure` out of it.  (The `Pipeline` object in `pipeline/top_level.py` does *not* handle this; rather, it's handled before a Pipeline is launched by `pipeline/exposure_processor.py`.)

 - `Exposure`: a single exposure of all CCDs on the focal plane, linked to (usually) raw data on disk.  If there is an entry for an exposure in this table, that exposure is considered to be "in" the SeeChange database.

 - `Image`: a simple image, that has been preprocessed to remove bias, dark, and flat fields, etc.  An `Image` can be linked to one `Exposure` (in which case the `section_id` field of the `Image` defines which chip of that exposure the image came from), or it can be linked to a list of other `Image` objects (if it is a coadded image) or it can be linked to a reference and new `Image` objects (if it is a difference image).  If the image is a coadded image, its `is_coadd` field will be true, it will be linked to multiple rows in the `image_coadd_component_table` (defined in `models/zero_point.py), which in turn point ot the `ZeroPoint` objects that (by tracing back) define the images that went into this coadd.  If the image is a difference image, then the `is_sub` field will be true, and the `Image` will have an entry in the `image_subtraction_components` table (which is defined in `models/reference.py`), with the `image_id` in that table matching the `id` of the `Image` object of the difference image.  To find the "new" (or "search" or "science") and "ref" (or "template") images that this difference image corresponds to, the `image_subtraction_components` table has a `new_zp_id` that points to a `ZeroPoint` object(which can be joined up the chain to the appropriate `Image` object if desired) and a `ref_id` object that points to a `Reference` object.

 - `SourceList`: a catalog of light sources found in an image. It can be extracted from a regular image or from a subtraction. The `SourceList` is linked to a single `Image` and will have the coordinates of all the sources detected.  (However, it's possible that multiple `SourceList` objects link back to the same `Image` object, as the `SourceList` objects may have been produced with different extraction provenances.)

 - `Background`: information about the sky background of the image.  Sky backgrounds and source lists are produced at the same time (e.g. using SExtractor), so there is a 1:1 relationship between a SourceList and a Background in the database.

 - `PSF`: a model of the point spread function (PSF) of an image.  This is linked to a SourceList object, and holds the PSF model for the image that that source list is linked to.  As with Background, the PSF is produced in the same process as the SourceList, so there is a 1:1 relationship between PSF and SourceList objects in the database.  (This connection is there because you often need a preliminary source list to find a PSF, but then you want to use that PSF to get a final source list.  As such, you can't really do one step before the other.)

 - `WorldCoordinates`: a set of transformations used to convert between image pixel coordinates and sky coordinates.  This is linked to a single `SourceList` and will contain the WCS information for that image that the source list is linked to.  (Each `SourceList` may have multiple associated `WorldCoordiantes` with different provenances.)

 - `ZeroPoint`: a photometric solution that converts image flux to magnitudes.  This is linked to a single `WorldCoordinates` and will contain the zeropoint information for the image that the source list is linked to.  (Each `WorldCoordinates` may have multiple associated `ZeroPoint` objets.)

 - `Object`: a table that contains information about a single object found on a difference image (real or bogus).  Practically speaking, an object is defined (for the most part) as a position on the sky (modulo some flags like `is_fake` and `is_test`).  (When the pipeline finds something on a difference image, it will link the measurements of what it finds to an `Object` within a small radius (1") of the discovery's position, creating a new object if an appropriate one does not exist.)

 - `Cutouts`: encapsulates a list of small pixel stamps around a point in the sky in a new image, reference image, and subtraction image. Each `Cutouts` object is linked back to a single subtraction based `SourceList`, and will contain the three cutouts for each source in that source list.

 - `MeasurementSet` and `Measuremenets`:  contains measurements made on something detected in a difference image.   The `MeasurementSet` object links back to the `Cutouts` object, and each `Measurements` object points to a single `MeasurementSet` object.  Measurements are made on the three stamps (new, ref, sub) of one of the list of such stamps in a linked `Cutouts`.  Values include flux+errors, magnitude+errors, centroid positions, spot width, analytical cuts, etc.  The `index_in_sources` field of a single `Measurements` object points to the index in the `SourceList` object you can get by tracing the `MeasurementSet` background its associated `Cutouts` object.

 - `DeepScoreSet` and `DeepScore`: contains ML scores associated with Measurements.  The `DeepScoreSet` object links back to a `MeasurementSet`, and each `DeepScore` points to the `DeepScoreSet` object.  The `DeepScore` objects have an `index_in_sources` field used to associate them with the right `Measurements` object.  It contains a score in the range 0-1 (where higher means more likely to be real) assigned based on machine learning/deep learning algorithms.  Additionally contains a JSONB field which can contain additional information.

 - `Reference`: An object that identifies an image and associated data products as being a potential reference for the relevant filter and location on the sky.  It points back to a single `ZeroPoint` object (which can then be traced back to the `Image` object that is reference or template image).

 - `FakeSet`: A set of fakes that were injected into an image before it was run through subtraction and subsequent pipeline steps.  A `FakeSet` points to a `ZeroPoint`, which can be traced back to the `Image` on to which the fakes in this set are intended to be injected.  Note that the sundry data products that result from the actual fake injection are *not* saved to the database, instead:

 - `FakeAnalysis`: Stores data measured from recovered fakes in a given `FakeSet`.  Also points back to a `DeepScore`, so as to define the provenance steps used to analyze the fakes (which must match the provenance used in the main pipeline on the non-fake-injected image for the analysis to be useful).

 - `Provenance`: A table containing the code version and critical parameters that are unique to this version of the data.  Each data product above must link back to a provenance row, so we can recreate the conditions that produced this data.

 - `CodeVersion`:

 - `CalibratorFile`: An object that tracks data needed to apply calibration (preprocessing) for a specific instrument.  The calibration could include an `Image` data file, or a generic non-image `DataFile` object.

 - `DataFile`: An object that tracks non-image data on disk for use in, e.g., calibration/preprocessing.

 - `CatalogExcerpt`: An object that tracks a small subset of a large catalog, e.g., a small region of Gaia DR3 that is relevant to a specific image.  The excerpts are used as cached parts of the catalog, that can be reused for multiple images with the same pointing.


#### Additional classes

The `Instrument` class is defined in `models/instrument.py`.  Although it is not mapped to a database table, it contains important tools for processing images from different instruments.  For each instrument we define a subclass of `Instrument` (e.g., `DECam`) which defines the properties of that instrument and methods for loading the data, reading the headers, and other instrument-specific tasks.

More on instruments can be found at :doc:`instruments`.

Mixins are defined in `models/base.py`.  These are used as additional base classes for some of the models used in the pipeline, to give them certain attributes and methods that are shared across many classes.

These include:

 - `UUIDMixin`: Most of our database tables use a uuid as a primary key, and they do this by deriving from this class.  The choice of this over an auto-incrementing integer makes it easier to run the pipeline in multiple places at once, building up cross-references between objects without having to contact the database as each individual reference is constructed.  The actual database column is `_id`, but usually you should refer to the `id` (without underscore) property of an object.  (Accessing that property will automatically generate an id for a new object if it does not already have one.)


 - `FileOnDiskMixin`: adds a `filepath` attribute to a model (among other things), which make it possible to save/load the file to the archive, to find it on local storage, and to delete it.

 - `SpatiallyIndexed`: adds a right ascension and declination attributes to a model, and also adds a q3c spatial index to the database table.  This is used for many of the objects that are associated with a point on the celestial sphere, e.g., an `Image` or an `Object`.

 - `FourCorners`: adds the RA/dec for the four corners of an object, describing the bounding box of the object on the sky.  This is particularly useful for images but also for catalog excerpts, that span a small region of the sky.

 - `HasBitFlagBadness`: adds a `_bitflag` and `_upstream_bitflag` columns to the model.  These allow flagging of bad data products, either because they are bad themselves, or because one of their upstreams is bad. It also adds some methods and attributes to access the badness like `badness` and `append_badness()`.  If you change the bitflag of such an object, and it was already used to produce downstream products, make sure to use `update_downstream_badness()` to recursively update the badness of all downstream products.

Enums and bitflag are stored on the database as integers (short integers for Enums and long integers for bitflags).  These are mapped when loading/saving each database object using a set of dictionaries defined in `models/enums_and_bitflags.py`.

More information on data storage and retrieval can be found at :doc:`data_storage`.


### Parallelization

The pipeline is built around an assumption that large surveys have a natural way of segmenting their imaging data (e.g., by CCDs).  We also assume the processing of each section is independent of the others.  Thus, it is easy to parallelize the pipeline to work with each section separately.  We can just run a completely independent process to analyze each CCD of each exposure.  (While this is *almost* embarassingly parallel, there are some resource contention and race conditions we deal with in the pipline to handle cases of, e.g., several processes all trying to get the Gaia catalog for the same region of the sky at once, or several processes all trying to load a shared calibraiton file into the database at the same time.)

The executablse that run the main pipeline are `pipeline/exposure_processor.py` and `pipeline/pipeline_exposure_launcher.py`.  The latter is a wrapper around the former that contacts the conductor (a server that keeps track of which exposures are available for processing and which ones have been "claimed" by systems and clusters) and waits to be given an exposure to process.  Once it gets that exposure, it will create an `ExposureProcessor` object (defined in `pipeline/exposure_processor.py`).  The exposure processor it will use python's `mutliprocessing` to launch a configurable number of processes to analyze all of the CCDs of that exposure in parallel.  (Ideally, this will be as many processes as there are CCDs.  However, it works just fine with fewer, as some processes will serially run more than one CCD until all of them are done.)  Once the `ExposureProcessor` exits, the pipeline exposure launcher will contact the conductor again and ask for a new exposure to do.  The pipeline exposure launcher can be configured to exit after processing a certain number of exposures, or after a certain amount of time has passed.

You may also run `pipeline/exposure_processor.py` directly from the command line to manually process a given exposure.  However, this exposure must be known by the Conductor for this to work.  (We hope to make it so that that is not necessary, but that isn't done yet.)

Both `pipeline/exposure_processor.py` and `pipeline/pipeline_exposure_launcher.py` can be run with `--help` to see how to actually use them.

Currently, the pipeline does not (intentionally) use multiprocessing while working on a single chip.  Because we will have a large number of images and chips to process, it's easier to parallelize by just dividing up the work by chip rather than trying to make a single chip run as fast as possible.
