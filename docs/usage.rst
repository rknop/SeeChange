.. toctree::

***************
Using SeeChange
***************

There are two primary usages for SeeChange.  One is if you want to access the SeeChange data products produced by a pipeline for futher processing, the second is if you want to run the discovery pipeline itself.  This documentation assumes that for the first usage, you will not be saving additional data products to the database.  Of course, in the future, we may want to have follow-up processing (e.g. a precision lightcurve pipeline that works on the objects discovered by the standard discovery pipeline?), but that is not currently supported.

**Important**: make sure you understand the data model (described in :doc:`data_storage`), and in particular make sure you understand :ref:`provenances`.  If you don't use provenances right, you probably won't get the data you think you're getting when you obtain data products for futher processing, and you'll just make a mess if you're running the pipeline.

###################################################################
Examining images and other data produced previously by the pipeline
###################################################################

Getting set up
==============

Make sure you've installed the right version of the SeeChange code and that you've properly set everythign up (see :doc:`setup`).  In this case, if you look at :ref:`config`, you should be working in the case where the person who runs the pipeline has given you a config file that gives you access to the database and the data file archive.

**If you want to look at the production LS4 pipeline data products**, then you must have an account on NERSC.  See :ref:`nersc-setup`.

Once you're set up, all code you run must be inside the container that the setup tells you to create.  You can edit python scripts outside of the container; put them in your ``$RUNDIR``, and they will be accessible inside the container at ``/home``.

Finding images
==============

Suppose you want to find what images have been loaded into the database.  (Note that just because you find an image, it doesn't guarantee that the image will have been run all the way through the pipeline to discover transients.)

Assuming you've set everything up, and have properly pointed the right environment variables to the right config file, you can just run the ``Image.find_images`` static method to search for images.  First, though, you need to get the right image Provenance.  The process to produce images is *usually* ``preprocessing``.   You will need to know which provenance tag you want to search.  This should be defined somewhere in documentation maintained by the manager of the discovery pipeline for the survey you're working with.  If you don't know what to do, you can try using ``default`` as a provenance tag and hope that it works.  (As of this moment, that will *not* work on the ls4 pipeline; ask Rob to find out what the right current provenance tag is, and be aware that it's rapidly evolving right now)  For example::

  from models.provenance import Provenance
  prov = Provenance.get_for_tag( tag="<tag>", process="preprocessing" )

replaceing ``<tag>`` with the right thing.  If all is well, ``prov`` will be an object of class ``Provenance``.  If ``prov`` is None, then there is no preprocessing provenance in the database for the requested tag.

Provenance in hand, you can now try to find images.  Read the docstring on ``Image.find_images`` in ``models/image.py`` for a full list of what you can search on.  For now, let's suppose that you're looking for all images in the filter "r" whose footprint on the sky include given ra and dec (which we'll call ``ra0, dec0``)::

  from models.image import Image
  images = Image.find_images( provenance_ids=prov.id, ra=ra0, dec=dec0, filter="r" )

This will return a list of ``Image`` objects.  There are a large number of things you can search on; run ``help(Image.find_images)`` to see the docstring.

For each image object, there are a number of properties that you can access.  Some of the most useful (and obvious) properties are ``mjd``, ``filter``, ``ra``, ``dec``, ``exp_time``, ``width``, ``height``.  The property ``section_id`` holds the name of the chip this image came from.  There are eight properties like ``ra_corner_00``, where you can put ``dec`` in place of ``ra``, and any of ``01``, ``10``, or ``11`` in place of ``00``.  These describe the four corners of the image.  (Note: these are the four corners of the image without regard to bad regions, e.g. dead amplifiers or the messy overlap region where multiple filters are blurred together.  You can get more useful corners by looking at the ``ra_good_00`` (etc.) properties of a ``WorldCoordinates`` object (more later).

You can find the image data in the ``data`` property, the weights (1/σ²) in the ``weight`` property, and pixel flags in the ``flags`` property.  (Generally speaking, if the value of ``flags`` for a pixel is 0, that's a good pixel, otherwise it's not.  There are definitions for all the bits; talk to Rob if you care.)  All of those are numpy arrays with the same length.  (They are lazy-loaded, so when you first get the list of images from ``find_images``, the data will not yet be using up memory.  They are loaded the first time you access the relevant property.)

Images *may*, but won't always, have properties loaded ``fwhm_estimate`` (seeing FWHM in arcseconds), ``zero_ponit_estimate``, ``bkg_mean_estimate``, ``bkg_rms_estimate``, and ``lim_mag_estimate``.  Because of the aforementioned provenance system, each image may have *multiple* versions of each of these properties associated with it!  The one that gets stored in the Image record is the one that happened to get stored first, which may not be the
best one.

Data is stored so that ``image.data[y, x]`` holds the pixel value of pixel ``(x, y)``.  (*Notice that y and x are indexed backwards in the array!*  This has to do with the most natural numpy data storage format and how images are stored on disk.)  SeeChange indexes all pixels from 0 (**which is different from FITS!**), so ``(0.0, 0.0)`` is the center of the lower-left pixel.  If you look at an image using ``ds9``, be aware that its pixel positions will be offset by 1 from anything you get in numpy arrays!


Finding Source Lists, PSFs, WCSes, and Zeropoints
=================================================

Once you have an Image object, you can get associated data products with::

  products = img.get_sources_wcs_zp( '<tag>' )

where ``<tag>``is the provenance tag discussed above.  You have to get the provenance tag right, or you wan't get what you're looking for.  The ``products`` return here is a dictionary with keys ``sources``, ``bg``, ``psf``, ``wcs``, and ``zp``.  Each of these has some useful properties.

sources
-------

This is a ``SourceList`` object defined in ``models/source_lists.py``.  Probably the most important properties are ``x``, ``y``, ``errx``, ``erry``, ``ra``, ``dec``, ``good``, and ``is_star``.  Each of these are arrays with the same length that are for all the objects in the source catalog.  SeeChange supports multiple different algorithms for finding source catalogs.  Right now, the LS4 production pipeline is using Sextractor.  As such, you can find the Sextractor data table in the ``data`` property, but the format of that property is not universally defined, so any code you write that uses it may break if we switch formats later.  **Import**: SeeChanges keeps all pixel positions as 0-offset, so all x and y values are offset by 1 from what you'd get directly reading the sextractor output file!

The method ``psffluxadu()`` returns two numpy arrays; the first has the flux of all objects measured with PSF photometry, in ADU, the second has the uncertainty on the first.  The lenght of these arrays should be the same as the length of the ``x`` property (etc) of the ``SourceList`` object.

The method ``apfluxadu( n )`` returns two numpy arrays, flux and uncertainty on flux, for aperture photometry in an aperture.  ``n`` corresponds the index into the ``aper_rads`` property.  For ls4, we've configured it so that ``n=1`` corresponds to a 1~FWHM radius aperture (which is often approximately what you want to use).  This flux is *not* aperture corrected.


bg (background)
---------------

This holds the background.  If you have a ``Background`` object ``bg``, you can run ``bg.subtract_me( data )``, where ``data`` is (say) the ``data`` property of an ``Image``, and you'll get back a background-subtracted data array.

psf
---

This holds information about the PSF.  Right now, the pipeline uses PSFEx to find PSFs, but in principle we might use something else.  The ``fwhm_pixels`` property gives you an approximation of the seeing FWHM; but, of course, it can vary across the image.  The method ``get_clip(x, y)`` returns you a thumbnail with an image of the PSF, normalized to 1 (which may not be exactly the right thing to do...).  ``x`` and ``y`` can be integers, in which case the PSF will be centered on the thumbnail.  If they have non-zero fractional parts, then the PSF will have a subpixel shift away from the cetner of the returned thumbnail.

wcs
----

This holds a WorldCoordinates object.  Useful properties are ``ra_good_00``, where you can replace ``ra`` with ``dec``, and you can replace ``00`` with ``01``, ``10``, and ``11``.  These give you a subregion on the image that is not masked. (The way it's done right now probably requires some further thought, and it's probably too aggressive; one cosmic ray might wipe the whole thing out.)

The *most* important property is ``wcs``, which gives you an astropy WCS to work with.  (So ``products['wcs'].wcs`` is an astropy WCS.)


zp (zeropoint)
--------------

This holds a ZeroPoint object.  The most useful properties are ``zp`` and ``dzp``, which hold AB-magnitude zeropoints that you can use together with the flux values you get out of the SourceList. You can also find estimates of aperture corrections in ``aper_cor_radii`` and ``aper_cors``, where ``aper_cors`` is an array holding magnitude aperture corrections for apertures with pixel radii in ``aper_cor_radii``.  (These radii should be the same as the ``aper_rads`` in the corresponding SourceList object.)

EVERYTHING BELOW IS OLD AND STILL NEEDS TO BE EDITED


Finding further data products
-----------------------------

When it runs all the way through, the pipeline produces a number of data products, including a WCS and zeropoint for the image, including source lists from images, difference images, lists of objects detected on difference images, measurements on those detections etc.  You *could*, provenance and image id in hand, construct database queries to find all of these, but there is an easier way.  The ``DataStore`` object, defined in ``pipeline/data_store.py``, is used internally by the pipeline to keep track of what it's produced so far and what it still needs to produce.  You can also use it to get access to the data products produced by a previous run of the pipeline.

First, create a data store from an ``Image`` object you already have, which we'll call ``image``.  (This might be, for instance, ``images[0]`` from the example above.)::

  from pipeline.data_store import DataStore
  ds = DataStore( image )

Next, load up the data store with all the possible provenances for data products you might wish to find.  For this, you will use the same provenance tag you found when finding the ``Image`` in the first place::

  ds.load_prov_tree( "<tag>" )

replacing ``<tag>`` with right string.  Now, the data store is primed to be able to give you a variety of data products.  The functions you can call on a ``DataStore`` object, and what it returns, are:

``ds.get_image()``
  Returns the ``Image`` for this datastore.  You already have this, though.  You can also access this more directly by just referring to the ``image`` property of the ``DataStore`` with ``ds.image``.

``ds.get_sources()``
  Returns a ``SourceList`` object (defined in ``models/source_list.py``) with sources extracted from the image (e.g. via SExtractor, depending on how the pipeline was configured).

``ds.get_psf()``
  Returns a ``PSF`` object (defined in ``models/psf.py``), which holds information about the PSF of the image.  This was probably found using ``psfex``.  The most useful method of a ``PSF`` object is probably ``get_clip``.  (See the docstring for documentation.)

``ds.get_background()``
  Returns a ``Background`` object (defined in ``models/background.py``) that holds information about sky subtraction that was performed on the image before further processing (such as image subtraction).

``ds.get_wcs()``
  Returns a ``WorldCoordinates`` object (defined in ``models/world_coordinates.py``).  The most useful property of a ``WorldCoordiantes`` object is probably ``.wcs``, which gives you astropy WCS object that you can use as you normally would.  **IMPORTANT**: Do not try reading the WCS from the image header.  That's going to be whatever raw WCS was in the header when we first got it from the telescope!  Use ``DataStore.get_wcs()`` to get a WCS that was determined by fitting star positions to the Gaia catalog.

``ds.get_zp()``
  Returns a ``ZeroPoint`` object (defined in ``models/zero_point.py``).  The most useful properties of a ``ZeroPoint`` object are ``.zp`` and ``.dzp`` (though the uncertainties currently produced are very dubious), with the zeropoint defined so that ``mag = 2.5*log10(adu) + zp``, where ``adu`` is the total counts on an image in a star or other point-like object (determined e.g. from PSF-fitting photometry).  If you're doing aperture photometry, you can also use the ``.aper_cor_radii`` property (which gives a list of the pixel radii for apertures with known aperture corrections) and ``.aper_cors`` (which gives those aperture corrections, defined so that ``mag = -2.5*log10(adu_aper) + zp + aper_cor``).  These aperture corrections are defined (I believe, check this) on the PSF at the center of the image, and will not be very accurate of the PSF varies significantly across the image.  **Note**: Because SeeChange, as currently conceived, is a discovery pipeline, not a precision photometry pipeline, we do not worry about precise calibration.  SeeChange doesn't know anything about color terms, for instances.  The zeropoints in a ``ZeroPoint`` object should not be assumed to be good to better than a couple of percent.  SeeChange determines its zeropoints by using the Gaia catalog as a photometric reference.  For them to be any good, the instrument you're using must have a set of transformations between the Gaia photometric system and the filter bandpasses of your instrument. AS of this writing, only DECam has a transformation implemented.

``ds.get_referenece()``
  Usually, returns an ``Image`` object that holds the reference image that was used by the pipeline in image subtraction.  This reference image is *not* aligned with the science image, nor is it scaled to the science image.  (Warning: this function does more than a quick database lookup.  It tries to identify the reference for this image from a defined set of references for the referencing provenance stored in the DataStore.  What it actually does can be a bit complicated.  If all is well, it will work as you expect, but it's *possible* that multiple references will be defined for a given image, which SeeChange does not handle well right now.  SeeChange currently operates in the assumption, which we *really hope* will hold for LS4, that science images are taken on a pre-defined grid, so we can pre-compute references that align with that grid, and only need to consider a single pre-computed reference for any given image.  Maybe, at some point in the future, SeeChange will be able to handle dealing with multiple references that overlap an arbitrarily positioned image, but it does not right now.)

``ds.get_sub_image()``
  Returns an ``Image`` that is the difference image produced by the pipeline, subtracting a (scaled and warped) reference image from the science image.  This ``Image`` is aligned with and scaled to the main image, so you can use the same WCS and zeropoint.  (Be careful about what the zeropoint means on the difference image, however; it is the obvious definition for people searching for supernovae, and is extremely confusing for people searching for variable stars.)

``ds.get_detections()``
  Returns a ``SourceList`` object (defined in ``models/source_list.py``) with sources detected on the difference image (e.g. by SExtractor).

``ds.get_cutouts()``
  Returns a ``Cutouts`` object (defined in ``models/source_list.py``).  You can dig into what these are, but they're much easier to use if you get a ``MeasurementSet`` first (see below).

``ds.get_measurement_set()``
  Returns a ``MeasurementSet`` object (defined in ``models/measurements.py``).  This holds a list of measurements on some of the detections you found from ``get_detections()``, but not all of them.  (Some detections will have been filtered out by cuts, and not saved to the database).  The ``.measurements`` property is a list of ``Measurements`` objects (also defined in ``models/measurements.py``), each of which has useful properties like ``.flux_psf``, ``.flux_psf_err``, ``.x``, ``.y``, and others.  (TODO: think about the ``.zp`` property and whether we should recommend setting it, or not using it and just using the zeropoint you get with ``ds.get_zeropoint()``.)

  You can get the cutouts associated with a single ``Measurements`` object by accessing the following properties:

  * new_data
  * new_weight
  * new_flags
  * ref_data
  * ref_weight
  * ref_flags
  * sub_data
  * sub_weight
  * sub_flags

  "new" refernces to a cutout on the science image, "ref" refers to a cutout on the reference image (warped to and scaled to the new image), and "sub" refers to a cutout on the difference image.  "data" is the image itself, "weight" is the weight data (1/σ²), and "flags" is the pixel flags/mask image.  (Generally, any pixel that is not 0 in a flags image indicates a "bad" pixel.)

    These are all lazy-loaded when you first access any one of them.  You can save a bit of database access by manually triggering the loading by calling the ``.get_data_from_cutouts`` method of a ``Measurements`` object, and passing it cutouts and detections objects you pulled from the datastore, but this is probably not a great gain in efficiency.

``ds.get_deepscore_set()``
  TODO

``ds.get_deepscores()``
  TODO

``ds.get_fakes()``
  TODO

``ds.get_fakeanal()``
  TODO


Finding detected objects
------------------------

TODO


Getting all measurements and/or cutouts for an object
-----------------------------------------------------

TODO


Running the pipeline to process images and find transients
==========================================================

TODO


