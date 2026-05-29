import pathlib
import time

import numpy as np

import sep

from models.base import SmartSession
from models.image import Image
from models.datafile import DataFile
from models.enums_and_bitflags import (
    image_preprocessing_inverse,
    string_to_bitflag,
    flag_image_bits_inverse,
    BitFlagConverter )


from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.logger import SCLogger


class ParsPreprocessor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.add_par( 'steps_required', [], list, "Steps that need to be done to each exposure" )

        self.add_par( 'preprocessing',
                      'internal',
                      str,
                      "Where was the preprocessing done?  'internal' means using SeeChange preprocessor starting "
                      "from raw images.  'noirlab_instcal' means image was loaded alredy preprocessed by the NOIRLab "
                      "pipeline.  Set this parameter manually to something other than 'internal' when loading "
                      "in already-preprocessed images.",
                      critical=True )

        self.add_par( 'overscan_method',
                      'median',
                      str,
                      ( "Method used for overscan.  Can median or polymedrej (see "
                        "instrument.py::Instrument.overscan_and_trim); possible that other instruments may "
                        "support other methods." ),
                      critical=True
                     )

        self.add_par( 'overscan_kwargs',
                      {},
                      dict,
                      ( "Additional keywords passed to Instrument.overscan_and_trim.  Meanings depend on "
                        "overscan_method" ),
                      critical=True
                     )

        self.add_par( 'use_base_mask',
                      True,
                      bool,
                      "Use the instrument's base mask when building the image mask.",
                      critical=True
                     )

        self.add_par( 'use_zero_mask',
                      True,
                      bool,
                      "If the bias subtraction step is applied, add masked pixels in the bias image to image mask",
                      critical=True
                     )

        self.add_par( 'use_flat_mask',
                      True,
                      bool,
                      "If flatfielding is performed, add masked pixels in the flat image to image mask",
                      critical=True
                     )

        self.add_par( 'masked_pixels_to_replace',
                      [],
                      list,
                      "Complicated",
                      critical=True
                     )

        self.add_par( 'masked_pixel_replacement',
                      {},
                      dict,
                      "Complicated",
                      critical=True
                     )

        self.add_par( 'calibset', 'externally_supplied', str,
                      "The calibrator set to use.  Choose one of the CalibratorSetConverter enum. ",
                      critical=True )
        self.add_alias( 'calibrator_set', 'calibset' )

        self.add_par( 'zero_provtag',
                      None,
                      ( str, type(None) ),
                      ( "If given, then when searching for a zero for bias subtraction, only accept "
                        "zero images that have a provenance tagged with this provenance tag.  "
                        "Do not use with externally_supplied, or you will probably regret it." ),
                      critical=True )

        self.add_par( 'flattype', 'externally_supplied', str,
                      "One of the FlatTypeConverter enum. ",
                      critical=True )

        self.add_par( 'flat_provtag',
                      None,
                      ( str, type(None) ),
                      ( "If given, when searching for a flat for flatfielding, only accept "
                        "flat images that have a provenance tagged with this provenance tag.  "
                        "Do not use with externally_supplied, or will will probably regret it." ),
                      critical=True )

        self.add_par( 'fringe_provtag',
                      None,
                      ( str, type(None) ),
                      ( "If given, when searching for a fringe image for fringe correction, only accept "
                        "fringe images that have a provenance tagged with this provenance tag.  "
                        "Do not use with externally_supplied, or will will probably regret it." ),
                      critical=True )

        self.add_par( 'purge_raw_data',
                      True,
                      bool,
                      "Set the raw_data field of image to None in an attempt to preserve memory",
                      critical=False )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'preprocessing'


class Preprocessor:
    def __init__(self, **kwargs):
        """Create a preprocessor.

        Preprocessing is instrument-defined, but usually includes a subset of:
          * overscan subtraction
          * bias (zero) subtraction
          * dark current subtraction
          * linearity correction
          * flatfielding
          * fringe correction
          * illumination correction

        After initialization, just call run() to perform the
        preprocessing.  This will return a DataStore with the
        preprocessed image.

        Parameters are parsed by ParsPreprocessor

        """

        self.pars = ParsPreprocessor( **kwargs )

        # Things that get cached
        self.instrument = None
        self.stepfilesids = {}
        self.stepfiles = {}

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False


    def preprocessing_done_bitfield( self ):
        strng = ','.join( self.pars.steps_required )
        return string_to_bitflag( strng, image_preprocessing_inverse )


    def run( self, *args, do_not_load=False, **kwargs ):
        """Run preprocessing for a given exposure and section_identifier.

        Parameters are passed to the data_store constructor (see
        DataStore.parse_args).  For preprocessing, an exposure and a
        sensorsection is required, so args must be one of:
          - DataStore (which has an exposure and a section)
          - exposure_id, section_identifier
          - Exposure, section_identifier
        Passing just an image won't work.

        Returns
        -------
        DataStore
          contains the products of the processing.

        """
        self.has_recalculated = False

        ds = None
        try:
            ds = DataStore.from_args( *args, **kwargs )
            t_start = time.perf_counter()
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            if ds.image is not None:
                baseobj = ds.image
                ds.section_id = ds.image.section_id
            elif ( ds.exposure is None ) or ( ds.section_id is None ):
                raise RuntimeError( "Preprocessing requires either an image, or an exposure and a sensor section" )
            else:
                baseobj = ds.exposure

            if ( self.instrument is None ) or ( self.instrument.name != baseobj.instrument ):
                self.instrument = baseobj.instrument_object

            # check that all required steps can be done (or have been done) by the instrument:
            known_steps = self.instrument.preprocessing_steps_available
            known_steps += self.instrument.preprocessing_steps_done
            known_steps = set(known_steps)
            steps_to_do = set(self.pars.steps_required)
            needed_steps = steps_to_do - set( self.instrument.preprocessing_steps_done )
            if not steps_to_do.issubset(known_steps):
                raise ValueError(
                    f'Missing some preprocessing steps {steps_to_do - known_steps} '
                    f'for instrument {self.instrument.name}'
                )

            # Get the calibrator files
            if not all( step in self.instrument.preprocessing_nofile_steps for step in needed_steps ):
                SCLogger.debug("preprocessing: getting calibrator files")
                preprocparam = self.instrument.preprocessing_calibrator_files( self.pars.calibset,
                                                                               self.pars.flattype,
                                                                               ds.section_id,
                                                                               baseobj.filter,
                                                                               baseobj.mjd,
                                                                               zero_provtag=self.pars.zero_provtag,
                                                                               flat_provtag=self.pars.flat_provtag,
                                                                               fringe_provtag=self.pars.fringe_provtag
                                                                              )
                SCLogger.debug("preprocessing: got calibrator files")
            else:
                preprocparam = {}

            # get the provenance for this step, using the current parameters:
            prov = ds.get_provenance('preprocessing', self.pars.get_critical_pars())

            # check if the image already exists in memory or in the database:
            image = None if do_not_load else ds.get_image( prov )

            image_was_from_exposure = False
            if image is None:  # need to make new image
                # get the single-chip image from the exposure
                image = Image.from_exposure( ds.exposure, ds.section_id )
                image_was_from_exposure = True

            if image is None:
                raise ValueError('Image cannot be None at this point!')

            if image.preproc_bitflag is None:
                image.preproc_bitflag = 0

            # Figure out how many steps we need to keep based on image type
            if image.type in self.instrument.preprocessing_steps_by_type:
                needed_steps = needed_steps.intersection( self.instrument.preprocessing_steps_by_type[ image.type ] )

            # Figure out if we skip any steps based on filter
            filter_skips = self.instrument.preprocessing_step_skip_by_filter.get(baseobj.filter, [])
            if not isinstance(filter_skips, list):
                raise ValueError(f'Filter skips parameter for {baseobj.filter} must be a list')
            filter_skips = set(filter_skips)
            needed_steps -= filter_skips

            if image._data is None:
                # in case we skip all preprocessing steps
                image.data = image.raw_data
                # Make sure the Exposure won't cache data we aren't using any more.
                if image_was_from_exposure:
                    ds.exposure.clear_cache()

            # The image keeps track of the steps already done to it in
            #   image.preproc_bitflag, which is translated into a string
            #   of comma-separated keywords in image.preprocessing_done.
            #   This includes the things that already were applied in
            #   the exposure, and should have been set when the image
            #   was extracted from the exposure, but does not include
            #   the things that were skipped for this filter (i.e., the
            #   instrument's preprocessing_step_skip_by_filter).
            already_done = set( image.preprocessing_done.split(', ') if image.preprocessing_done else [] )

            stilltodo = needed_steps - already_done
            # If self.pars.preprocessing is anything other than
            #   'internal', there should be nothing left to do except
            #   set the image provenance, and maybe calculate the weight
            #   and flags.  Verify that.
            if self.pars.preprocessing != 'internal':
                if len( stilltodo ) > 0:
                    raise ValueError( f"Preprocessing error: self.pars.preprocessing is {self.pars.preprocessing}, "
                                      f"but we still need to do steps {stilltodo} "
                                      f"(needed_steps={needed_steps}, preproc_bitflag={image.preproc_bitflag})" )

            if len( stilltodo ) == 0:
                SCLogger.debug( f"{pathlib.Path(image.filepath).name} has already been preprocessed, returning." )
                if ds.update_runtimes:
                    ds.runtimes['preprocessing'] = time.perf_counter() - t_start
                if ds.update_memory_usages:
                    import tracemalloc
                    ds.memory_usages['preprocessing'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB
                return ds

            else:
                # Still stuff to do!
                self.has_recalculated = True
                # Overscan is always first (as it reshapes the image)
                if 'overscan' in stilltodo:
                    SCLogger.debug('preprocessing: overscan and trim')
                    image.data = self.instrument.overscan_and_trim( image, method=self.pars.overscan_method,
                                                                    **self.pars.overscan_kwargs )
                    image.flags = np.zeros( image.data.shape, dtype=np.int16 )
                    # Update the header ra/dec calculations now that we know the real width/height
                    try:
                        image.set_corners_from_header_wcs(setradec=True)
                    except Exception as ex:
                        # No header WCS.  (Probably that's why there's an exception.)
                        SCLogger.warning( f"No header WCS, not setting corners for {image.filepath}" )
                        SCLogger.debug( str(ex) )
                    image.preproc_bitflag |= string_to_bitflag( 'overscan', image_preprocessing_inverse )
                    image.header['HISTORY'] = 'overscan corrected and trimmed by SeeChange'

                # If, for some reason, we don't yet have a  flags array, make sure we now do
                if image.flags is None:
                    image.flags = np.zeros( image.data.shape, dtype=np.int16 )

                # At this point, we won't use image.raw_data again.  Set it
                #   to None so the memory will be freed if it's not also
                #   referred somewhere else.
                if self.pars.purge_raw_data:
                    image.raw_data = None

                # Apply steps in the order expected by the instrument
                for step in self.pars.steps_required:
                    if step not in stilltodo:
                        continue
                    if step == 'overscan':
                        continue
                    SCLogger.debug(f"preprocessing: {step}")
                    stepfileid = None
                    # Acquire the calibration file
                    if f'{step}_fileid' in kwargs:
                        stepfileid = kwargs[ f'{step}_fileid' ]
                    elif f'{step}_fileid' in preprocparam:
                        stepfileid = preprocparam[ f'{step}_fileid' ]
                    else:
                        raise RuntimeError( f"Can't find calibration file for preprocessing step {step}" )

                    if stepfileid is None:
                        raise FileNotFoundError( f"Failed to find a {step} calibrator for filter "
                                                 f"section {image.section_id}, filter {baseobj.filter}" )

                    # Use the cached calibrator file for this step if it's the right one; otherwise, grab it
                    if ( stepfileid in self.stepfilesids ) and ( self.stepfilesids[step] == stepfileid ):
                        calibfile = self.stepfiles[ stepfileid ]
                    else:

                        with SmartSession() as session:
                            if step in [ 'zero', 'dark', 'flat', 'illumination', 'fringe' ]:
                                calibfile = session.get( Image, stepfileid )
                                if calibfile is None:
                                    raise RuntimeError( f"Unable to load image id {stepfileid} "
                                                        f"for preproc step {step}" )
                            elif step == 'linearity':
                                calibfile = session.get( DataFile, stepfileid )
                                if calibfile is None:
                                    raise RuntimeError( f"Unable to load datafile id {stepfileid} "
                                                        f"for preproc step {step}" )
                            else:
                                raise ValueError( f"Preprocessing step {step} has an unknown "
                                                  f"file type (image vs. datafile)" )
                        self.stepfilesids[ step ] = stepfileid
                        self.stepfiles[ step ] = calibfile
                    if step in [ 'zero', 'dark' ]:
                        # Subtract zeros and darks
                        image.data -= calibfile.data
                        if ( step == 'zero' ) and ( self.pars.use_zero_mask ) and ( calibfile.flags is not None ):
                            image.flags = np.bitwise_or( image.flags, calibfile.flags )
                        image.header['HISTORY'] = f'{step} subtracted by SeeChange with {calibfile.id}'
                        image.header['HISTORY'] = f'{step}: {calibfile.filepath}'

                    elif step in [ 'flat', 'illumination' ]:
                        # Divide flats and illuminations
                        image.data /= calibfile.data
                        if ( step == 'flat' ) and ( self.pars.use_flat_mask ) and ( calibfile.flags is not None ):
                            image.flags = np.bitwise_or( image.flags, calibfile.flags )
                        image.header['HISTORY'] = f'{step} divided by SeeChange with {calibfile.id}'
                        image.header['HISTORY'] = f'{step}: {calibfile.filepath}'

                    elif step == 'fringe':
                        # TODO FRINGE CORRECTION
                        SCLogger.warning( "Fringe correction not implemented" )

                    elif step == 'linearity':
                        # Linearity is instrument-specific
                        self.instrument.linearity_correct( image, linearitydata=calibfile )
                        image.header['HISTORY'] = f'linearity corrected by SeeChange with {calibfile.id}'
                        image.header['HISTORY'] = f'linearity: {calibfile.filepath}'

                    else:
                        # TODO: Replace this with a call into an instrument method?
                        # In that case, the logic above about acquiring step files
                        # will need to be updated.
                        raise ValueError( f"Unknown preprocessing step {step}" )

                    image.preproc_bitflag |= string_to_bitflag( step, image_preprocessing_inverse )

                # After all steps are done (so we won't be undermining
                # any steps by doing this), and so that things look
                # right in DS9 with the annoying "image" vs. "physical"
                # coordinates, strip out the stuff from the header that
                # told us how to trim now that we have trimmed.
                for yeet in self.instrument.overscan_trim_keywords_to_strip():
                    if yeet in image.header:
                        del image.header[yeet]

            # If we STILL don't have a flags image, by golly, we need one
            if image.flags is None:
                image.flags = np.zeros( image.data.shape, dtype=np.int16 )

            # OR in the standard instrument mask if we're supposed to
            if self.pars.use_base_mask:
                basemask = self.instrument.get_standard_flags_image( ds.section_id )
                image.flags |= basemask

            # Build the weight images (if necessary)
            if image.weight is None:
                # Estimate the background rms with sep
                boxsize = self.instrument.background_box_size
                filtsize = self.instrument.background_filt_size
                SCLogger.debug( "Subtracting sky and estimating sky RMS" )
                # Dysfunctionality alert: sep requires a *float* image for the mask
                # IEEE 32-bit floats have 23 bits in the mantissa, so they should
                # be able to precisely represent a 16-bit integer mask image
                # In any event, sep.Background uses >0 as "bad"
                fmask = np.array( image._flags, dtype=np.float32 )
                backgrounder = sep.Background( image.data, mask=fmask,
                                               bw=boxsize, bh=boxsize, fw=filtsize, fh=filtsize )
                del fmask
                variance = backgrounder.rms()   # It's not variance yet, but it will be
                sky = backgrounder.back()
                subim = image.data - sky
                SCLogger.debug( "Building weight image and augmenting flags image" )

                # Anywhere the rms given by the sep backgrounder is <=0, set the 'zero weight'
                #   bit in the flags image
                image.flags[ variance <= 0 ] &= ( np.int16(1) << BitFlagConverter.to_int('zero weight') )

                variance = variance ** 2        # see?
                subim[ subim < 0 ] = 0
                gain = self.instrument.average_gain( image )
                gain = gain if gain is not None else 1.
                # Shot noise from image above background
                # THOUGHT REQUIRED.  Where the sky fluctuates high, this will add
                #   additional noise that isn't real.  My instinct is that it's so
                #   piddly that we shouldn't worry about it, but in a very low-sky
                #   (e.g. short exposure?) situation, it could be non-piddly.
                variance += subim / gain
                wgood = ( image.flags == 0 )
                image.weight = np.zeros( image.data.shape, dtype=np.float32 )
                image.weight[ wgood ] = 1. / variance[ wgood ]
                # Figure out saturated pixels
                satlevel = self.instrument.average_saturation_limit( image )
                if satlevel is not None:
                    # WORRY.  We should really do this on *raw* data.  Hopefully doing it on
                    #   flatfielded data is not *that* big a deal.  (It's not sky-subtracted; that's subim.)
                    wsat = image.data >= satlevel
                    image.flags[ wsat ] |= string_to_bitflag( "saturated", flag_image_bits_inverse )
                    image.weight[ wsat ] = 0.

            # Replace some flagged pixels in the image if we were told to
            if len(self.pars.masked_pixels_to_replace) > 0:
                SCLogger.debug( "Replacing some flagged pixels in image" )
                didmask = np.int16(0)
                for pixname in self.pars.masked_pixels_to_replace:
                    bitmask = np.int16(1) << BitFlagConverter.to_int( pixname )
                    try:
                        repval = float( self.pars.masked_pixel_replacement[ pixname ] )
                        w = ( image.flags & bitmask != 0 ) & ( image.flags & didmask == 0 )
                        image.data[w] = repval
                        image.header['HISTORY'] = f'{pixname} flagged pixels replaced with {repval}'
                    except ValueError:
                        # TODO : support things like inpainting, image median
                        raise RuntimeError( "Currently only support floats for masked_pixel_replacmeent" )
                    didmask |= bitmask

            if image.provenance_id is None:
                image.provenance_id = prov.id
            else:
                if image.provenance_id != prov.id:
                    # Logically, this should never happen
                    raise ValueError('Provenance mismatch for image and provenance!')

            image.filepath = image.invent_filepath()
            SCLogger.debug( f"Done with {pathlib.Path(image.filepath).name}" )

            if image._upstream_bitflag is None:
                image._upstream_bitflag = 0
            if image_was_from_exposure:
                image._upstream_bitflag |= ds.exposure.bitflag

            ds.image = image

            if ds.update_runtimes:
                ds.runtimes['preprocessing'] = time.perf_counter() - t_start
            if ds.update_memory_usages:
                import tracemalloc
                ds.memory_usages['preprocessing'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Preprocessor.run: {e}" )
            if ds is not None:
                ds.exceptions.append( e )
            raise
