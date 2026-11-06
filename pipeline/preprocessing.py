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
from pipeline.processobject import ProcessObject
from pipeline.pipeline import Pipeline

from util.logger import SCLogger


class ParsPreprocessor( Parameters ):
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

        self.add_par( 'purge_raw_data',
                      True,
                      bool,
                      "Set the raw_data field of image to None in an attempt to preserve memory",
                      critical=False )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'preprocessing'


class Preprocessor( ProcessObject ):
    _pars_class = ParsPreprocessor
    _process_name = 'preprocessing'
    
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

        The pipeline that uses this must have properties raw_image and
        preprocparam.  A call to run() will set the image property of
        the passed pipeline, and return the corresponding Image object.

        See ProcessObject for parameters that can be passed to the
        constructor.

        """

        super().__init__( **kwargs )

        # Cached calibrator files
        self.setpfileids = {}
        self.stepfiles = {}
        

    def preprocessing_done_bitfield( self ):
        strng = ','.join( self.pars.steps_required )
        return string_to_bitflag( strng, image_preprocessing_inverse )


    def run( self, pipeline ):
        """Run preprocessing.

        Updates the image property of pipeline, and returns the IMage it set there.

        """

        if not isinstance( pipeline, Pipeline ):
            raise TypeError( f"pipeline needs to be a Pipeline, not a {type(pipeline)}" )
        
        self.has_recalculated = False

        made_image = False
        made_flags = False
        try:
            t_start = time.perf_counter()
            if pipeline.track_memory:
                import tracemalloc
                tracemalloc.reset_peak()

            if not ( hasattr( pipeline, "preprocparam" ) and hasattr( pipeline, "raw_image" ) ):
                raise ValueError( "Preprocessor can't run, the passed pipeline needs to have both "
                                  "preprocparam and raw_image." )

            self.instrument = pipeline.raw_image.instrument_object

            # check that all required steps can be done (or have been done) by the instrument:
            known_steps = instrument.preprocessing_steps_available
            known_steps += instrument.preprocessing_steps_done
            known_steps = set(known_steps)
            steps_to_do = set(self.pars.steps_required)
            needed_steps = steps_to_do - set( instrument.preprocessing_steps_done )
            if not steps_to_do.issubset(known_steps):
                raise ValueError(
                    f'Missing some preprocessing steps {steps_to_do - known_steps} '
                    f'for instrument {instrument.name}'
                )

            preprocparam = pipeline.preprocparam
            
            # get the provenance for this step, using the current parameters:
            prov = pipeline.provenances[ self.process_name ]

            # check if the image already exists in memory or in the database:
            image = pipeline.get_product( 'image' )
            if image is not None:
                if not isinstance( image, Image ):
                    raise TypeError( f"pipeline.image needs to be an Image, not a {type(image)}" )
                return image

            # We didn't find an image, so we have to make it
            raw_image = pipeline.get_product( 'raw_image' )
            if not isinstance( image, Image ):
                raise TypeError( f"pipeline.raw_image needs to be an Image, not a {type(raw_image)}" )

            image = Image.copy_image( raw_image, no_copy_data=True )
            image.provenance_id = prov.id
            if any( getattr( image, prop ) for prop in ( 'is_coadd', 'is_sub', 'is_trim' ) ):
                raise RuntimeError( "Preprocessor got a raw_image with one of is_coadd, is_sub, or "
                                    "is_trim set... that should not happen." )
            for prop in ( 'fwhm_estimate', 'zero_point_estimate', 'lim_mag_estiamte',
                          'bkg_mean_estimate', 'bkg_rms_estimate', 'coadd_alignment_target' ):
                # ...or should we verify that these are None?  They should be for raw images.
                # But what if it's not really raw.  Hurm.
                setattr( image, prop, None )
            image._bitflag = 0
            image._upstream_bitflag = 0

            image.data = raw_image.data.copy()
            image.weight = raw_image.weight.copy() if raw_image.weight is not None else None
            image.flags = raw_image.flags.copy() if raw_image.flags is not None else None
            
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
                SCLogger.debug( "Image has already been preprocessed." )
                if ( image.weight is None ) or ( image.flags is None ):
                    raise RuntimeError( "Image was already preprocessed, but we're missing weight and/or flags!" )

            else:
                # Still stuff to do!
                self.has_recalculated = True
                # Overscan is always first (as it reshapes the image)
                if 'overscan' in stilltodo:
                    SCLogger.debug('preprocessing: overscan and trim')
                    image.data = self.instrument.overscan_and_trim( image, method=self.pars.overscan_method,
                                                                    **self.pars.overscan_kwargs )
                    image.flags = np.zeros( image.data.shape, dtype=np.int16 )
                    made_flags = True
                    # Update the header ra/dec calculations now that we know the real width/height
                    try:
                        image.set_corners_from_header_wcs(setradec=True)
                    except Exception as ex:
                        # No header WCS.  (Probably that's why there's an exception.)
                        SCLogger.warning( f"No header WCS, not setting corners for {image.filepath}" )
                        SCLogger.debug( str(ex) )
                    image.preproc_bitflag |= string_to_bitflag( 'overscan', image_preprocessing_inverse )
                    image.header['HISTORY'] = 'overscan corrected and trimmed by SeeChange'

                # If, for some reason, we don't yet have a flags array, make sure we now do
                if image.flags is None:
                    image.flags = np.zeros( image.data.shape, dtype=np.int16 )
                    made_flags = True

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
                        calibfile = self.stepfiles[ step ]
                    else:
                        with PGDB() as pgdb:
                            if step in [ 'zero', 'dark', 'flat', 'illumination', 'fringe' ]:
                                calibfile = Image.get_by_id( stepfileid, pgdb=pgdb )
                                if calibfile is None:
                                    raise RuntimeError( f"Unable to load image id {stepfileid} "
                                                        f"for preproc step {step}" )
                            elif step == 'linearity':
                                calibfile = DataFile.get_by_id( stepfileid, pgdb=pgdb )
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
                        image.info[step] = calibfile.id
                        image.header['HISTORY'] = f'{step} subtracted by SeeChange with {calibfile.id}'
                        image.header['HISTORY'] = f'{step}: {calibfile.filepath}'

                    elif step in [ 'flat', 'illumination' ]:
                        # Divide flats and illuminations
                        image.data /= calibfile.data
                        if ( step == 'flat' ) and ( self.pars.use_flat_mask ) and ( calibfile.flags is not None ):
                            image.flags = np.bitwise_or( image.flags, calibfile.flags )
                        image.info[step] = calibfile.id
                        image.header['HISTORY'] = f'{step} divided by SeeChange with {calibfile.id}'
                        image.header['HISTORY'] = f'{step}: {calibfile.filepath}'

                    elif step == 'fringe':
                        # TODO FRINGE CORRECTION
                        SCLogger.warning( "Fringe correction not implemented" )

                    elif step == 'linearity':
                        # Linearity is instrument-specific
                        self.instrument.linearity_correct( image, linearitydata=calibfile )
                        image.info['linearity'] = calibfile.id
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
                made_flags = True

            # OR in the standard instrument mask if we're supposed to
            if made_flags and self.pars.use_base_mask:
                basemask = self.instrument.get_standard_flags_image( image.section_id )
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

            # Replace some flagged pixels in the image if we were told to,
            #   and if we did anything to the image.  (Otherwise, don't touch it.)
            if ( made_image or ( len(stilltodo) > 0 ) ) and ( len(self.pars.masked_pixels_to_replace) > 0 ):
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

            image.filepath = image.invent_filepath()
            SCLogger.debug( f"Done preprocessing {pathlib.Path(image.filepath).name}" )


            # TODO : upstream bitflag from raw_image !!!!!
            if image._upstream_bitflag is None:
                image._upstream_bitflag = 0
            if image_was_from_exposure:
                image._upstream_bitflag |= ds.exposure.bitflag

            pipeline.image = image

            if pipeline.trac_runtimes:
                pipeline.runtimes['preprocessing'] = time.perf_counter() - t_start
            if pipeline.track_memory:
                import tracemalloc
                pipeline.memory_usages['preprocessing'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return image

        except Exception as e:
            SCLogger.exception( f"Exception in Preprocessor.run: {e}" )
            pipeline.exceptions.append( e )
            raise
