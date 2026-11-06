from models.instrument import Instrument
from models.exposure import Exposure
from models.image import Image

from pipeline.processobject import ProcessObject
from pipeline.pipeline import Pipeline

import util.util


class ParsExposureSplitter( Parameters ):
    def __init__( self, **kwargs ):
        super().__init__()

        self.deep_copy = self.add_par(
            name = 'deep_copy',
            default = True,
            par_types = bool,
            docstring = ( "When passing multiple section_ids, make deep copies of most pipeline "
                          "properties for the returned pipeline.  (See keep_products below.)" ),
            # Should we make this critical?  Probably not... but going to, so that if it turns
            #   out there is some subtle bug that crops up from doing deep vs. shallow
            #   copies, we can test it by running different provenances.
            critical = True
        )

        self.keep_status = self.add_par(
            name = 'keep_status',
            default = True,
            par_types = bool,
            docstring = "When passing multiple section_ids, keep the existing status properties.",
            critical = True
        )

        self.keep_products = self.add_par(
            name = 'keep_products',
            default = True,
            par_types = bool,
            docstring = "When passing multiple section_ids, keep existing data products.",
            critical = True
        )

        self.deep_copy_products = self.add_par(
            name = 'deep_copy_products',
            default = False,
            par_types = bool,
            docstring = ( "When passing multiple section_ids, make deep copies of the products dictionary? "
                          "Otherwise, copy the dictionary, but don't allocate new memory for all the objects. "
                          "True currently errors out." ),
            critical = True
        )
        
        self._enforce_new_new_attrs = True
        self.override( kwargs )

    def get_process_name( self ):
        return 'split_exposures'


class ExposureSplitter( ProcessObject ):
    """Extract raw images from an exposure (usually raw, anyway).

    Can either add to the image property of the passed pipeline, or
    return a whole bunch of new pipelines that each have a different
    image property for a different sensor section.
    
    """

    def __init__( self, **kwargs ):
        super().__init__( **kwargs )


    def run( self, pipeline, section_id, raw=True ):
        """Extract image(s) from exposure.

        Parameters
        ----------
           pipeline : Pipline
              The working pipeline.

           section_id : str or list of str
              The section ID of the image, *or* a list of same.

           raw : bool, default True
              The extracted image is a raw image

        Returns
        -------
           Image or list of Pipeline

           If section_id is a str, then the returned image is also now
           in pipeline.products['raw_image'].

           If section_id is a list, then pipeline is not modified, and
           the return value is a list of Pipelines, each of which has
           their 'raw_image' entry in products set (unless raw=False, in
           which case products['image'] is set.  Status fields (like
           runtimes, exceptions) are deep copied, as are properties
           holding provenances.  The products property of the pipeline
           is copied, but *not* deepcopied (so the actual product
           objects are the same, it's just the dictionary that's
           copied).

        """
        
        if not isinstance( pipeline, Pipeline ):
            raise TypeError( f"pipeline needs to be a Pipeline, not a {type(pipeline)}")

        if ( not hasattr( pipeline, exposure ) ) or ( not isinstance( pipeline.exposure, Exposure ) ):
            raise RuntimeError( "pipeline must have an exposure property that's an Exposure" )

        improp = 'raw_image' if raw else 'image'
        
        if util.util.is_actual_iterable( section_id ):
            SCLogger.info( f"Exposure splitter exctrating {len(section_id)} images." )
            pips = [ pipeline.copy( deep=self.pars.deep_copy,
                                    keep_status =self.pars.keep_status,
                                    keep_products=self.pars.keep_products,
                                    deep_copy_products=self.pars.deep_copy_procuts )
                     for i in section_id ]
            for secid, pip in zip( section_id, pips ):
                try:
                    pip.products[improp] = Image.from_exposure( pipeline.exposure, secid )
                except:

            return pips

        else:
            if ( improp in pipeline.products ) and ( pipeline.products[improp] is not None ):
                if pipeline.products[improp].section_id != section_id:
                    raise ValueError( f"Ran ExposureSplitter with section_id={section_id} on a pipeline "
                                      f"that already had an image with sectoin_id "
                                      f"{pipeline.products[improp].section_id}" )
                return pipeline.image
            else:
                SCLogger.info( f"Exposure splitter exracting section {section_id}")
                pipeline.products[improp] = Image.from_exposure( pipeline.exposure, secid )
                return pipeline.image
