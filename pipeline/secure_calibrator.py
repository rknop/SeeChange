from models.instrument import Instrument

from pipeline.processobject import ProcessObject


class ParsCalibratorSecurer( Parameters ):
    def __init__( self, **kwargs ):
        super().__init__()

        self.add_par( name='instrument',
                      default=None,
                      par_types= (str, None),
                      docstring='The name of the instrument.',
                      critical=True
                     )
        
        self.add_par( name='calibset',
                      default='externally_supplied',
                      par_types=str,
                      docstring="The calibrator set to use.  Choose one of the CalibratorSetConverter enum.",
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

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name( self ):
        return 'secure_calibrators'


class CalibratorSecurer( ProcessObject ):
    """A class that secures calibrators for a pipeline.

    A pipeline that calls this must have properties section_id, filter,
    and mjd.
    
    Saves a dictionary to pipeline.preprocparam which Preprocessor then uses.

    Most of the work is actually done inside Instrument subclasses.

    """
    
    _pars_class = ParsCalibratorSecurer
    _process_name = 'secure_calibrators'

    def __init__( self, **kwargs ):
        super().__init__( **kwargs )

    def run( self, pipeline ):
        if self.pars.instrument is None:
            raise ValueError( "Must give instrument name." )

        instrument = Instrument.get_instrument_instance( self.pars.instrument )
        pipeline.preprocparam = instrument.preprocessing_calibrator_files( self.pars.calibset,
                                                                           self.pars.flattype,
                                                                           pipeline.section_id,
                                                                           pipeline.filter,
                                                                           pipeline.mjd,
                                                                           zero_provtag=self.pars.zero_provtag,
                                                                           flat_provtag=self.pars.flag_provtag,
                                                                           fringe_provtag=self.pars.fringe_provtag )
        return pipeline.preprocparam
