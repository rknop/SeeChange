from models.instrument import Instrument
from models.provenance import Provenance

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

        for calibitem in ( 'zero', 'dark', 'flat', 'linearity', 'fringe', 'illumination' ):
            self.add_par( name=f'{calibitem}_provenance_id',
                          default=None,
                          par_types=( str, None ),
                          docstring=( f"The provenance id of valid {calibitem} images.  Should be None for "
                                      f"calibset{' or flattype' if calibitem=='flat' else ''} "
                                      f"externally_supplied.  Maybe None if you don't care which provenance "
                                      f"of {calibitem} is found, as long as it's in the right calibset." )
                          critical=True
                         )

            self.add_par( name=f"{calibitem}_provtag",
                          default=None,
                          par_types=(str, None ),
                          docstring=( f"Upon process object creation, if the {calibitem}_provenance_id parameter is "
                                      f"None and this is not None, then use this (with {calibitem}_process} "
                                      f"to fill in the {calibitem}_provenance_id parameter.  Should be None for "
                                      f"calibset{' or flattype' if calibitem=='flat' else ''} "
                                      f"externally_supplied.  May be None if you don't care which provenance "
                                      f"of {calibitem} is found, as long as it's in the right set. "
                                      f"Ignored if {calibitem}_provenance_id is not None" )
                          critical=False
                         )

            self.add_par( name=f"{calibitem}_process",
                          default=None if calibitem in ('linearity', 'fringe') else 'flat_biase_builder',
                          par_types=(str, None),
                          docstring=(  f"process to use with {calibitem}_provtag.  Ignored if "
                                       f"{calibitem}_provenance_id is not None or if "
                                       f"{calibitem}_provtag is None" )
                         )


        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name( self ):
        return 'secure_calibrators'


class CalibratorSecurer( ProcessObject ):
    """A class that secures calibrators for a pipeline.

    A pipeline that calls this must have properties section_id, filter,
    and mjd.

    Most of the work is actually done inside Instrument and CalibratorSet.

    Will always insert rows into the database unless do_not_insert is passed to run().
    
    """
    
    _pars_class = ParsCalibratorSecurer
    _process_name = 'secure_calibrators'

    def __init__( self, **kwargs ):
        super().__init__( **kwargs )

        # Update the *_provenance_id parameters
        for calibitem in ( 'zero', 'dark', 'flat', 'linearity', 'fringe', 'illumination' ):
            if ( ( getattr( self.pars, f'{calibitem}_provenance_id' ) is None ) and
                 ( getattr( self.pars, f'{calibitem}_provtag' ) is not None )
                ):
                provtag = getattr( self.pars, f'{calibitem}_provtag' )
                process = getattr( self.pars, f'{calibitem}_process' )
                prov = Provenance.get_for_tag( provtag, process=process )
                if prov is None:
                    raise RuntimeError( f"Failed to find provenance with provtag {provtag} process {process}" )
                setattr( self.pars, f'{calibitem}_provenance_id', prov.id )
                

    def run( self, pipeline, do_not_insert=False ):
        if self.pars.instrument is None:
            raise ValueError( "Must give instrument name." )

        instrument = Instrument.get_instrument_instance( self.pars.instrument )

        provkwargs = { f'{calibitem}_provid': getattr( self.pars, f'{calibitem}_provenance_id' )
                       for calibitem in ( 'zero','dark', 'flat', 'linearity', 'fringe', 'illumination' ) }
        preprocparam = instrument.preprocessing_calibrator_files(
            self.pars.calibset,
            self.pars.flattype,
            pipeline.section_id,
            pipeline.filter,
            pipeline.mjd,
            **provkwargs
        )            

        calibset = CalibratorSet()
        for calibtype in ( 'zero', 'dark', 'linearity', 'flat', 'fringe', 'illumination' ):
            fileid = preprocparam[ f'{calibtype}_fileid' ]
            if fileid is not None:
                if preprocparam[ f'{calibtype}_isimage' ]:
                    setattr( calibset, f'{calibtype}_image_id', fileid )
                else:
                    setattr( calibset, f'{calibtype}_datafile_id', fileid )

        if not do_not_insert:
            calibset.get_or_insert()
                    
