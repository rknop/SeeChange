import copy
import textwrap

from psycopg import sql

from models.base import PGDB
from models.exposure import Exposure
from models.image import Image
from models.provenance import Provenance, CodeVersion
from util.config import NotDefined
from util.logger import SCLogger

from pipeline.data_store import ProvenanceTree
from pipeline.split_exposure import ExposureSplitter
from pipeline.calibrator_builder import CalibratorBuilder
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.ref_maker import RefMaker
from pipeline.alignment import Aligner
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.scoring import Scorer
from pipeline.alerting import Alerting

class Pipeline:
    """An abstract pipeline class.

    Instantiate a Pipeline with:

      pipeline_object = <Pipeline>( starting_prov, starting_point=<class>,
                                    <process>=<dict>, <process>=<dict>, ... )

    starting_prov is either the Provenance, or the id of a Provenance,
    of the object that the pipline would start workign on.
    starting_point is the class of the starting object.  For instance,
    for DifferenceImagingTransientDiscoveryPipeline, starting_point must
    be either Exposure or Image, and defaults to Exposure.

    IMPORTANT : most subclasses do not verify that starting_point is
    consistent with starting_prov; they trust that you've passed
    consistent things!  Stuff may break, or, worse, stuff may run but
    not do the right thing, if you mess this up.
    
    The remaining arguments are dictionaries of configuration parameters
    for the processes that correspond to the keys of the provenance tree
    that get_prov_tree will produce.  Any or all may be omitted.  Each
    process worker object is configured with:
    
      1. the defaults for that process
      2. <process> at the top leve of the config
      3. <pipeline_name>.<process> in the config
      4. The value of the <process> kwarg passed to the Pipeline
         constructor.
    
    where a later thing in the list above overrides an earlier thing.

    """

    # Subclasses need to define all four of these
    _name = "unknown_pipeline"
    _external_pipeline_processes = []
    _process_classes = {}
    _tree_upstreams = None

    @property
    def name( self ):
        return self._name

    @property
    def prov_tree( self ):
        return self._prov_tree

    @prov_tree.setter
    def prov_tree( self, tree ):
        if not isinstance( tree, ProvenanceTree ):
            raise TypeError( f"prov_tree must be a ProvenanceTree, not a {tree.__class__.__name__}" )
        self._prov_tree = tree
    
    def __init__( self, starting_prov, *args, starting_point=NotDefined, **kwargs ):
        if self.__class__.__name__ = "Pipeline":
            raise TypeError( "Can't instantiate a Pipeline, it's abstract.  Instantiate a Pipeline subclass." )

        if starting_point != NotDefined:
            kwargs['starting_point'] = starting_point
            
        self.prov_tree = self.get_prov_tree( starting_prov, *args, **kwargs )
        self.process_objects = {}


    def make_prov_tree( self, starting_prov=None, starting_process=None,
                        provtag=None, read_pars_from_prov=False, no_process_objects=False, pars={},
                        match_code_version=True, match_full_code_version=False,
                        handle_unexpected_processes='ignore', validate_upstream_steps=True,
                        pgdb=None,
                        _tree_upstreams=NotDefined, _process_classes=NotDefined,
                        _external_pipeline_processes=NotDefined ):
        """Return a ProvenanceTree for the full run of this pipeline.

        TODO : add a final_prov paramter, that, if given, specifies the
        last-step provenance, and build a tree including that provenance
        and all of its upstreams (honoring starting_prov and
        starting_process/provtag if given).
        
        Will save any not-yet-saved provenances to the database.
        
        See datastore.py::ProvenanceTree for what a ProvenanceTree is.
        But, basically, it's a dict of { process: Provenance }, where
        process is a string.  It also includes a dictionary
        upstream_steps of { process: [ list of process ] }, where all of
        "process" are strings.

        Will also fill self.process_objects with instances of classes
        that perform pipeline steps, unless no_process_objects is True.
        
        Parameters
        ----------
          starting_prov : Provenance or str
            The Provenance, or the id of the Provenance in the database,
            for the starting point object.  You must either give this,
            or you must give starting_process and provtag; in the latter
            case, the provenance defined by (starting_process, provtag)
            must exist in the database.

          starting_process : str, default None
            The process that should be the starting point of this
            provenance tree.  You don't need to give this if you give
            starting_prov.  In the returned ProvenanceTree, this process
            will not appear in as a key; rather, the provenance for that
            process will have the key 'starting_point'.

            If neither this nor starting_prov is given, if
            read_pars_from_prov is True, this function will try to
            automatically figure out the starting point.  Hopefully it
            guesses right.
        
          provtag : str, default None
            If not None, then associate these provenances with this provenance tag.

          read_pars_from_prov : bool, default False
            If False (default), then process objects will be generated
            using the parameters in pars (which override what's in the
            config and the process defaults).  If True, then we are
            assuming all of the provenances already exist and are tagged
            with provtag.  Read those provenances, and instantiate the
            process objects using those parameters.

            WARNING: in this case, pars is ignored, and there is no
            validation to find out if the parameters in the provenances
            read from the database match anything you passed in pars.

          no_process_objects : bool, default False
            Ignored if read_pars_from_prov is False; in that case,
            process objects will always be made, as they'll be needed to
            determine provenances.  By default, this method will load up
            self.process_objects with a dictionary of str->object, where
            object is an object of the class needed to perform that
            step.  If this is True AND read_pars_from_prov is True,
            then don't do this, and leave self.process_objects
            untouched.

          pars : dict
            A dictionary of {str: dict}, where str is a process name,
            and the dict are the **kwargs you'd pass to the cosntructor
            of the process class for that process.  If you want to use
            all the defaults for critical parameters from config, then
            you don't need to pass this.  If read_provs_from_prov is
            True, then you shouldn't pass this.  Otherwise, you need to
            pass this so that the right Provenances can be constructed.

          match_code_version : bool, default True
            Ignored if read_pars_from_prov is False.  If this is True, and the
            major.minor of a code version read from the provenance does not match
            the current code's major.minor version, raise an exception.  Otherwise,
            just issue a warning.

          match_full_code_version : bool, default False
            Ignored if read_pars_from_prov is False.  If this is True, and the
            major.minor.patch of a code version read from the provenance does not match
            the current code's major.minor version, raise an exception.  Otherwise,
            just issue a warning.  (Note that provenance hashes are built only from
            major.minor, so it's possible that two different code versions will be
            associated with the same provenance.  The idea is that we're really doing
            semantic verisoning right, and a change in the patch version will NOT change
            any existing data products were the process rerun with the same parameters.)

          handle_unexpected_processes : str, default "ignore"
            Ignored if read_pars_from_prov is False.  If the structure
            of the provenance tree is known — as it should be, because
            every Pipeline subclass should define an internal variable
            that specifies it — this says what to do if processes are
            read for a provtag that aren't expected.  This can easily
            happen if you associate a given provenance tag with additional
            things beyond what is done by the pipeilne.  If this
            is "ignore", the default, then those unexpected processes
            are just ignored.  If this is "warning", then those unexpected
            processes produce a warning, but are otherwise ignored.
            If this is "error", then when that happens an exception is raised.
        
          validate_upstream_steps : bool, default True
            Ignored if read_pars_from_prov is False.  If everything's
            working the way it's suppsoed to, then this shouldn't be
            needed if match_code_version is True.  However, by default
            do it anyway to be safe.  Even if match_code_version is
            False, you might still want to do this.  If this is True,
            make sure that the upstream steps of all the processes read
            from the database match what's expected.

          pgdb: PGDB, default None
            A database connection.  If not given, a new one will be
            opened and closed.

          _tree_upstreams, _process_classes, _external_pipeline_processes:
            These are all used internally and should only be passed by
            subclasses' get_prov_tree methods calling this parent method.

        Returns
        -------
          ProvenanceTree, or None

            If you set read_pars_from_prov, and no provenances were
            found in provtag, then you will get None back.

        """

        _pgdb = pgdb
        
        if self.__class__.__name__ == "Pipeline":
            # This may be redundant given the check in __init__
            raise TypeError( "Don't call get_prov_tree on a Pipeline object, Pipeline is an abstract class. "
                             "Call it on an object of a subclass." )

        starting_prov = None if starting_prov is None else Provenance.get( starting_prov )
        if ( ( starting_process is not None ) and ( starting_prov is not None ) and
             ( starting_prov.process != starting_process ) ):
            raise ValueError( f"Gave both starting_prov and starting_process, but starting_prov's process "
                              f"{starting_prov.process} doesn't match starting_process {starting_process}" )
        
        default_upstream_steps = self._tree_upstreams if _tree_upstreams == NotDefined else _tree_upstreams
        process_classes = self._process_classes if _process_classes == NotDefined else _process_classes
        external_pipeline_processes = ( self._external_pipeline_processes if _external_pipeline_processes == NotDefined
                                        else _external_pipeline_processes )

        # See if we're just reading everything
        if read_pars_from_prov:
            if provtag is None:
                raise ValueError( "read_pars_from_prov requires a non-None provtag" )
            prov_tree = ProvenanceTree()
            self.process_objects = {}
            upstream_steps = {}
            with PGDB( _pgdb ) as pgdb:
                rows, _cols = pgdb.execute( sql.SQL( "SELECT provenance_id FROM provenance_tags WHERE tag={tag}" )
                                            .format( tag=provtag ) )
                if len(rows) == 0:
                    return None

                # Build prov_tree and upstream_steps
                processes_seen = set()
                for row in rows:
                    prov = Provenance.get( row[0], pgdb=pgdb )
                    if prov.process in processes_seen:
                        raise RuntimeError( f"process {prov.process} shows up more than once for provenance tag "
                                            f"{provtag}; this should never happen." )
                    processes_seen.add( prov.process )
                    prov_tree[ prov.process ] = prov
                    if ( ( starting_prov is None ) and ( starting_process is not None ) and
                         ( prov.process == starting_process ) ):
                        starting_prov = prov
                    elif ( ( starting_process is None ) and ( starting_prov is not None ) and
                           ( prov.id == starting_prov.id ) ):
                        starting_process = prov.process
                    if prov.process in external_pipeline_processes:
                        upstream_steps[ prov.process ] = []
                    else:
                        upstream_steps[ prov.process ] = [ p.process for p in
                                                           prov.get_upstreams( pgdb=pgdb, save_to_object=True ) ]

                # Sort so upstreams are before their downstreams
                sortedprocs = ProvenanceTree.sort_processes( upstream_steps )
                
                # Crop out everything before the starting point if that is given and found
                if starting_process is not None:
                    try:
                        dex = sortedprocs.index( starting_process )
                    except Exception:
                        starting_process = None
                        starting_prov = None
                        dex = -1
                    if dex > 0:
                        for proc in sortedprocs[ :dex ]:
                            del prov_tree[ proc ]
                            del upstream_steps[ proc ]
                        sortedprocs = sortedprocs[ dex: ]

                # Call the first step starting_point, and wipe out its upstream steps (since it's the starting point!)
                upstream_steps['starting_point'] = []
                del upstream_steps[ sortedsteps[0] ]
                prov_tree['starting_point'] = prov_tree[ sortedsteps[0] ]
                del prov_tree[ sortedsteps[0] ]
                sortedsteps[0] = 'starting_point'

                # Validate upstream steps if requested
                if validate_upstream_steps:
                    mykeys = set( upstream_steps.keys() )
                    defaultkeys = set( default_upstream_steps.keys() ).union( {'starting_point'}  )
                    if not mykeys.issubset( defaultkeys ):
                        raise ValueError( f"The set of steps found for provenance tags {provtag} "
                                          f"included some unexpected things: {mykeys - defaultkeys}" )
                    mismatches = []
                    for proc in upstream_steps:
                        if proc == 'starting_point':
                            continue
                        myups = set( upstream_steps[proc] )
                        defups = set( default_upstream_steps[proc] )
                        if myups != defups:
                            mismatches.append( f"Upstream steps found for process {key} in provtag {provtag} "
                                               f"{upstream_steps[key]} does not match what was "
                                               f"expected {default_upstream_steps[key]}" )
                    if len(mismatches) > 0:
                        nl = "\n    "
                        raise ValueError( f"Unexpected provenance tree structure:{nl}{nl.join(mismatches)}" )
                
                # Validate code versions
                for step in sortedsteps:
                    process = starting_process if step == 'starting_point' else step
                    curcv = Provenance.get_code_version( process, pgdb=pgdb )
                    provcv = CodeVersion.get_by_id( prov_tree[step].code_version_id, pgdb=pgdb )
                    errmsg = ( f"Current version for process {process} is "
                               f"{curcv.version_major}.{curcv.version_minor}.{curcv.version_patch}, "
                               f"which does not match what's found for provenance tag {provtag}: "
                               f"{provcv.version_major}.{provcv.version_minor}.{provcv.version_patch" )
                    if curcv.id == provcv.id:
                        # If curcv.id = provcv.id, then major and minor must match, but patch may not
                        if curcv.version_patch != provcv.version_patch:
                            if match_full_code_version:
                                raise ValueError( errmsg )
                            else:
                                SCLogger.warning( errmsg )
                    else:
                        if match_code_version:
                            raise ValueError( errmsg )
                        else:
                            SCLogger.warning( errmsg )

            # Make all the process objects
            if not no_process_objects:
                thingstomake = sortedsteps if starting_process is None else sortedsteps[1:]
                if not set(thingstomake).issubset(process_classes.keys()):
                    raise ValueError( f"There are steps found for which we don't have process classes: "
                                      f"{set(thingstomake) - set(proces_classes.keys())}" )
                self.process_objects = {}
                for step in thingstomake:
                    self.process_objects[step] = process_classes[step]( **(prov_tree[step].params) )

            # Done
            prov_tree.upstream_steps = upstream_steps
            return prov_tree

        else:
            # If we're here, read_pars_from_prov was false, so we have to construct all the
            #   provenances from parameters. 
                
            if starting_prov is None:
                if ( starting_process is None ) or ( provtag is None ):
                    raise RuntimeError( "Must either give a starting_prov, or a starting_process and "
                                        "a provtag so we can find it in the database." )
                starting_prov = Provenance.get_for_tag( provtag, process=process )
                if starting_prov is None:
                    raise ValueError( f"Could not find starting point provenance for process {process} "
                                      f"in provenance tag {provtag}" )
            else:
                # Convert id to Provenance object if necessary
                starting_prov = Provenance.get( starting_prov )
                if starting_prov is None:
                    raise ValueError( f"Unknown starting provenance {starting_prov}" )

            upstream_steps = copy.deepcopy( default_upstream_steps )

            pars = {} if pars is None else pars
            if ( ( not isinstance( pars, dict ) ) or
                 ( not all( isinstance( v, dict ) for v in pars.values() ) ) or
                 ( not all( isinstance( k, str ) for k in pars.keys() ) )
                ):
                raise TypeError( "pars must be a dictionary of str:dict" )

            if ( ( upstream_steps is None ) or
                 ( not isinstance( upstream_steps, dict ) ) or
                 ( not all( isinstance( k, str ) for k in upstream_steps.keys() ) ) or
                 ( not all( isinstance( v, list ) for v in upstream_steps.values() ) ) or
                 ( not all( all( isinstance( vv, str ) for vv in v ) for v in upstream_steps.values() ) ) or
                 ( 'starting_point' not in upstream_steps ) or
                 ( len( upstream_steps['starting_point'] ) > 0 )
                ):
                raise TypeError( "upstream_steps must be a dict of str: list of str, "
                                 "and must include a starting_point that has no upstreams" )

            errors = []
            known_steps = set( upstream_steps.keys() )
            for process, upstreams in upstream_steps.items():
                ups = set( upstreams )
                if not ups.issubset( known_steps ):
                    errors.append( f"Unknown upstreams for {process}: {ups - known_steps}" )
            if len(errors) > 0:
                raise ValueError( "\n".join( errors ) )

            # Figure out the order the steps need to go in
            sortedprocs = ProvenanceTree.sort_processes( upstream_steps )

            if sortedprocs[0] != 'starting_point':
                raise RuntimeError( "starting_point didn't come first.  This might actually not be an error, "
                                    "just an indication that more code needs to be written right here to move "
                                    "it around.  But maybe it is an error.  If you see this exception, then tell "
                                    "Rob about it, and he will have to think harder." )
            
            # Make process objects and provenance tree
            known_steps = set( process_classes.keys() ).union( { 'starting_point' } )
            if not set( sortedprocs ).is_subset( known_steps ):
                raise ValueError( f"Some steps don't have known process objects: {set(sortedprocs) - known_steps}" )

            if not set( pars.keys() ).issubset( known_steps ):
                raise ValueError( f"Parameters given for unknown steps: {set(pars.keys()) - known_steps}" )

            provs = ProvenanceTree()
            self.process_objects = {}
            with PGDB( pgdb ) as pgdb:
                for step in sortedprocs:
                    if step == 'starting_point':
                        provs[step] = starting_prov
                    else:
                        if step in external_pipeline_processes:
                            provs[step] = Provenance.get( external_pipeline_processes[step] )
                            # TODO : gracefully fail if this provenance doesn't exist.  That could
                            #   happen, if a pipeline has been configured but things that have to
                            #   be run before this pipeline could run haven't happened yet.
                            #   In this case, we'll probably need to wipe out all downstreams.
                            if provs[step] is None:
                                raise RuntimeError( f"Failed to find provenance for external process {step}.  "
                                                    f"Ideally, the code should handle this, but it doesn't yet." )
                            self.process_objects[step] = process_classes[step]( **(prov.params) )
                        else:
                            kwargs = pars[step] if step in pars else {}
                            procobj = process_classes[step]( **kwargs )
                            params = {} if not hasattr( procobj, 'pars' ) else procobj.pars.get_critical_pars()
                            upstreams = [ provs[s] for s in upstream_steps[step] ]
                            upstreams.sort( key=lambda x: x.id )
                            code_version = Provenance.get_code_version( process=step ).id
                            prov = Provenance( code_version=code_version,
                                               process=step,
                                               parameters=params,
                                               upstreams=upstreams )
                            provs[step] = prov
                            self.process_objects[step] = procobj

            # Associate and verify provenance tags

            ROB TODO YOU NEED TO DO THIS OMG
                            
            # Done
            provs.upstream_steps = upstream_steps
            return provs
                        

    def make_process_objects( self, always_remake=False ):
        """Create the process objects the pipeline will use."""
        raise NotImplementedError( f"{self.__class__.__name__} needs to implement make_process_objects" )
        


# ======================================================================

class DifferenceImagingTransientDiscoveryPipeline( Pipeline ):

    _name = "dia_transientsearch"

    _all_steps = [ 'split_exposure', 'secure_calibrators', 'preprocessing', 'extraction', 'astrocal', 'photocal',
                   'referencing', 'alignment', 'subtraction', 'detection', 'cutting', 'measuring', 'scoring',
                   'alerting' ]

    # NOTE : referencing and secure_calibrators are special cases.
    #   referencing always has upstreams, and secure_calibrators *might*
    #   have upstreams, but those are handled in separate pipelines, so
    #   the upstreams are not tracked here.  The provenance resulting
    #   from referencing is defined by the refset parameter of
    #   subtraction, and provenance of calibrators is defined by the
    #   calibset parameter of preprocessing.
    _external_pipeline_processes = { 'secure_calibrators', 'referencing' }
    _tree_upstreams = { 'starting_point': [],
                        'split_exposure': [ 'starting_point' ],
                        'secure_calibrators': [],
                        'preprocessing': [ 'split_exposure', 'secure_calibrators' ],
                        'extraction': [ 'preprocessing' ],
                        'astrocal': [ 'extraction' ],
                        'photocal': [ 'astrocal' ],
                        'referencing': [],
                        'alignment': [ 'referencing', 'photocal' ],
                        'subtraction': [ 'alignment' ],
                        'detection': [ 'subtraction' ],
                        'cutting': [ 'detection' ],
                        'measuring': [ 'cutting' ],
                        'scoring': [ 'measuring' ]
                       }

    _process_classes = { 'split_exposure': ExposureSplitter,
                         'calibrator_builder': CalibratorBuilder,
                         'preprocessing': Preprocessor,
                         'extraction': Detector,
                         'astrocal': AstroCalibrator,
                         'photocal': PhotCalibrator,
                         'ref_maker': RefMaker,
                         'alignment': Aligner,
                         'subtraction': Subtractor,
                         'cutting': Cutter,
                         'measuring': Measurer,
                         'scoring': Scorer,
                         'alerting': Alerter
                        }
        
                        
    def get_prov_tree( self, starting_prov, starting_point=Exposure ):
    """See Pipeline::get_prov_Tree.

        starting_point must be either Exposure or Image

        If starting_point is Exposure, then the pipeline will start at
        split_exposure.  Otherwise, starting_point must be Image.  In
        that case, if the process of starting_prov is 'preprocessing',
        then the pipeline will start at 'secure_calibrators' followed by
        'preprocessing'.  Otherwise, the pipeline will start at
        'extraction'.

        """

        if starting_point not in ( Exposure, Image ):
            raise TypeError( f"starting_point must be Exposure or Image, not {starting_point}" )
        
        starting_prov = Provenance.get( starting_prov )
        if starting_prov is None:
            raise ValueError( f"Failed to find provenance {starting_prov}" )

        upstreams = copy.deepcopy( self._tree_upstreams )
        
