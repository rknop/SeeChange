import io
import copy
import collections

from psycopg import sql

from models.base import PGDB, UUIDMixin, HasProvenance
from models.provenance import Provenance, CodeVersion
from util.config import NotDefined
from util.logger import SCLogger

from pipeline.processobject import ProcessObject


class Pipeline( ProcessObject ):
    """An abstract pipeline class.

    Instantiate a Pipeline of subclass <Pipeline> with:

      pipeline_object = <Pipeline>( starting_prov, starting_point=<class>,
                                    <process>=<dict>, <process>=<dict>, ... )

    starting_prov is either the Provenance, or the id of a Provenance,
    of the object that the pipline would start working on.
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
      2. <process> at the top level of the config
      3. <pipeline_name>.<process> in the config
      4. The value of the <process> kwarg passed to the Pipeline
         constructor.

    where a later thing in the list above overrides an earlier thing.

    """

    # Subclasses need to define all four of these.
    # See dia_discovery_pipeline.py for the canonical example.

    # This is the name of the pipeline.
    #   Do NOT define a _name that already exists!  Before writing a new
    #   class, try calling Pipeline.get( name ) on your name to see if
    #   it returns anything.  If it raises a ValueError with "Unknown
    #   pipeline", then you're safe.
    _name = "unknown_pipeline"

    # This is the subclass of Parameters that holds the parameters
    #   for this pipeline.  (This is actually defined originally
    #   in ProcessObject, the superclass of this class, and also
    #   the superclass of all process classes.)
    _pars_class = None

    # This is the default name for this process.  Config will use this
    #   as the top-level key to get the base configuration for this
    #   process.  (Configuration is complicated, ROB WRITE MORE.)
    _process_name = None

    # A list of strings; all the steps (process names) of this pipeline
    # in the order they will normally be run.
    _all_steps = []

    # A dict of str: class.  The keys are the names of properties that
    #   the pipeline needs to get started.  The values are the
    #   classes of those starting products.  Many pipelines will
    #   only have a single entry in this dictionary.
    _starting_products = {}

    # A dict of str: list of tuple, where each tuple is (class, str) or
    #    (list, str, class).  The key defines the prodcess that produces
    #    products in the list.  In the tuple that is each element of the
    #    list, str defines the property of a DataStore that holds a data
    #    product (or input), and class is the class of that product.  In
    #    the case of a 3-element tuple, the first element is literally
    #    list, not a list.  In that case, the property# str points to a
    #    list of objects whose class is defined by the third element of
    #    the tuple.
    _process_products = {}

    # A dict of str: str.  The key is a process product (which is either
    # one of the second elements of one of the tupels from the lists in
    # the values of _process_products, or is something that the pipeline
    # will start with).  The value is the name of the process used for
    # looking up the provenance (e.g. with a provenance tag).  If the
    # value is None, this indicates that this is a process that does not
    # have a provenance.  (E.g., Background and PSF do not have
    # provenances, because there is a 1:1 relationship between
    # SourceList and each of those, so the SourceList provenance defines
    # the Background and PSF provenance.)  If the value is NotDefined,
    # that indicates the pipeline has to do something custom to set this
    # provenance process.  (That will be the case for data products that
    # are pre-defined as starting points for the pipeline.)  If a product
    # does not appear in this dictionary, then the process is the key
    # from _process_products where this data product appeared.  (So, in principle,
    # this could be a nearly empty dictionary.)
    #
    # This dictionary does *not* need to be complete.  It will be automatically
    # populated from _starting_products and _process_products.  Only
    # put in here entries where a product has a provenance, but the
    # process of that provenances doesn't match the step name of the process.
    _product_provenance_processes = {}

    # A dict of str: list of str.  The key is a data product, the value
    # is a list of data products that are immediately upstream of this
    # data product.  Those upstream data products are used both to
    # construct Provenances, and for a Pipeline to find things.  Every
    # data product that will be referenced by the pipeline, or that will
    # be produced by the pipeline, should have an entry.  If the data
    # product is something that's not actually produced by the pipeline
    # (because it's a starting point, or because the process that
    # produces it is a loader process, not the thing that actually makes
    # it), the value should be NotDefined instead of a list of str.
    _product_upstreams = {}

    # A dict of str: class.  For each process (step) that is part of
    # this pipeline's running, the class is the class that performs that
    # process.
    _process_classes = {}

    # A dictionary of product: function, special code for loading
    # self.products for that product.
    _product_loaders = {}

    # A dictionary of product: function, special code for validating
    # that a data product is consistent with other data products
    # already stored.  Should raise an exception if it's not.
    _product_validators = {}


    @property
    def name( self ):
        return self._name


    @classmethod
    def get( cls, pipeline, pars=None, provtag=None, read_pars_from_prov=False, never_execute=NotDefined,
             match_code_version=True, match_full_code_version=False,
             validate_upstream_steps=True, ok_processes_missing=True, handle_unexpected_processes='ignore',
             pgdb=None, **kwargs ):
        """Get a Pipeline subclass.

        Different subclasses may have different additional requirements,
        but you must supply one of the following sets of kwargs.  (You
        may well want to provide others as well!)

          * <product>=<object, uuid, or str> [, <product>=<object>, ...]
              The starting points for the pipeline.  For example, for
              the image reduction pipeline, you could do:

                 pip = Pipeline.get( exposure=exp )

              where exposure_obj is either the instance of an Exposure,
              or the id of an exposure.

          * provtag=str, <product>=<object, uuid, or str>
              Pass any data product that's anywhere in the pipeline.
              You can pass the appropriate object, or the id of the
              object.  For instance, for the image reduction pipeline,
              you could do
                 pip = Pipeline.get( provatag='mytag', wcs=my_wcs )
              The pipeline will complain if it cannot find all the
              provenances it needs given what you gave it.

              You *probably* want to set read_pars_from_prov=True when
              using this version.

          * starting_prov : either a str, Provenance, or a dict of str: (str or Provenance)
              For pipelines that have a single starting point
              (e.g. ImageReductionPipeline starts with 'exposure"), this
              is the provenance (or provenance id ) of we'd
              start with.  For a pipeline with multiple starting points,
              the dictionary gives all of them.

          * provtag=str, read_pars_from_prov=True
              In this case, read all of the Provenances from the given provenance tag.

        Parameters
        ----------
          pipeline : str
             The name of the pipeline, e.g. 'image_reduction',
             'dia_transientsearch', etc.  Controls the class of the
             object that will be returned.

          pars : dictionary, default None
             OMG complicated, Rob write docs.

          provtag : str, default None
             A provenance tag.  If given, and read_pars_from_prov is
             True, then all pipeline process object parameters are read
             from the provenances in the database, and the config is
             completely ignored.  If read_pars_from_prov is False, then
             if any Provenances saved in this provenance tag are
             inconsistent with the parameters in active configuration
             files and what was passed in pars, an exception will be
             raised.  Additionally, if read_pars_from_prov is False,
             the pipeline will create Provenances and save them to the
             database, tagged with this tag (if it's not None).

          read_pars_from_prov: str, default False
             If True, then pars must not be given.  The config will be
             ignored (...sort of...), and all process object parameters
             will be set by what's read from the provenances tagged with
             this provenance.  Use this when you just want to get existing
             data products that may not be from the current configuration.

          never_execute : bool, default NotDefined
             If this is not given, then it defaults to True if
             read_pars_from_prov is given, otherwise it defaults to
             False.  When this is True, the pipeline will not create any
             new Provenances, nor will it do anything.

          match_code_version : bool, default True
             Ignored if provtag is not given.  If any found Provenances have
             a major or minor code version that doesn't match what's current,
             raise an exception. You *might* be able to use this with
             read_pars_from_prov if you know what you're doing; otherwise,
             you really don't want to make this False.

          match_full_code_version : bool, default False
             Ignored if porvtag isn't given. Require that the patch
             part of the code version also match.  If we did our
             semantic verisoning right, then it should be safe to leave
             this at False.  Make it True if you don't trust us.  I
             don't know if I trust us.

          validate_upstream_steps : bool, default True
             Ignored if provtag isn't given.  Leave this as True.

          ok_processes_missing : bool, default True
             Ignored if provtag isn't given.  When provenances are read
             from the database, if this is False and provenances aren't
             found for all expected processes, raise an Exception.

          handle_unexpected_processes : str, default 'ignore'
             Ignored if provtag isn't given.  If processes are found
             in provtag that aren't part of what this pipeline expects,
             raise an exception.  That can easily happen, so you probably
             just want to leave this at ignore.

        Returns
        -------
          An object of the appropriate subclass of Pipeline

        """
        queue = collections.deque( cls.__subclasses__() )
        subclasses = {}
        while queue:
            subclass = queue.popleft()
            if subclass._name not in subclasses:
                subclasses[ subclass._name ] = subclass
                queue.extend( subclass.__subclasses__() )
        if pipeline not in subclasses:
            raise ValueError( f"Unknown pipeline {pipeline}" )
        else:
            return subclasses[ pipeline ]( **kwargs)


    def _is_product_upstream( self, downstream, upstream ):
        if self.product_upstreams[ downstream ] == NotDefined:
            return False
        elif upstream in self.product_upstreams[ downstream ]:
            return True
        else:
            for check in self.product_upstreams[ downstream ]:
                if self._is_product_upstream( check, upstream ):
                    return True
        return False


    def __init__( self, starting_prov=None, provtag=None, pars=None, read_pars_from_prov=False,
                  match_code_version=True, match_full_code_version=False,
                  handle_unexpected_processes='ignore', ok_processes_missing=True, validate_upstream_steps=True,
                  reports=False, pgdb=None, track_runtimes=True, track_memory=True,
                  _preset_provenances={}, _ok_to_instantiate=False, **kwargs
                 ):
        """Do not ever directly instantiate a Pipeline or subclass.  Use Pipeline.get()."""

        _pgdb = pgdb

        if not _ok_to_instantiate:
            raise RuntimeError( f"Don't instantiate a {self.__class__.__name__}, instead use Pipeline.get()" )

        if self.__class__.__name__ == "Pipeline":
            raise TypeError( "Can't instantiate a Pipeline, it's abstract.  Instantiate a Pipeline subclass." )

        pars = pars if pars is not None else {}
        if not isinstance( pars, dict ):
            raise TypeError( f"pars must be a dict, not a {type(pars)}")

        # Make copies of the pipeline definition so we can modify it for this instance
        for prop in [ 'all_steps', 'starting_products', 'process_products',
                      'product_provenance_processes', 'product_upstreams', 'process_classes' ]:
            if prop in kwargs:
                setattr( self, prop, copy.deepcopy( kwargs[prop] ) )
            else:
                setattr( self, prop, copy.deepcopy( getattr( self, f"_{prop}" ) ) )

        self._do_reports = reports
        self.track_runtimes = track_runtimes
        self.track_memory = track_memory
        self.runtimes = {}
        self.memory_usages = {}
        self.exceptions = []
        
        # Make a dictionary of product classes.  The keys of
        #   self.product_classes will have every product we might try to
        #   access inside the pipeline itself (so process objects called
        #   by the pipeline may access other things), or that we will
        #   produce as part of the running of the pipeline.  The value
        #   of the dictionary will *usually* be a class, but it might be
        #   a tuple of (list, class).
        #
        # Also, at the same time, make sure that
        #   self.product_provenance_processes is complete (has an entry
        #   for everything in self.product_classes).  It will be
        #   NotDefined for products that are starting points, and None
        #   for classes that aren't subclasses of HasProvenance.  for
        #   products that don't have provenances.
        self.product_classes = self.starting_products.copy()
        self.product_provenance_processes.extend( { c: NotDefined for c in self.product_classes.keys() } )
        for process, mess in self.process_products.items():
            for tup in mess:
                islist = tup[0] is list
                product = tup[1]
                cls = tup[2] if islist else tup[0]

                if product in self.product_classes:
                    raise RuntimeError( f"{self.__class__.__name__} is ill-defined, {product} shows up "
                                        f"more than once either in _starting_products or _process_products" )
                if islist:
                    self.product_classes[ product ] = ( list, cls )
                else:
                    self.product_classes[ product ] = cls

                if cls not in self.product_provenance_processes:
                    if isinstance( cls, HasProvenance ):
                        self.product_provenance_processes[product] = process
                    else:
                        self.product_provenance_processes[product] = None

        # Make sure all our dicts of products to stuff all have the same keys
        produpkeys = set( self.product_upstreams.keys() )
        prodclskeys= set( self.product_classes.keys() )
        prodprovprockeys =set( self.product_provenance_processes.keys() )
        if ( produpkeys != prodclskeys ) or ( produpkeys != prodprovprockeys ):
            strio = io.StringIO()
            strio.write( f"{self.__class__.__name__} is ill-defined. The keys of _product_upstreams "
                         f"don't match both the keys of product_classes and product_provenance_processes." )
            if len( produpkeys - prodclskeys ) != 0:
                strio.write( f"  In product_upstreams but not product_classes: {produpkeys-prodclskeys}" )
            if len( prodclskeys - produpkeys ) != 0:
                strio.write( f"  In product_classes but not product_upstreams: {prodclskeys-produpkeys}" )
            if len( produpkeys - prodprovprockeys ) != 0:
                strio.write( f"  In product_upstreams but not product_provenance_processes: "
                             f"{produpkeys-prodprovprockeys}" )
            if len( prodprovprockeys - produpkeys ) != 0:
                strio.write( f"  In product_provenance_processes but not product_upstreams: "
                             f"{prodprovprockeys-produpkeys}" )
            SCLogger.exception( strio.getvalue())
            raise RuntimeError( strio.getvalue() )

        # Initialize our internal tracking of provenaces (values are all Provenance)
        # This is preliminary... we're later going to validate this against the provtag
        # and/or the provenances of all the process objects.
        self.provenances = { k: Provenance.get(v, pgdb=_pgdb) for k, v in _preset_provenances }

        # Populate any passed data products (and update self.provenances with them)
        self.products = {}
        for k, cls in self.product_classes.items():
            if k in kwargs:
                obj = UUIDMixin.get_object( kwargs[k], cls, no_none=True, pgdb=_pgdb )
                if ( k in self.provenances ) and ( obj.provenance_id != self.provenances[k].id ):
                    raise ValueError( f"Passed {k} has provenance_id {obj.provenance_id}, which doesn't match "
                                      f"already-determined provenance of {self.provenances[k].id}" )
                self.provenances[ k ] = Provenance.get( obj.provenance_id, pgdb=_pgdb )
                self.products[ k ] = obj
                del kwargs[ k ]

        # Make sure we got enough to go on
        self.starting_prov = {}
        if starting_prov is None:
            gotallstarts = all( k in self.products for k in self.starting_products.keys() )
            gotonederived = any( ( k in self.products ) and ( k not in self.starting_products.keys() )
                                 for k in self.product_classes.keys() )
            if gotallstarts:
                if any( (prod in self.product_upstreams) and (self.product_upstrams[prod] != NotDefined)
                        for prod in self.starting_products.keys() ):
                    raise RuntimeError( "Something bad has happened." )

                self.starting_prov = { k: Provenance.get( getattr( self, k ).provenance_id,
                                                          must_exist=True, pgdb=_pgdb )
                                       for k in self.starting_products }

            if not ( gotallstarts or gotonederived ):
                if ( provtag is None ) or ( not read_pars_from_prov ):
                    raise ValueError( "Not enough information to instantiate a pipeline.  Look at the "
                                      "docstring for pipeline.Pipeline.get" )

        else:
            if not isinstance( starting_prov, dict ):
                if len( self.starting_products ) != 1:
                    raise ValueError( f"{self.__class__.__name__} has {len(self.starting_products)} "
                                      f"starting products ({self.starting_products}), but you only "
                                      f"gave one starting_prov." )
                if isinstance( starting_prov, ( Provenance, str ) ):
                    if isinstance( starting_prov, str ):
                        starting_prov = Provenance.get( starting_prov, must_exist=True, pgdb=_pgdb )
                    self.starting_prov = { self.starting_products[0] : starting_prov }
                else:
                    given = set( starting_prov.keys() )
                    needed = set( self.starting_products.keys() )
                    if given != needed:
                        strio = io.StringIO()
                        strio.write( "Didn't give the right set of starting provenances." )
                        if len( needed - given ) > 0:
                            strio.write( f"  Missing: {needed-given}" )
                        if len( given - needed ) > 0:
                            strio.write( f"  Unknown: {given-needed}" )
                        raise ValueError( strio.getvalue() )
                    self.starting_prov = { k: v if isinstance( v, Provenance )
                                           else Provenance.get(v, must_exist=True, pgdb=_pgdb)
                                           for k, v in starting_prov.items() }

        # Read existing provenancs and do a consistency check.
        if provtag is not None:
            existingprovs = self.read_existing_provenances( provtag,
                                                            match_code_version=match_code_version,
                                                            match_full_code_version=match_full_code_version,
                                                            validate_upstream_steps=validate_upstream_steps,
                                                            handle_unexpected_processes=handle_unexpected_processes,
                                                            pgdb=pgdb )
            mismatches = {}
            for process, prov in self.provenances:
                if ( process in existingprovs ) and ( self.provenances[process].id !=
                                                      existingprovs[process].id ):
                    mismatches[process] = ( self.provenancnes[process].id, existingprovs[process].id )

            if len(mismatches) > 0:
                strio = io.StringIO()
                strio.write( f"Some provenances in the database for provtag {provtag} don't match "
                             f"provenances you gave me, or provenances of objects you gave me." )
                for process, them in mismatches.items():
                    strio.write( f"\n  {process:>24s} : database : {them[1]}  given : {them[0]}" )
                SCLogger.error( strio.getvalue() )
                raise ValueError( strio.getvalue() )


        # If read_pars_from_prov was true, then update the pars dictionary accordingly
        if read_pars_from_prov:
            readpars = { proc: self.provenances[proc].parameters
                         for proc in self.provenances if proc in self.all_steps }
            overlaps = set( readpars.keys() ).intersection( pars.keys() )
            if len(overlaps) != 0:
                SCLogger.warning( f"You passed parameters for {overlaps}, but set read_pars_from_prov; "
                                  f"the parameters you passed are going to get wiped out." )
            pars.update( readpars )

            if not ok_processes_missing:
                needed = set( p for p in self.all_steps if len(self.process_products[p]) > 0 )
                if not needed.issubset( readpars ):
                    raise RuntimeError( f"Failed to read provenances for processes: {needed-readpars}" )

        # We are finally ready to call our superclass
        super(ProcessObject, self).__init__( pipeline=None, ignore_missing_config_values=True, **pars )

        # Make process objects
        self.process_objects = { process: cls( pipeline=self, process_name=process )
                                 for process, cls in self.process_classes }

        # Fill out the products array
        for product in self.product_classes.keys():
            if product not in self.products:
                self.products[ product ] = None


    # ======================================================================

    def get_product( self, product, do_not_load=False, always_load=False, pgdb=None ):
        if ( not always_load ) and ( self.products[ product ] is not None ):
            return self.products[ product ]

        if do_not_load:
            return self.products[ product ]

        if product in self._product_loaders:
            self.products[ product ] = self._product_loaders[ product ]( do_not_load=do_not_load,
                                                                         always_load=always_load,
                                                                         pgdb=pgdb )
        else:
            if self.product_upstreams[ product ] == NotDefined:
                self.products[ product ] = None

            else:
                upstrs = {}
                for produp in self.product_upstreams[ product ]:
                    upstrs[produp] = self.get_product( produp, do_not_load=do_not_load,
                                                       always_load=always_load, pgdb=pgdb )
                if any( v is None for v in upstrs.values() ):
                    self.products[ product ] = None
                else:
                    with PGDB( pgdb, dictcursor=True ) as pgdb:
                        tab = sql.Identifier( self.product_classes[product].__tablename__ )
                        q = sql.SQL( "SELECT * FROM {tab}").format( tab=tab )
                        where = sql.SQL( "WHERE" )
                        for produp, upstr in upstrs.items():
                            q += sql.SQL( " {where} {column}={upid}"
                                          ).format( where=where,
                                                    column=sql.Identfier( f"{produp}_id" ),
                                                    upid=upstr.id )
                            where = sql.SQL( "AND" )
                        rows = pgdb.execute( q )
                        if len(rows) > 1:
                            raise RuntimeError( "This should never happen." )
                        elif len(rows) == 0:
                            self.products[ product ] = None
                        else:
                            self.products[ product ] = self.product_clases[product].create( **(rows[0]))

        return self.products[ product ]


    # ======================================================================

    def set_product( self, product, value ):
        if not isinstance( value, self.product_classes[ product ] ):
            raise TypeError( f"{product} must be a {self.product_classes[product].__name__}, not a {type(value)}" )

        provprod = self.product_provenance_processes[ product ]
        if ( provprod is not None ) and ( value.provenance_id != provprod.id ):
            raise ValueError( f"Can't set {product} with an object whose provenance id is {value.provenance_id}, "
                              f"that product needs to have provenance {provprod.id}" )

        if product in self._product_validators():
            self._product_validators[product]( value )

        else:
            if self.product_upstreams[ product ] != NotDefined:
                for upstr in self.product_upstrems[ product ]:
                    if self.products[ upstr ] is None:
                        raise RuntimeError( f"Can't set {product}, need {upstr} to be set first." )
                    if getattr( value, f"{upstr}_id" ) != self.products[ upstr ].id:
                        raise ValueError( f"Can't set {product}, it's {upstr}_id is not consistent with "
                                          f"the id of the existing {upstr}" )

        self.products[ product ] = value


    # ======================================================================

    def read_existing_provenances( self, provtag, match_code_version=True, match_full_code_version=False,
                                   validate_upstream_steps=True, handle_unexpected_processes='ignore',
                                   pgdb=None ):
        match_code_version = match_code_version or match_full_code_version

        provs = Provenance.get_for_tag( provtag )

        if handle_unexpected_processes != 'ignore':
            # ...this is gonna happen, just leave it at ignore
            if not set( provs.keys() ).issubset( set( self.all_steps ) ):
                raise ValueError( f"Provenance tag {provtag} has unexpected processes "
                                  f"{set(provs.keys()) - set(self.all_steps)}" )

        if match_code_version:
            for process in self.all_steps:
                if process in provs:
                    cv = CodeVersion.get_by_id( provs[process].code_version_id, pgdb=pgdb )
                    curcv = Provenance.get_code_version( process, pgdb=pgdb )
                    tomatch = ( ('version_major', 'version_minor', 'version_patch') if match_full_code_version
                                else ('version_major', 'version_minor') )
                    if any( getattr(cv, par) != getattr(curcv, par) for par in tomatch ):
                        raise RuntimeError( f"Code version for process {process} in provenance tag {provtag}"
                                            f"is {cv.version_major}.{cv.version_minor}.{cv.version_patch}, "
                                            f"which doesn't match the current code version for "
                                            f"{curcv.version_major}.{curcv.version_minor}.{curcv.version_patch}" )

        if validate_upstream_steps:
            mismatches = []
            for product, process in self.product_provenance_processes.items():
                if ( ( process is not None ) and ( process in provs ) and
                     ( self.product_upstreams[product] != NotDefined )
                    ):
                    upstr_products = set( self.product_upstreams[product] )
                    expected_upprocs = set( self.product_provenance_processes[p] for p in upstr_products )
                    provupprocs = set( p.process for p in provs[process].upstreams )

                    # OK... if any of the upstream products are starting products, then
                    #   we may not actually know what the process is supposed to be.
                    #   Figure out if we have to worry about this.
                    upstr_starts = set( p for p in upstr_products if p in self.starting_products.keys() )
                    unknown_provs = upstr_starts - set( self.starting_prov.keys() )

                    if len( unknown_provs ) == 0:
                        # Woot.  Somehow we know all the provenances we need to know, so we can do a direct comparsion.
                        if provupprocs != expected_upprocs:
                            mismatches.append( [ product, process, provupprocs, expected_upprocs ] )
                    else:
                        # This is more painful.
                        # First, all the ones that are from this pipeline had better be there
                        knownupprocs = set( self.product_provenance_processes[p]
                                            for p in ( upstr_products - unknown_provs ) )
                        if not knownupprocs.issubset( provupprocs ):
                            mismatches.append( [ product, process, provupprocs, knownupprocs ] )
                        else:
                            # The number of processes that we don't recognize should be what we expect.
                            other_upstrs = provupprocs - knownupprocs
                            if len(other_upstrs) != len(unknown_provs):
                                mismatches.append( [ product, process, provupprocs, knownupprocs ] )

            if len(mismatches) > 0:
                strio = io.StringIO()
                strio.write( f"Provenances found in provenance tag {provtag} did not have upstream provenances "
                             f"with the processes that we exepcted." )
                for uhoh in mismatches:
                    strio.write( f"\n    product {uhoh[0]} process {uhoh[1]}"
                                 f"\n        expected upstreams: {uhoh[3]}"
                                 f"\n        database provenance upstreams: {uhoh[2]}" )
                SCLogger.error( strio.getvalue() )
                raise ValueError( strio.getvalue() )

        return provs


    # ======================================================================

    def run( self, *args, save=NotDefined, save_after_each_step=NotDefined, save_after_steps=NotDefined, **kwargs ):
        """The default run method is probably not good enough; subclasses should override."""

        self.exceptions = []
        
        if save == NotDefined:
            if hasattr( self.pars, "save" ):
                save = self.pars.save
            else:
                save = False
        if save_after_each_step == NotDefined:
            if hasattr( self.pars, "save_after_each_step" ):
                save_after_each_step = self.pars.save_after_each_step
            else:
                save_after_each_step = False
        if save_after_steps == NotDefined:
            if hasattr( self.pars, "save_after_steps" ):
                save_after_steps = self.pars.save_after_steps
            else:
                save_after_steps = []

        just_saved = False
        for step, procobj in self.process_objects.items():
            just_saved = False
            SCLogger.info( f"Running step {step}" )
            procobj( self )
            if save and ( save_after_each_step or ( step in save_after_steps ) ):
                SCLogger.info( f"Saving data products after step {step}" )
                self.save_and_commit()
                just_saved = True

        if save and ( not just_saved ):
            self.save_and_commit()

    # ======================================================================

    def save_and_commit( self ):
        """Save all dataproducts to disk, commit all data products to the database."""

        raise NotImplementedError( "OMG" )
