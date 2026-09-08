import numbers
import uuid

import numpy as np
from psycopg import sql

from models.base import PGDB
from models.provenance import Provenance
import models.object
from models.object import ObjectPosition
from models.reference import Reference
from models.refset import RefSet
from models.source_list import SourceList
from util.config import Config, NoValue
from util.logging import SCLogger
from util.util import listify
from pipeline.parameters import Parameters
from pipeline.data_store import ProvenanceTree


class ParsLightcurve(Parameters):
    def __init__( self, **kwargs ):
        super().__init__()

        self.zp_prov = self.add_par(
            name = 'zp_prov',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance of the zeropoint to use when searching for images to build into the "
                          "lightcurve.  Pass either this or zp_prov_tag; if you pass both, zp_prov_tag is "
                          "ignored." ),
            # Not critical, because the zp provenance will be in the forced photometry provenance upstreams
            critical = False
        )

        self.zp_prov_tag = self.add_par(
            name = 'zp_prov_tag',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance tag for the zeropoint to use when searching for images to build into the "
                          "lightcurve.  Ignored if zp_prov is given.  You must include one of the two." ),
            critical = False
        )

        self.zp_prov_tag_process = self.add_par(
            name = 'zp_prov_tag_process',
            default = 'photocal',
            par_types = str,
            docstring = "The process to use when searching provenance tags for the zeropoint provenance.",
            critical = False
        )
        
        self.object_position_prov = self.add_par(
            name = 'object_position_prov',
            default = None,
            par_types = ( str, None ),
            docstring = ( "Provenance if the object position to use for finding the object's position.  If neither "
                          "this nor object_position_prov_tag is given, will use the raw position from the object." ),
            # Not critical because the positon provenance will be an upstream of the forced phot provenance
            critical = False
        )

        self.object_position_prov_tag = self.add_par(
            name = 'object_position_prov_tag',
            default = None,
            par_types = ( str, None )
            docstring = ( "Provenance tag for object positions to use for finding the object's position.  "
                          "Ignored if object_positon_prov is given." ),
            critical=False
        )

        self.object_position_prov_tag_process = self.add_par(
            name = 'object_position_prov_tag_process',
            default = None,
            par_types = ( str, None ),
            docstring = "The process to use when searching provenance tags for object position provenance.",
            critical = False
        )

        self.only_existing_subtractions = self.add_par(
            name = "only_existing_subtractions",
            default = False,
            par_types = bool,
            docstring = ( "Don't do any new subtractions, only look at existing subtractions to do forced "
                          "photometry on." ),
            critical = False
        )

        # On to actual critical parameters
        
        self.subtraction_config = self.add_par(
            name = "subtraction_config",
            default = {
                'method': 'zogy',
                'refset': None,
                'alignment_index': 'new',
                'alignment': { 'method': 'swarp' },
                'reference': { 'search_by': 'ra/dec',
                               'match_instrument': True,
                               'match_filter': True,
                               'min_overlap': None,
                               'max_dist': 30. / 3600.,
                               'skip_bad': True,
                               'multiple_ok': False,
                               'choice_criteria': [ 'distance', 'unconstrained' ],
                              }
            },
            par_types = dict,
            docstring = ( "A dictionary with subtraction config.  Will override what's in config files "
                          "and defaults for subtraction." )
            # Not critical because a subtraction provenance will be in the upstreams of the forced
            #   phot provenance
            critical = False
        )
        
        self.crop_image = self.add_par(
            name = "crop_image"
            default = [ 100, 100 ],
            par_types = ( list, None ),
            docstring = ( "If given, 2-element list (width, height).  Science images will be trimmed to at most "
                          "this size before being fed to subtractions.  If None, use full-size images "
                          "(which is *usually* not what you want)." )
            critical=True
        )

        # Finally, non-critical parameters that say what to actually do
        
        self.mjd0 = self.add_par(
            name = "mjd0",
            default = None,
            par_types = ( float, None ),
            docstring = "The earliest mjd to do forced photometry for",
            critical = False
        )

        self.mjd1 = self.add_par(
            name = "mjd1",
            default = None,
            par_types = ( float, None ),
            docstring = "The latest mjd to do forced photometry for",
            critical = False
        )

        self.instrument = self.add_par(
            name = "instrument",
            default = None,
            par_types = ( str, None ),
            docstring = "The instrument that we're building a lightcurve for.  Only do one instrument at a time.",
            # Not critical because the forced phot upstream provs will have an image prov that (effectively)
            #   specifies instrument
            critcal = False
        )
        
        self.filters = self.add_par(
            name = "filters",
            default = None,
            par_types = ( list, None ),
            docstring = "Only do forced photometry for these filters (all filters found if not given).",
            critical = False
        )

        self.object_id = self.add_par(
            name = "object_id",
            default = None,
            par_types = ( uuid.UUID, str, None ),
            docstring = "The id of the object to build a lightcurve for.  Specify either this or object_name.",
            critical = False
        )

        self.object_name = self.add_par(
            name = "object_name",
            default = None,
            par_types = ( str, None ),
            docstring ="The name of the object to build a lightcurve for.  Ignored if object_id is given.",
            critical = False
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name( self ):
        return 'lightcurve'


class Lightcurve:
    def __init__( self, **kwargs ):
        """Do forced photometry."""

        cfg = Config.get()

        self.pars = ParsPipeline( **(cfg.value('lightcurve', {})) )
        self.pars.augment( kwargs )

        subtraction_config = cfg.value( 'subtraction', {} )
        subtraction_config.update( self.pars.subtraction_config )
        self.subtractor = Subtractor( **subtraction_config )

        self.object = None
        self.object_position_prov = None
        self.object_position = None
        self.zp_prov = None
        self.refset = None
        self.refs = {}
        
        
    def setup( self, object_id=NoValue(), object_name=NoValue(), mjd0=NoValue(),
               mjd1=NoValue(), filters=NoValue(), pgdb=None ):
        pgdb_in = pgdb
        
        self.pars.object_id = object_id if not isinstance( object_id, NoValue ) else self.pars.object_id
        self.pars.object_name = object_name if not isinstance( object_name, NoValue ) else self.pars.object_name
        self.pars.mjd0 = mjd0 if not isinstance( mjd0, NoValue ) else self.pars.object_mjd0
        self.pars.mjd1 = mjd1 if not isinstance( mjd1, NoValue ) else self.pars.mjd1
        self.pars.filters = listify(filters) if not isinstance( filters, NoValue ) else self.pars.filters
        
        self.object = None
        if self.pars.object_id is not None:
            objcol = "_id"
            objval = self.pars.object_id
            if self.pars.object_name is not None:
                SCLogger.warning( "Gave both object_id and object_name, ignoring object_name" )
        elif self.pars.object_name is not None:
            objcol = "name"
            objval = self.pars.object_name
        else:
            raise ValueError( "Must give either object_id or object_name" )

        if self.pars.crop_image is not None:
            if ( ( len(self.pars.crop_image) != 2 ) or
                 ( not all ( isinstance(x, numbers.Integral) for x in self.pars.crop_image ) ) ):
                raise ValueError( f"Must give two integer values for crop_image, got {self.pars.crop_image}" )

        with PGDB( pgdb_in, dictcursor=True ) as pgdb:
            if ( self.zp_prov is None ) or ( self.zp_prov.id != self.pars.zp_prov ):
                self.zp_prov = None
                if self.pars.zp_prov is not None:
                    self.zp_prov = Provenance.get( self.pars.zp_prov, pgdb=pgdb )
                elif self.pars.zp_prov_tag is not None:
                    self.zp_prov = Provenance.get_for_tag( self.pars.zp_prov_tag, self.pars.zp_prov_tag_process,
                                                           pgdb=pgdb )

            if self.zp_prov is None:
                raise RuntimeError( f"Could not find a zeropoint provenance to use to find images. "
                                    f"zp_prov={self.pars.zp_prov}, zp_prov_tag={self.pars.zp_prov_tag}, "
                                    f"zp_prov_tag_process={self.pars.zp_prov_tag_process}" )

            if ( self.pars.object_position_prov is None ) and ( self.pars.object_position_prov_tag is None ):
                self.object_position_prov = None
            else:
                if self.pars.object_position_prov is not None:
                    if self.pars.object_position_prov_tag is None:
                        SCLogger.warning( "Both object_position_prov and object_position_prov_tag given; "
                                          "ignoring the latter." )
                    if ( ( self.object_position_prov is None ) or
                         ( self.object_position_prov.id != self.pars.object_position_prov )
                        ):
                        self.object_position_prov = Provenance.get( self.pars.object_position_prov, pgdb=pgdb )
                        if self.object_position_prov is None:
                            raise ValueError( f"Could not find object position provenance "
                                              f"{self.pars.object_position_prov}" )
                else:
                    self.object_position_prov = Provenance.get_by_tag( self.pars.object_position_prov_tag,
                                                                       self.pars.object_position_prov_tag_process,
                                                                       pgdb=pgdg_in )
                    if self.object_position_prov is None:
                        raise ValueError( f"Could not find object position provenance for "
                                          f"tag {self.pars.object_position_prov_tag} and "
                                          f"process { self.pars.object_position_prov_tag_process}" )

            if self.subtractor.refset is None:
                raise ValueError( "Subtractor has no refset defined!" )
            self.refset = Refset.get_by_name( self.subtractor.pars.refset, pgdb=pgdb )
            if self.refset is None:
                raise ValueError( f"Can't find refset {self.subtractor.pars.refset}" )
                    
            rows = pgdb.execute( sql.SQL( "SELECT * FROM objects WHERE {col}={val}" )
                                 .format( col=sql.Identifier(objcol), val=objval  ) )
            if len(rows) > 0:
                raise RuntimeError( "This should never happen" )
            elif len(rows) == 0:
                raise ValueError( f"Could not find object with {col}={objval}" )
            else:
                self.object = models.object.Object( **(rows[0]) )

            self.object_position = None
            if self.object_position_prov is not None:
                rows = pgdb.execute( sql.SQL( "SELECT * FROM object_positions "
                                              "WHERE object_id={objid} AND provenance_id={provid} "
                                             ).format( objid=self.object.id,
                                                       provid=self.object_position_prov.id ) )
                if len(rows) > 0:
                    raise RuntimeError( "This should never hapen, I don't think, but I'm not really sure." )
                elif len(rows) == 0:
                    raise ValueError( f"Could not find object position for object {self.object.id} "
                                      f"and object position provenacne {self.object_position_prov.id}" )
                else:
                    self.object_position = ObjectPosition( **(rows[0]) )

                self.ra = self.object_position.ra
                self.dec = self.object_position.dec
            else:
                self.ra = self.object.ra
                self.dec = self.object.dec


    def make_prov_tree( self, just_read=False, save=True,
                        provtag=None, process=None,
                        zp_provtag=None, zp_process=None,
                        pos_provtag=None, pos_process=None,
                        sub_provtag=None, sub_process=None,
                        trim_provtag=None, trim_processes=None,
                        pgdb=None ):
        # Build a full provenance tree for DataStore to chew on
        # DataStore.make_prov_tree is designed for use with top_level, and is
        #   not easy to use here, so just make one manually.

        pgdb_in = pgdb
        
        process = process if process is not None else self.pars.forced_phot_prov
        zp_process = zp_process if zp_process is not None else self.pars.zp_prov_tag_process
        pos_process = pos_process if pos_process is not None else self.pars.object_position_prov_tag_process
        sub_process = sub_process if sub_process is not None else 'subtraction'
        trim_processes = ( trim_processes if trim_processes is not None
                           else [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp' ] )

        # The arduous process of reading all the provenances from the
        #   database, recursively trolling upstreams

        def _append_provs( prov, provs, upstream_procs, procs, pgdb ):
            # Special case handling for referencing and positioning
            if prov.process in ( 'positioning', pos_process, 'referencing' ):
                expected_upstreams = []
            else:
                expected_upstreams = [ p.process for p in prov.get_upstreams( pgdb=pgdb, save_to_object=True ) ]

            if prov.process in provs.keys():
                if prov.id != provs[prov.process].id:
                    raise RuntimeError( f"Process {prov.process} came up with inconsistent values "
                                        f"when bulding the provenance tree!" )
                if upstream_procs[prov.process] != expected_upstreams:
                    raise RuntimeError( f"Process {prov.process} came up with inconsistent upstream "
                                        f"processes when building the provenance tree!" )
            else:
                provs[prov.process] = prov
                upstream_procs[prov.process] = expected_upstreams

            if prov.process not in ( 'positioning', pos_process, 'referencing' ):
                for upproc in prov.upstreams:
                    _append_provs( upproc, provs, upstream_procs, procs, pgdb )

        db_provs = {}
        db_upstream_procs = {}
        must_have_procs = { 'starting_point', 'extraction', 'astrocal', zp_process, 'referencing', sub_process }
        allowed_procs = must_have_procs.union( set( trim_processes ) )
        allowed_procs.add( pos_process )

        with PGDB( pgdb_in ) as pgdb:
            # First, see if we can find the forced photometry tag
            if ( provtag is None ) and ( self.pars.forced_phot_prov is not None ):
                found_prov = Provenance.get( self.pars.forced_phot_prov, pgdb=pgdb )
            elif provtag is not None:
                found_prov = Provenance.get_for_tag( provtag, process, pgdb=pgdb )
            else:
                found_prov = None

            if found_prov is not None:
                _append_provs( found_prov, db_provs, db_upstream_procs, procs, pgdb )
                must_have_procs.add( process )
                
            else:
                if just_read:
                    raise RuntimeError( "just_read is true, but could not find forced photomtery provenance for "
                                        "provenance tag {provtag} and process {process}" )

                # OK... didn't find an existing forcedphot provenance so try to read as much as we can
                # First, there's *gotta* be a reference provenances, or we won't be able to do anything
                db_provs['referencing'] = Provenance.get_by_id( self.refset.provenance_id, pgdb=pgdb )
                db_upstream_procs['referencing'] = {}
                if db_provs['referencing'] is None:
                    raise RuntimeError( f"Failed to find provenance {self.refset.provenance_id} for "
                                        f"refset {self.refset.name}" )

                # Likewise, there must be a zeropoint provenance
                if zp_provtag is not None:
                    zpprov = Provenance.get_for_tag( zp_provtag, zp_process, pgdb=pgdb )
                    if zpprov is None:
                        raise RuntimeError( f"Failed to find provenance for passed "
                                            f"tag {zp_provtag} process {zp_process}" )
                elif self.pars.zp_prov is not None:
                    zpprov = Provenance.get( self.pars.zp_prov, pgdb=pgdb )
                    if zpprov is None:
                        raise RuntimeError( f"Failed to find zeropoint provenance {self.pars.zpprov}" )
                elif self.pars.zp_prov_tag is not None:
                    zpprov = Provenance.get_for_tag( self.pars.zp_prov_tag, zp_process, pgdb=pgdb )
                    if zpprov is None:
                        raise RuntimeError( f"Failed to find provenance for configured tag "
                                            f"{self.pars.zp_prov_tag} process {zp_process}" )
                else:
                    raise ValueError( f"Must pass zp_provtag, or must configure zp_prov or zp_prov_tag" )

                # Get the zeropoint prov upstreams; they should exist, since the zeropoint provenance does!
                _append_provs( zpprov, db_provs, db_upstream_procs, procs, pgdb )
                
                # Get the object position provenance if any
                if pos_provtag is not None:
                    db_provs[pos_process] = Provenance.get_for_tag( pos_provtag, pos_process, pgdb=pgdb )
                elif self.pars.object_position_prov is not None:
                    db_provs[pos_process] = Provenance.get_by_id( self.pars.object_position_prov, pgdb=pgdb )
                elif self.pars.object_position_prov_tag is not None:
                    db_provs[pos_process] = Provenance.get_for_tag( self.pars.object_position_prov_tag,
                                                                    pos_process, pgdb=pgdb )
                if pos_process in db_provs:
                    if db_provs[pos_process] is None:
                        raise RuntimeError( f"Failed to find object positioning provenance (process {posproc})" )
                    db_upstream_procs[posproc] = []

                # Get the subtraction provenance if any
                if sub_provtag is not None:
                    subprov = Provenance.get_for_tag( sub_provtag, sub_process, pgdb=pgdb )
                elif self.pars.subtraction_prov_tag is not None:
                    subbprov = Provenance.get_for_tag( self.pars.subtraction_prov_tag, sub_process, pgdb=pgdb )

                if subbprov is not None:
                    # Add the subtraction provenacne and its upstreams.  Most (all?) of the upstreams
                    #   will have already been added above, but _append_provs (supposedly) handles that.
                    _append_provs( subprov, db_provs, db_upstream_procs, procs, pgdb )

            # Make sure stuff we read out of the database has processes we expected
            have_procs = set( db_provs.keys() )
            missing_procs = zero_offset_procs - have procs
            unknown_procs = have_procs - allowed_procs
            if ( len(missing_procs) > 0 ) or ( len(unknown_procs) > 0 ):
                raise RuntimeError( f"Failure trawling the database for provenances, unexpected processes.  "
                                    f"missing: {missing_procs} ; unknown: {unknown_procs}" )

            if just_read:
                # We're done
                return ProvenanceTree( db_provs, db_upstream_procs )

            else:
                # OK!  db_provs now has all known provenances, including at *least* referencing and zeropoint
                # Make the provenances for the things this pipeline will create; if they were found in
                #   the database, make sure they match.

                upstream_steps = db_upstream_procs
                provs = db_provs
                subups = [ self.refset.provenance ]
                subupsteps = [ 'referencing' ]

                # Get trim image provenances
                
                trimprovs = None
                if self.pars.crop_image is not None:
                    trimupsteps = { trim_processes[0]: [ 'starting_point', 'astrocal' ],
                                    trim_processes[1]: [ 'Image.trim' ],
                                    trim_processes[2]: [ trimprocs[1], 'astrocal' ],
                                    trim_processes[3]: [ zp_process ] ] )

                    trimprovs = Image.get_trim_provs( self.pars.crop_image[0], self.pars.crop_image[1],
                                                      upstreams=[ provs['starting_point'] ],
                                                      wcs_prov=provs['astrocal'], zp_prov=provs[zp_process],
                                                      save=save, provtag=provtag, pgdb=pgdb )
                    trimprovs.append( Provenance( process=zp_process, upstreams=[ provs[zp_process] ] ) )
                        
                    if trim_processes[0] in provs:
                        if any( provs[trim_processes[i]].id != trimprovs[i].id for i in range(4) ):
                            raise ValueError( "Pre-existing image trim provenances don't match what what "
                                              "they should have been given config." )
                        if any( set( trimupsteps[trim_processes[i]] ) != set( upstream_steps[trim_proceses[i]] )
                                for i in range(4) ):
                            raise ValueError( "Pre-existing trim upstream steps weren't what was expected." )
                    else:
                        # This next if should be False by construction.  If it's True, it
                        #   means that there is a code error either here or in Image.get_trim_provs
                        if ( len( trimprovs[0].upstreams == 2 ) != 0
                             or ( 'astrocal' not in [ p.process for p in trimprovs[0].upstreams ] )
                             or  any( set( u.process for u in trimprovs[i].upstreams )
                                      != set( trimupsteps[trim_processes[i]] ) )
                                      for i in (1, 2, 3) )
                            ):
                            raise RuntimeError( "I am surprised." )
                        upstream_steps.update( trimupsteps )
                        for i in range(4):
                            provs[ trim_processes[i] ] = [ trimprovs[i] ]
                            if save:
                                trimprovs[i].insert( pgdb=pgdb, nocommit=True )
                        if save:
                            pgdb.commit()

                    subups.append( trimprovs[0] )
                    subupsteps.extend( [ trim_processes[0], trim_processes[3] ] )

                else:
                    subups.append( provs[zp_process] )
                    subupsteps.append( zp_process )

                # Get subtraction provenance
                subprov = Provenance( code_version_id=Provenance.get_code_version('subtraction', pgdb=pgdb).id,
                                      process=sub_process,
                                      parameters=self.subtractor.pars.get_critical_pars(),
                                      upstreams=subups )
                if sub_process in provs:
                    if subprov.id != provs[sub_process].id:
                        raise ValueError( f"Found provenance for subtraction {provs[sub_process].id} does not "
                                          f"match what this pipeline will create {subprov.id}" )
                    if set( subupsteps ) != set( upstream_steps[sub_process] ):
                        raise ValueError( f"Subtraction upstream steps mismatch" )
                else:
                    provs[sub_process] = subprov
                    upstream_stemps[sub_process] = subupsteps
                    if save:
                        subprov.insert( pgdb=pgdb )
                
                # Get the forced photometry provenance
                ups = [ provs[sub_process] ]
                upsteps = [ sub_process ]
                if pos_process in provs:
                    ups.append( provs[pos_process] )
                    upsteps.append( pos_process )
                forcedprov = Provenance( code_version_id=Provenance.get_code_version('forcedphot', pgdb=pgdb).it,
                                         process=process, parameters=self.pars.get_critical_pars(),
                                         upstreams_ups )
                if process in provs:
                    if forcedprov.id != provs[process].id:
                        raise ValueError( f"Found provenance for forced photometry {provs[process].id} does not "
                                          f"match what this pipeline will create {forcedprov.id}" )
                    if set( upsteps ) != set( upstream_steps[process] ):
                        raise ValueError( "Forced photometry upstream steps mismatch" )
                else:
                    provs[process] = forcedprov
                    upstream_steps[process] = upsteps
                    if save:
                        forcedprov.save( pgdb=pgdb )

                return ProvenanceTree( provs, upstream_steps )

        raise RuntimeError( "This should never happen." )


    def find_refs( self, ds, filters=None, mjd0=None, mjd1=None, pgdb=None ):
        if self.object_position is not None:
            ra = self.object_position.ra
            dec = self.object_position.dec
        else:
            ra = self.object.ra
            dec = self.object.dec

        kwargs = self.pars.reference.copy()
        kwargs['instrument'] = self.pars.instrument
        kwargs['provenances'] = self.refset.provenance_id
        kwargs['ra'] = ra
        kwargs['dec'] = dec
        kwargs['mjd0'] = mjd0
        kwargs['mjd1'] = mjd1

        if any( x in kwargs for x in ( 'must_match_section', 'must_match_target' ) ):
            raise ValueError( "Don't use must_match_section or must_match_target in subtraction_conifg['reference']" )

        refs = {}
        for filt in filters:
            ref = ds.get_reference( ra=ra, dec=dec, filter=filt, pgdb=pgdb, **kwargs )
            if ref is None:
                raise RuntimeError( f"Cannot find a reference at ({ra:.rf, dec:.4f}) for instrument "
                                    f"{instrument}, filter {filt}, and parameters {kwargs}" )
            refs[filt] = ref

        return refs


    def run( self ):
        provtree = self.make_prov_tree( save=True )


        
        if self.pars.filter is None:
            imgs, wcsen, zps = Image.find_images( ra=self.ra, dec=self.dec, type='Sci',
                                                  provenance_ids=provtree['photocal'], provenance_ids_are_zp=True,
                                                  instrument=self.pars.instrument,
                                                  min_mjd=self.pars.mjd0, max_mjd=self.pars.mjd1,
                                                  order_by='earliest', return_wcs=True, return_zeropoints=True )
            filters = set( i.filter for i in imgs )
        else:
            imgs = []
            wcsen = {}
            zps = {}
            filters = self.pars.filter
            for filt in filters:
                thisimgs, thiswcsen, thiszps = Image.find_images( ra=self.ra, dec=self.dec, type='Sci',
                                                                  provenance_ids=provtree['photocal'],
                                                                  provenance_ids_are_zp=True,
                                                                  instrument=self.pars.instrument,
                                                                  min_mjd=self.pars.mjd0, max_mjd=self.pars.mjd1,
                                                                  order_by='earliest',
                                                                  return_wcs=True, return_zeropoints=True )
                imgs.extend( thisimgs )
                wcsen.update( thiswcsen )
                zps.update( thiszps )
                imgs.sort( key=lambda x: x.mjd )
        
        if len(imgs) == 0:
            SCLogger.warning( "No images found to build a lightcurve for!" )
            return None

        # Make an empty datastore to do use for finding references.  (Issue #550)
        ds = DataStore()
        ds.prov_tree = provtree
        refs = self.find_refs( ds, filters=filters, mjd0=imgs[0].mjd, mjd1=imgs[1].mjd )
        
        for img in imgs:
            ds = DataStore( image )
            ds.prov_tree = provtree
            ds.reference = refs[ img.filter ]
            ds.sources = SourceList.get_by_id( wcsen[img.id].sources_id )
            ds.wcs = wcsen[ img.id ]
            ds.zp = zps[ img.id ]

            # Trim if we have to
            if self.pars.crop_image is not None:
                xctr, yctr = ds.wcs.wcs.pixel_to_world_values( self.ra, self.dec )
                ixctr = int( np.floor( xctr + 0.5 ) )
                iyctr = int( np.floor( yctr + 0.5 ) )
                x0 = ixctr - ( self.pars.crop_image[0] / 2 )
                x1 = x0 + self.pars.crop_image[0]
                y0 = iyctr - ( self.pars.crop_image[1] / 2 )
                y1 = y0 + self.pars.crop_image[1]
                ( cropim, cropsrc,
                  cropwcs, cropprovs ) = img.trim( x0, x1, y0, y1,
                                                   sources=ds.sources, wcs=ds.wcs,
                                                   trimprovs=[ provtree['Image.trim'],
                                                               provtree['Image.trim.sources'],
                                                               provtree['Image.trim.wcs']
                                                              ] )
                cropbg = ds.get_bg().trim( x0, x1, y0, y1, trimmed_sources=cropsrc )
                croppfs = ds.get_psf().trim( x0, x1, y0, y1, trimmed_sources=cropsrc )
                ds.get_zp()
                cropzp = ZeroPoint( zp=ds.zp.zp, dzp=ds.zp.dzp, aper_cor_radii=ds.zp.aper_cor_radii,
                                    aper_cors=ds.aper_cors.copy(), provenacne_id=provtree )
                                    

                
                ds.image = cropim
                ds.image_id = cropim.id
                ds.sources = crompsrc
                ds.bg = cropbg
                ds.psf = croppsf
                ds.wcs = cropwcs
                ds.zp = ZeroPoint( wcs_id=cropwcs.id, zp=ds.zp.ap, aper_cor_radii=ds.zp.aper_cor_radii,
                                   aper_cors=ds.zp.aper_cors, provenance_id=provtree[zp_process].id )
                
            ds = self.subtractor.run( ds, self.ra, self.dec, trust_datastore_reference=True )
            ds.save_and_commit( overwrite=False )
            
            # Now actually do photometry
            # First, make things the way photutils wants them
            sub_image = ds.get_sub_image()
            sub_mask = np.full_like( sub_image.flags, False, dtype=bool )
            sub_mask[ sub_image.flags != 0 ] = True
            sub_mask[ sub_image.weight <= 0. ] = True
            sub_noise = 1. /np.sqrt( sub_image.weight )
            sub_noise[ sub_mask ] = np.nan

            new_zp = ds.get_zp()
            new_wcs = ds.get_wcs()
            # TODO FIGURE THIS OUT (Issue #194)
            new_psf = ds.get_psf()

            measurements = photometry( sub_image, sub_noise, sub_mask, positions=[(xctr, yctr)],
                                       pfsobj=newpsf, apers=new_zp.aper_cor_radii )

            # ROB YOU WERE HERE
            
            
