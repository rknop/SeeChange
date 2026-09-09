import numbers
import uuid

import numpy as np
from psycopg import sql

import improc
import models.object
from models.base import PGDB
from models.provenance import Provenance
from models.object import ObjectPosition
from moels.image import Image
from models.source_list import SourceList
from models.zero_point import ZeroPoint
from models.refset import RefSet
from util.config import Config, NoValue
from util.logging import SCLogger
from util.util import listify
from pipeline.parameters import Parameters
from pipeline.data_store import ProvenanceTree, DataStore
from pipeline.subtration import Subtractor


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
            par_types = ( str, None ),
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
                          "and defaults for subtraction." ),
            # Not critical because a subtraction provenance will be in the upstreams of the forced
            #   phot provenance
            critical = False
        )

        self.crop_image = self.add_par(
            name = "crop_image",
            default = [ 100, 100 ],
            par_types = ( list, None ),
            docstring = ( "If given, 2-element list (width, height).  Science images will be trimmed to at most "
                          "this size before being fed to subtractions.  If None, use full-size images "
                          "(which is *usually* not what you want)." ),
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

        self.pars = ParsLightcurve( **(cfg.value('lightcurve', {})) )
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
                                                                       pgdb=pgdb_in )
                    if self.object_position_prov is None:
                        raise ValueError( f"Could not find object position provenance for "
                                          f"tag {self.pars.object_position_prov_tag} and "
                                          f"process { self.pars.object_position_prov_tag_process}" )

            if self.subtractor.refset is None:
                raise ValueError( "Subtractor has no refset defined!" )
            self.refset = RefSet.get_by_name( self.subtractor.pars.refset, pgdb=pgdb )
            if self.refset is None:
                raise ValueError( f"Can't find refset {self.subtractor.pars.refset}" )

            rows = pgdb.execute( sql.SQL( "SELECT * FROM objects WHERE {col}={val}" )
                                 .format( col=sql.Identifier(objcol), val=objval  ) )
            if len(rows) > 0:
                raise RuntimeError( "This should never happen" )
            elif len(rows) == 0:
                raise ValueError( f"Could not find object with {objcol}={objval}" )
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


    def _generate_provenances( self, provtree, pgdb=None ):
        subups = [ provtree['referencing'] ]
        subupsteps = [ 'referencing' ]

        # Get trim image provenances
        if self.pars.crop_image is not None:
            trim_processes = [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp' ]
            trimprovs = Image.get_trim_provs( self.pars.crop_image[0], self.pars.crop_image[1],
                                              upstreams=[ provtree['starting_point'] ],
                                              wcs_prov=provtree['astrocal'], zp_prov=provtree['photocal'],
                                              save=False )
            trimprovs.append( Provenance( process='photocal', upstreams=[ provtree['photocal'] ] ) )

            if 'Image.trim' in provtree:
                if any( provtree[trim_processes[i]].id != trimprovs[i].id for i in range(4) ):
                    raise ValueError( "Pre-existing image trim provenances don't match what what "
                                      "they should have been given config." )
                trimupsteps = { 'Image.trim':        [ 'starting_point', 'astrocal' ],
                                'Image.trm.sources': [ 'Image.trim' ],
                                'Image.trim.wcs':    [ 'Image.trim.sources' ],
                                'Image.trim.zp':     [ 'photocal' ] }
                if any( set( trimupsteps[trim_processes[i]] ) != set( provtree.upstream_steps[trim_processes[i]] )
                        for i in range(4) ):
                    raise ValueError( "Pre-existing trim upstream steps weren't what was expected." )
            else:
                # This next if should be False by construction.  If it's True, it
                #   means that there is a code error either here or in Image.get_trim_provs
                if ( len( trimprovs[0].upstreams == 2 ) != 0
                     or ( 'astrocal' not in [ p.process for p in trimprovs[0].upstreams ] )
                     or  any( set( u.process for u in trimprovs[i].upstreams )
                              != set( trimupsteps[trim_processes[i]] )
                              for i in (1, 2, 3) )
                    ):
                    raise RuntimeError( "I am surprised." )
            # Gotta include both Image.trim.wcs and
            # Image.trim.zp because Image.trim.zp has only
            # photocal as an upstream, so we don't tag the image
            # provenance by just tagging the Image.trim.zp
            # provenance.
            subups.append( [ trimprovs[2], trimprovs[3] ] )
            subupsteps.extend( [ 'Image.trim.wcs', 'Image.trim.zp' ] )

        else:
            trimprovs = None
            subups.append( provtree['photocal'] )
            subupsteps.append( 'photocal' )

        # Get subtraction provenance
        subprov = Provenance( code_version_id=Provenance.get_code_version('subtraction', pgdb=pgdb).id,
                              process='subtraction',
                              parameters=self.subtractor.pars.get_critical_pars(),
                              upstreams=subups )
        if 'subtraction' in provtree:
            if subprov.id != provtree['subtraction'].id:
                raise ValueError( f"Found provenance for subtraction {provtree['subtraction'].id} does not "
                                  f"match what this pipeline will create {subprov.id}" )
            if set( subupsteps ) != set( provtree.upstream_steps['subtraction'] ):
                raise ValueError( "Subtraction upstream steps mismatch." )

        # Get the forced photometry provenance
        ups = [ provtree['subtraction'] ]
        upsteps = [ 'subtraction' ]
        if ( self.objectposition_prov is not None ) or ( self.object_position_prov_tag is not None ):
            ups.append( provtree['positioning'] )
            upsteps.append( 'positioning' )
        forcedprov = Provenance( code_version_id=Provenance.get_code_version('forcedphot', pgdb=pgdb).it,
                                 process='forcedphot', parameters=self.pars.get_critical_pars(),
                                 upstreams=ups )
        if 'forcedphot' in provtree:
            if forcedprov.id != provtree['forcedphot'].id:
                raise ValueError( f"Found provenance for forced photometry {provtree['forcedphot'].id} does not "
                                  f"match what this pipeline will create {forcedprov.id}" )
            if set( upsteps ) != set( provtree.upstream_steps['forcedphot'] ):
                raise ValueError( "Forced photometry upstream steps mismatch." )

        return forcedprov, subprov, trimprovs


    def make_prov_tree( self, just_read=False, save=True, save_tag=True, provtag=None, pgdb=None,
                        ok_if_preexisting_prov_tag_without_forcedphot=False ):
        """Make a provenance tree for the Lightcurve.

        Datastore.make_prov_tree is designed specifically for use with
        top_level, and is not easy to use here.  Probably that code
        should be moved to top_level.py.

        Parameters
        ----------
           just_read : bool, default False
              Normally, the provenance tree will be generated looking at
              the parameters attached to the Lightcurve object, and
              attached to the Subtractor object that the Lightcurve
              object makes.  Provenances will be generated for all of
              forcedphot, subtraction, Image.trim, Image.trim.sources,
              Image.trm.wcs, and Image.trim.zp.  If just_read is true,
              all of that is thrown out, and instead the provenances are
              read from the database using provtag.  WARNING: if you do
              this, then don't generate new forced photometry, just read
              what's there!

           save : bool, default True
              Save any generated provenances to the database?  Must be
              False if just_read is True.

           provtag : str, default None
              The provenance tag to use to find existing subtraction and
              forcedphot provenances.  If save and save_tags are both
              true, than any newly generated provenacnes will be tagged
              with this provenacne tag.  If just_read is False, then if
              preexisting provenances in the database with this tag are
              inconsistent with the ones generated using the object's
              configured parameters, an exception will be raised.

           save_tag : bool, default True
              Ignored if save is False.  If True, then all provenances
              saved to the database are also tagged with the provenance
              tag given in provtag.

           pgdb : base.PGDB, default None
              A database connection.  If not given, one will be created
              and closed as necesary.

        Returns
        -------
          data_store.ProvenanceTree

        """

        if just_read and save:
            raise ValueError( "Can't use just_read and save together." )

        if just_read and ( provtag is None ):
            raise ValueError( "just_read requires provtag" )

        if save_tag and ( not save ):
            SCLogger.warning( "save_tag is True but save is False, ignoring save_tag." )

        # Build a full provenance tree for DataStore to chew on
        # DataStore.make_prov_tree is designed for use with top_level, and is
        #   not easy to use here, so just make one manually.

        pgdb_in = pgdb

        # Read any existing provenances from the databaes.  First make some sets
        #   of what we must have to do anything, and what is allowed.

        must_have_procs = { 'starting_point', 'extraction', 'astrocal', 'photocal', 'referencing' }
        all_procs = must_have_procs.union( { 'subtraction', 'forcedphot' } )
        trim_procs = [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp']

        provtree = ProvenanceTree( noupstreams=['positioning', 'referencing', 'starting_point'],
                                   processmap={'preprocessing': 'starting_piont'} )
        with PGDB( pgdb_in ) as pgdb:
            # First, see if we can find the forced photometry tag
            if provtag is not None:
                found_prov = Provenance.get_for_tag( provtag, 'forcedphot', pgdb=pgdb )
            else:
                found_prov = None

            if found_prov is not None:
                # This will build the whole tree, adding all the upstreams
                provtree.append_provenance( found_prov, pgdb=pgdb )
                if not just_read:
                    # If we're not just reading, then we know which optional processes should be there
                    if self.pars.crop_image is not None:
                        all_procs = all_procs.union( trim_procs )
                    if ( ( self.pars.object_position_prov is not None ) or
                         ( self.pars.object_position_prov_tag is not None )
                        ):
                        all_procs.add( 'positioning' )
                    must_have_procs = all_procs
                else:
                    # Generate expected provenances for later validation
                    forcedprov, subprov, trimprovs = self._generate_provenances( provtree )

            else:
                if just_read:
                    raise RuntimeError( "just_read is true, but could not find forced photometry provenance for "
                                        "provenance tag {provtag}" )

                # Based on config, we know what processes are legal
                if self.pars.crop_iamge is not None:
                    all_procs = all_procs.union( trim_procs )
                if ( ( self.pars.object_position_prov is not None ) or
                     ( self.pars.object_position_prov_tag is not None )
                    ):
                    all_procs.add( 'positioning' )

                # OK... didn't find an existing forcedphot provenance so try to read as much as we can
                # First, there's *gotta* be a reference provenances, or we won't be able to do anything
                # (Use Provenance.get_by_id here rather than self.refset.provenance property, so that
                # we can use pgdb.)
                refprov = Provenance.get_by_id( self.refset.provenance_id, pgdb=pgdb )
                if refprov is None:
                    raise RuntimeError( f"Failed to find provenance {self.refset.provenance_id} for "
                                        f"refset {self.refset.name}" )
                provtree.append_provenance( refprov, pgdb=pgdb )

                # Likewise, there must be a zeropoint provenance
                if self.pars.zp_prov is not None:
                    zpprov = Provenance.get( self.pars.zp_prov, pgdb=pgdb )
                elif self.pars.zp_prov_tag is not None:
                    zpprov = Provenance.get_for_tag( self.pars.zp_prov_tag, 'photocal', pgdb=pgdb )
                else:
                    raise ValueError( "Must have one of zp_prov or zp_prov_tag" )
                if zpprov is None:
                    raise RuntimeError( f"Failed to find the zeropoint provenance for "
                                        f"zp_prov={self.pars.zp_prov} and zp_prov_tag={self.pars.zp_prov_tag}" )
                elif zpprov.process != 'photocal':
                    raise RuntimeError( f"zeropoint provenance process is {zpprov.process}, "
                                        f"expected 'photocal'" )

                # This will also add the astrocal, soruces, and preprocessing (starting_point) provenances
                provtree.append_provenance( zpprov, pgdb=pgdb )

                # Get the object position provenance if any
                posprov = None
                notfound = False
                if self.pars.object_position_prov is not None:
                    posprov = Provenance.get_by_id( self.pars.object_position_prov, pgdb=pgdb )
                    if posprov is None:
                        notfound = True
                    elif posprov.process != 'positioning':
                        raise ValueError( f"The process of provenance {self.pars.object_positon_prov} is "
                                          f"{posprov.process}, but should be 'positioning'." )
                elif self.pars.object_position_prov_tag is not None:
                    posprov = Provenance.get_for_tag( self.pars.object_position_prov_tag, 'positioning', pgdb=pgdb )
                    notfound = posprov is None
                if notfound:
                    raise RuntimeError( f"Failed to find object positioning provenance given "
                                        f"object_position_prov={self.pars.object_position_prov} and "
                                        f"object_position_prov_tag={self.pars.object_position_prov_tag}" )
                if posprov is not None:
                    provtree.append_provenance( posprov, pgdb=pgdb )

                if provtag is not None:
                    # Get the trim provenances if any
                    found_trimprovs = []
                    for proc in [ 'Image.trim', 'Image.trim.sources', 'Image.trim.wcs', 'Image.trim.zp' ]:
                        found_trimprovs.append( Provenance.get_for_tag( provtag, proc, pgd=pgdb ) )
                    if any( i is not None for i in found_trimprovs ):
                        if not all( i is not None for i in found_trimprovs ):
                            raise RuntimeError( f"Database corruption: found some, but not all, image trim provs "
                                                f"in provtag {provtag}" )
                        for prov in found_trimprovs:
                            provtree.append_provenance( prov, pgdb=pgdb )

                    # Get the subtraction provenance if any
                    found_subprov = Provenance.get_for_tag( provtag, 'subtraction', pgdb=pgdb )
                    if found_subprov is not None:
                        # Add the subtraction provenacne and its upstreams.  Most (all?) of the upstreams
                        #   will have already been added above, but _append_provs (supposedly) handles that.
                        provtree.append_provenance( found_subprov, pgdb=pgdb )

                # ...and we don't need to get the forced phot prov here because we wouldn't be
                #   inside this "else" if it could be found.

            # Make sure stuff we read out of the database has processes we expected
            have_procs = set( provtree.keys() )
            missing_procs = must_have_procs - have_procs
            unknown_procs = have_procs - all_procs
            if ( len(missing_procs) > 0 ) or ( len(unknown_procs) > 0 ):
                raise RuntimeError( f"Failure building provtree, unexpected processes.  "
                                    f"missing: {missing_procs} ; unknown: {unknown_procs}" )

            # If just_read is True, then we're done!
            if not just_read:
                # OK!  provtree now has all known provenances, including at *least* referencing and zeropoint
                # Make the provenances for the things this pipeline will create; if they were found in
                #   the database, make sure they match.

                forcedprov, subprov, trimprovs = self._generate_provenances( provtree, pgdb=pgdb )

                # Add the generated provenances to the provenance tree.  Do this piece by piece,
                #   so that self-consistency will be checeked.  (It does mean redundant database
                #   queries... I think.)
                if trimprovs[0] is not None:
                    for p in trimprovs:
                        provtree.append_provenance( p, nodb=True )
                    must_have_procs = must_have_procs.union( trim_procs )
                provtree.append_provenance( subprov, nodb=True )
                provtree.append_provenance( forcedprov, nodb=True )
                must_have_procs = must_have_procs.union( { 'subtraction', 'forcedphot' } )

                # So... the provenance three thinks it's self consistent.  Let's check again
                #    that the expected provenances are there, and they should ALL be there now.
                have_procs = set( provtree.keys() )
                missing_procs = must_have_procs - have_procs
                unknown_procs = have_procs - must_have_procs
                if ( len(missing_procs) > 0 ) or ( len(unknown_procs) > 0 ):
                    raise RuntimeError( f"Failure trawling the database for provenances, unexpected processes.  "
                                        f"missing: {missing_procs} ; unknown: {unknown_procs}" )


                # Save them if necessary.  Do this in the right order so upstreams exist.
                if save:
                    provs = []
                    if self.pars.crop_image is not None:
                        provs.extend( provtree[p] for p in [ 'Image.trim', 'Image.trim.sources',
                                                             'Image.trim.wcs', 'Image.trim.zp' ] )
                    provs = [ provtree['subtraction'], provtree['forcedphot'] ]
                    for prov in provs:
                        prov.insert_if_needed( pgdb=pgdb, nocommit=True )
                    pgdb.commit()
                    if provtag is not None:
                        Provenance.addtag( provtag, provs, pgdb=pgdb )

        return provtree

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
                                    f"{self.pars.instrument}, filter {filt}, and parameters {kwargs}" )
            refs[filt] = ref

        return refs

    def process_one_image( self, img ):
        pass

    def run( self ):
        self.provtree = self.make_prov_tree( save=True )

        if self.pars.filter is None:
            imgs, wcsen, zps = Image.find_images( ra=self.ra, dec=self.dec, type='Sci',
                                                  provenance_ids=self.provtree['photocal'], provenance_ids_are_zp=True,
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
                                                                  provenance_ids=self.provtree['photocal'],
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
        ds.prov_tree = self.provtree
        refs = self.find_refs( ds, filters=filters, mjd0=imgs[0].mjd, mjd1=imgs[1].mjd )

        for img in imgs:
            ds = DataStore( img )
            ds.prov_tree = self.provtree
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
                                                   trimprovs=[ self.provtree['Image.trim'],
                                                               self.provtree['Image.trim.sources'],
                                                               self.provtree['Image.trim.wcs']
                                                              ] )
                cropbg = ds.get_bg().trim( x0, x1, y0, y1, trimmed_sources=cropsrc )
                croppsf = ds.get_psf().trim( x0, x1, y0, y1, trimmed_sources=cropsrc )
                ds.get_zp()
                cropzp = ZeroPoint( zp=ds.zp.zp, dzp=ds.zp.dzp, aper_cor_radii=ds.zp.aper_cor_radii,
                                    aper_cors=ds.aper_cors.copy(), provenance_id=self.provtree['Image.trim.zp'] )
                ds.image = cropim
                ds.image_id = cropim.id
                ds.sources = cropsrc
                ds.bg = cropbg
                ds.psf = croppsf
                ds.wcs = cropwcs
                ds.zp = cropzp

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

            measurements = improc.photometry( sub_image, sub_noise, sub_mask, positions=[(xctr, yctr)],
                                              pfsobj=newpsf, apers=new_zp.aper_cor_radii )

            # ROB YOU WERE HERE
