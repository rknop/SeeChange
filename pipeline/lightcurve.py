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
from util.config import Config, NoValue
from util.logging import SCLogger
from util.util import listify
from pipeline.parameters import Parameters


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
            docstring = ( "The process to use when searching provenance tags for object position provenance." )
            critical = False
        )

        self.only_existing_subtractions = self.add_par(
            name = "only_existing_subtractions",
            default = False,
            par_types = bool,
            docstring = ( "Don't do any new subtractions, only look at existing subtractions to do forced "
                          "photometry on.  This will be existing subtractions that have the same provenance "
                          "*and* that use the same reference as what the pipeline would do for new subtractions." )
            critical = True
        )

        # self.referencing_config = self.add_par(
        #     name = "referencing_config",
        #     default = {},
        #     par_types = dict,
        #     docstring = ( "A dictionary with referencing config.  Lightcurve will start with the base "
        #                   "referencing config, which is for searching.  Then, values here will override "
        #                   "anything that's in the base config before the RefMaker is actually made.  "
        #                   "If you want to guarantee use of an existing refset, just make this "
        #                   "{'maker':{ 'name': name, 'ignore_config_use_config_from_refset': True, "
        #                   "{refset_must_already_exist': True}} where name is the name of the refset you "
        #                   "want to use." ),
        #     # Not critical because a referencing provenance will be in the upstreams of the
        #     #   forced phot provenance.
        #     critical = False
        # )

        self.reference_max_dist = self.add_par(
            name = "reference_max_dist",
            default = 40. / 3600.,
            par_types = float,
            docstring = ( "When searching for an existing reference, only keep one if its center is at most "
                          "this many degrees from the ra/dec of the object we're building the lightcurve for." ),
            critical = True
        )
        
        self.subtraction_config = self.add_par(
            name = "subtraction_config",
            default = {},
            par_types = dict,
            docstring = "A dictionary with subtraction config.  Will override what's in subtraction config.",
            # Not critical because a subtraction provenance will be in the upstreams of the forced
            #   phot provenance
            critical = False
        )
        
        self.crop_image = self.add_par(
            name = "crop_image"
            default = None,
            par_types = ( tuple, None ),
            docstring = ( "If given, a tuple of (width, height).  Science images will be trimmed to at most "
                          "this size before being fed to subtractions.  You want something small, because you're "
                          "just after the image aroud your ra/dec, BUT you need enough to have stars for "
                          "psf, wcs, and zerooint to work.  TODO: implement the ability to create a psf for a "
                          "subset image from an existing psf.  This may be hard, as it might involve hacking "
                          "the internal format of psfex files.  We already have it for wcs, and zp is trivial. "
                          "For now, though, all the steps are rerun on the cropped image." ),
            critical=True
        )

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

        if ( self.zp_prov is None ) or ( self.zp_prov.id != self.pars.zp_prov ):
            self.zp_prov = None
            if self.pars.zp_prov is not None:
                self.zp_prov = Provenance.get( self.pars.zp_prov, pgdb=pgdb )
            elif self.pars.zp_prov_tag is not None:
                self.zp_prov = Provenance.get_for_tag( self.pars.zp_prov_tag, self.pars.zp_prov_tag_process, pgdb=pgdb )

        if self.zp_prov is None:
            raise RuntimeError( f"Could not find a zeropoint provenance to use to find images. "
                                f"zp_prov={self.pars.zp_prov}, zp_prov_tag={self.pars.zp_prov_tag}, "
                                f"zp_prov_tag_process={self.pars.zp_prov_tag_process}" )

        if ( self.object_position_prov is None ) or ( self.object_position_prov.id != self.pars.object_position_prov ):
            if self.pars.object_position_prov is not None:
                self.object_position_prov = Provenance.get( self.pars.object_position_prov, pgdb=pgdb )
                if self.object_position_prov is None:
                    raise ValueError( f"Could not find object position provenance {self.pars.object_position_prov}" )
                elif self.pars.object_position_prov_tag is not None:
                    self.object_position_prov = Provenance.get_by_Tag( self.pars.object_position_prov_tag,
                                                                       self.pars.object_position_prov_tag_process,
                                                                       pgdb=pgdg )
                    if self.object_position_prov is None:
                        raise ValueError( f"Could not find object position provenance for "
                                          f"tag {self.pars.object_position_prov_tag} and "
                                          f"process { self.pars.object_position_prov_tag_process}" )

        with PGDB( pgdb, dictcursor=True ) as mypgdb:
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
                    raise RuntimeError( "This should neve rhapen, I don't think, but I'm not really sure." )
                elif len(rows) == 0:
                    raise ValueError( f"Could not find object position for object {self.object.id} "
                                      f"and object position provenacne {self.object_position_prov.id}" )
                else:
                    self.object_position = ObjectPosition( **(rows[0]) )


    def find_refs( self, pgdb=None ):
        if self.object_position is not None:
            ra = self.object_position.ra
            dec = self.object_position.dec
        else:
            ra = self.object.ra
            dec = self.object.dec

        with PGDB( pgdb, dictcursor=True ) as pgdb:
            self.refset = RefSet.get_by_name( self.subtrator.pars.refset, pgdb=pgdb )
            self.refset.provenance = Provenance.get_by_id( self.refset.provenance_id, pgdb=pgdb )
            if self.refset is None:
                raise ValueError( f"Can't find refset {self.subtractor.pars.refset}" )

            for filt in self.pars.filters:
                refs, imgs = Reference.get_references( ra=ra, dec=dec, filter=filt, refset=self.refset.name, pgdb=pgdb )
                if len(refs) == 0:
                    raise RuntimeError( f"Can't find reference for filter {filt}" )
                if len(refs) > 0:
                    # Sort by distance
                    dist = np.array( [ np.sqrt( ( (i.ra - ra) * np.cos(dec * np.pi/180.) )**2 +
                                                (i.dec - dec)**2 )
                                       for i in imgs ] )
                    distdex = np.argsort( dist )
                    if dist[ distdex[0] ] > self.reference_max_dist:
                        raise RuntimeError( f"Can't find reference for filter {filt}" )
                    if np.fabs( dist[distdex[0]] - dist[distdex[1]] ) < 1./3600.:
                        raise RuntimeError( f"...found more than one matching refernce within 1\"!" )

                    self.refs[ filt ] = refs[ distdex[0] ]
                else:
                    self.refs[ filt ] = refs[0]

    def make_provs( self, save=True, provtag=None, pgdb=None ):
        # Build a full provenance tree for DataStore to chew on
        # DataStore.make_prov_tree is designed for use with top_level, and is
        #   not easy to use here, so just make one manually.

        procs = [ 'photocal', 'astrocal', 'extraction', 'preprocessing' ]
        provs = { 'photocal': self.zp_prov }
        for i in range(len(procs)-1):
            upstrs = provs[procs[i]].get_upstreams( pgdb=pgdb )
            if ( len(upstrs) != 1 ) or ( upstrs[0].process != procs[i+1] ):
                raise RuntimeError( f"Failed to get {procs[i+1]} upstream provenance for {procs[i]}" )
            provs[procs[i]] = upstrs[0]
        provs['starting_point'] = provs['preprocessing']
        del provs['preprocessing']

        subups = [ self.refset.provenance ]
        if self.pars.crop_image is not None:
            ( provs['Image.trim'],
              provs['Image.trim.nullsources'],
              provs['Image.trim.wcs'],
              provs['Image.trim.zp']
             ) = Image.get_trim_provs( self.pars.crop_image[0], self.pars.crop_image[1],
                                       upstreams=[ provs['starting_point'] ],
                                       wcs_prov=provs['astrocal'], zp_prov=provs['photocal'],
                                       save=save, provtag=provtag, pgdb=pgdb )
            subups.append( provs['Image.trim.zp'] )
        else:
            subups.append( provs['photocal'] )

        provs['subtraction'] = Provenance( code_version_id=Provenance.get_code_version('subtraction', pgdb=pgdb).id,
                                           process='subtraction',
                                           parameters=self.subtractor.pars.get_critical_pars(),
                                           upstreams=subups )
        if save:
            provs['subtraction'].insert_if_needed( session=pgdb )
            if provtag is not None:
                Provenance.addtag( provtag, [provs['subtraction']], pgdb=pgdb )
            
                                           
        


        upstrs = self.zp_prov.get_upstreams( pgdb=pgdb )
        if ( len(upstrs) != 1 ) or ( upstrs[0].process != 'astrocal' ):
            raise RuntimeError( "Failed to find astrocal upstream for zeropoint provenance." )
        wcsprov = upstrs[0]

        upstrs = wcsprov.get_upstreams( pgdb=pgdb )
        if ( len(upstrs) != 1 ) or ( upstrs[0].process != "extraction" ):
            raise RuntimeError( "Failed to find extraction upstream for wcs provenance." )
        sourcesprov = upstrs[0]

        upstrs = sourcesprov.get_upstreams( pgdb=pgdb )
        if ( len(upstrs)
            


            
