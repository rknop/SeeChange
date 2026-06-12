import io
import datetime
import textwrap
import argparse

import numpy as np
from psycopg import sql
import astropy.wcs

from pipeline.parameters import Parameters
from pipeline.coaddition import CoaddPipeline
from pipeline.data_store import DataStore, ProvenanceTree

from models.base import SmartSession, PGDB
from models.enums_and_bitflags import ImageTypeConverter
from models.provenance import Provenance
from models.reference import Reference
from models.image import Image
from models.refset import RefSet

from util.config import Config
from util.logger import SCLogger
from util.util import parse_dateobs


class ParsRefMaker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = self.add_par(
            'name',
            'default',
            str,
            'Name of the reference set. ',
            critical=False,  # the name of the refset is not in the Reference provenance!
            # this means multiple refsets can refer to the same Reference provenance
        )

        self.description = self.add_par(
            'description',
            '',
            str,
            'Description of the reference set. ',
            critical=False,
        )

        self.absolute_pixel_scale = self.add_par(
            'absolute_pixel_scale',
            1.0,
            float,
            ( 'Pixel scale of the coadded image in "/pixel if and only if coaddtion.alignment_index '
              'is set to "absolute"' ),
            critical=True
        )

        self.start_time = self.add_par(
            'start_time',
            None,
            (None, str, float, datetime.datetime, datetime.date),
            'Only use images taken after this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the start time. ',
            critical=True,
        )

        self.end_time = self.add_par(
            'end_time',
            None,
            (None, str, float, datetime.datetime, datetime.date),
            'Only use images taken before this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the end time. ',
            critical=True,
        )

        self.corner_distance = self.add_par(
            'corner_distance',
            0.8,
            (None, float),
            ( 'When finding references, make sure that we have at least min_number references overlapping '
              'nine positions on the rectangle we care about, specified by minra/maxra/mindec/maxdec passed '
              'to run().  One is the center.  The other eight are in a rectangle around the center; '
              'corner_distance is the fraction of the distance from the center to the edge along the '
              'relevant direction.  If this is None, then only consider the center; in that case, pass '
              'only ra and dec to run().' ),
            critical=True,
        )

        self.overlap_fraction = self.add_par(
            'overlap_fraction',
            0.9,
            (None, float),
            ( "When looking for pre-existing references, only return ones whose are overlaps this "
              "fraction of the desired rectangle's area.  Must be None if corner distance is None." ),
            critical=True,
        )

        self.coadd_overlap_fraction = self.add_par(
            'coadd_overlap_fraction',
            0.1,
            (None, float),
            ( "When looking for images to coadd into a new reference, only consider images whose "
              "min/max ra/dec overlap the sky rectangle of the target by at least this much.  "
              "Ignored when corner_distance is None." ),
            critical=True,
        )

        self.instrument = self.add_par(
            'instrument',
            'DECam',
            str,
            'The instrument for which we are building a reference.',
            critical=True,
        )

        self.projects = self.add_par(
            'projects',
            None,
            (None, list),
            'Only use images from these projects. If None, will not limit the projects. '
            'If given as a list, will use any of the projects in the list. ',
            critical=True,
        )

        self.zp_prov_id = self.add_par(
            'zp_prov_id',
            'placeholder',
            str,
            'The provenance of the ZeroPoint for images to be coadded into the reference.',
            critical=True
        )

        self.__image_query_pars__ = ['airmass', 'background', 'seeing', 'lim_mag', 'exp_time']

        for name in self.__image_query_pars__:
            for min_max in ['min', 'max']:
                self.add_limit_parameter(name, min_max)

        self.__docstrings__['min_lim_mag'] = ('Only use images with lim_mag larger (fainter) than this. '
                                              'If None, will not limit the minimal lim_mag. ')
        self.__docstrings__['max_lim_mag'] = ('Only use images with lim_mag smaller (brighter) than this. '
                                              'If None, will not limit the maximal lim_mag. ')

        self.min_number = self.add_par(
            'min_number',
            1,
            int,
            ( 'Construct a reference only if there are at least this many images that pass all other criteria '
              'If corner_distance is not None, then this applies to all test positions on the image. '
              'This *can* be zero if you\'re OK with some spots not having references, but in that case make '
              'sure that center_min_number is not 0, or things will probably go haywire.' ),
            critical=True,
        )

        self.center_min_number = self.add_par(
            'center_min_number',
            None,
            ( int, type(None) ),
            ( 'Like min_number, but specifically for the center of the image.  (Technically, the minimum '
              'number at the cetner is actually the greater of this and min_number.)  If this is None, then '
              'min_number is used for the center.  You might want to require more images overlapping '
              'at the center.  If corner_distance is None, then it probably doesn\'t make sense to '
              'make this number different from min_number, but whatevs.' ),
            critical=True,
        )

        self.max_number = self.add_par(
            'max_number',
            None,
            (None, int),
            ( 'If there are more than this many images at any position on the image, pick the ones with the '
              'highest "quality".' ),
            critical=True,
        )

        self.seeing_quality_factor = self.add_par(
            'seeing_quality_factor',
            3.0,
            float,
            'linear combination coefficient for adding limiting magnitude and seeing FWHM '
            'when calculating the "image quality" used to rank images. ',
            critical=True,
        )

        self.save_new_refs = self.add_par(
            'save_new_refs',
            True,
            bool,
            'If True, will save the coadd image and commit it and the newly created reference to the database. '
            'If False, will only return it. ',
            critical=False,
        )

        self.coadd_pipeline_config = self.add_par(
            'coadd_pipeline_config',
            {},
            dict,
            ( "Do not actually set this anywhere.  This is here so that coadd pipeline parmeters "
              "can go into the referencing provenance.  (There's a circular dependency between "
              "referencing and coaddition; the coaddition of a reference depends on the referencing "
              "config, because it affects what coadd is run, but then of course the referencing provenance "
              "should depend on the coadd configuration.  The solution we have is to make referencing "
              "an upstream of coaddition, so that coaddition doesn't have to know about image selection "
              "parameters.  That means we can't have the coaddition provenance as an upstream of the referencing "
              "provenance.  Use this parameter as a place to store the full coadd pipeline "
              "config so that the referencing provenance will be different if the coadd parameters change. "
              "This is handled entirely internally, and any values set in referencing.coadd_pipeline_config "
              "in a config file will get wiped out." ),
            critical=True
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)

    def add_limit_parameter(self, name, min_max='min'):
        """Add a parameter in a systematic way. """
        if min_max not in ['min', 'max']:
            raise ValueError('min_max must be either "min" or "max"')
        compare = 'larger' if min_max == 'min' else 'smaller'
        setattr(
            self,
            f'{min_max}_{name}',
            self.add_par(
                f'{min_max}_{name}',
                None,
                (None, float),
                f'Only use images with {name} {compare} than this value. '
                f'If None, will not limit the {min_max}imal {name}.',
                critical=True,
            )
        )

    def get_process_name(self):
        return 'referencing'


class RefMaker:
    def __init__(self, **kwargs):
        """Initialize a reference maker object.

        The possible keywords that can be given are: maker, pipeline,
        coaddition. Each should be a dictionary.  maker keys are defined
        by ParsRefMaker above.  pipeline keys are defined as described
        pipeline/top_level.py::Pipeline, and may have subdictionaries
        pipeline, preprocessing, extraction, sources, bg, wcs, and zp.
        coaddition keys are defined by
        pipeline/coaddition.py::ParsCoadd.

        Parameters are set by first looking at the referencing.pipeline,
        referencing.coaddition, and referncing.maker trees from the
        config file.  They are then overridden by anything passed to the
        constructor.

        The maker contains a Pipeline object, that doesn't do any work,
        but is instantiated so it can build up the provenances of the
        images and their products, that go into the coaddition.  Those
        images need to already exist in the database before calling
        run().  Pass kwargs into the pipeline object using
        kwargs['pipeline'].

        The maker also contains a coadd_pipeline object, that has two
        roles: one is to build the provenances of the coadd image and
        the products of that image (extraction on the coadd) and the
        second is to actually do the work of coadding the chosen images.
        Pass kwargs into this object using kwargs['coaddition'].

        The choice of which images are loaded into the reference coadd
        is determined by the parameters object of the maker itself (and
        the provenances of the images and their products).  To set these
        parameters, use the "referencing.maker" dictionary in the
        config, or pass them in kwargs['maker'].

        """

        # We are going to *include* the parametrs of the coadd
        # provenance as part of the parmeters of the referencing
        # provenance, because coadd is no longer an upstream of the
        # referencing.

        if len(kwargs) > 0:
            raise ValueError(f'Unknown parameters given to RefMaker: {kwargs.keys()}')

        # now read the config file
        config = Config.get()

        # coadd config comes from, first, coaddition config, second,
        # updated by referencing.coaddition, finally, updated by keyword
        # arguments.
        coadd_dict = config.value( 'coaddition', {} )
        for kw in [ 'coaddition', 'extraction', 'astrocal', 'photocal' ]:
            if kw not in coadd_dict:
                coadd_dict[kw] = {}
        coadd_dict['coaddition'].update( config.value( 'referencing.coaddition.coaddition', {} ) )
        coadd_dict['coaddition'].update( kwargs.pop( 'coaddition', {} ) )
        coadd_dict['extraction'].update( config.value( 'referencing.coaddition.extraction', {} ) )
        coadd_dict['extraction'].update( kwargs.pop( 'extraction', {} ) )
        coadd_dict['astrocal'].update( config.value( 'referencing.coaddition.astrocal', {} ) )
        coadd_dict['astrocal'].update( kwargs.pop( 'astrocal', {} ) )
        coadd_dict['photocal'].update( config.value( 'referencing.coaddition.photocal', {} ) )
        coadd_dict['photocal'].update( kwargs.pop( 'photocal', {} ) )
        self.coadd_pipeline = CoaddPipeline( **coadd_dict )

        maker_dict = config.value('referencing.maker')
        maker_dict.update( **kwargs )
        # Include the full coaddition pipeline config, *not* just what was modified for ref_maker,
        #   in the paremeters for ref_maker.  Reason: coadd is not an upstream of ref_maker,
        #   so if that stuff changes, we need it here to make the ref_make provenance change.
        # Some of this is a little circular, since the coadd provenance will include this
        #   provenance as an upstream, but whatever.  Gotta have it like this to make sure
        #   the referencing provenacne actually changes when something material about making
        #   the reference changes.
        maker_dict.update( { "coadd_pipeline":
                             { "pipeline": self.coadd_pipeline.pars.get_critical_pars(),
                               "coaddition": self.coadd_pipeline.coadder.pars.get_critical_pars(),
                               "extraction": self.coadd_pipeline.extractor.pars.get_critical_pars(),
                               "astrocal": self.coadd_pipeline.astrometor.pars.get_critical_pars(),
                               "photocal": self.coadd_pipeline.photometor.pars.get_critical_pars()
                              }
                            } )
        self.pars = ParsRefMaker(**maker_dict )

        if ( self.pars.corner_distance is None ) != ( self.pars.overlap_fraction is None ):
            raise ValueError( "Configuration error; for RefMaker, must have a float for both of "
                              "corner_distance and overlap_fraction, or both must be None." )

        self.coadd_provs = None
        self.ref_prov = None
        self.refset = None

        self.reset()

    # ======================================================================

    def reset( self ):
        # these attributes tell us the place in the sky (in degrees)
        # where we want to look for objects (given to run()), # and the
        # filter we want to be in.  Optionally, it can also specify a
        # target and section_id to limit images to.

        self.minra = None
        self.maxra = None
        self.mindec = None
        self.maxdec = None
        self.target = None
        self.ra = None  # in degrees
        self.dec = None  # in degrees
        self.target = None  # the name of the target / field ID / Object ID
        self.section_id = None  # a string with the section ID

    # ======================================================================

    def setup_provenances(self, session=None):
        """Make the provenances for the coadd image and all its products.

        The created provenances are loaded into the database.

        """

        # Previously, we had the coadd provenance as an upstream of the referencing
        #   provenance, but really it's the other way around: the coadd is produced
        #   based on stuff referencing calculates.
        pars = self.pars.get_critical_pars()
        code_version = Provenance.get_code_version(self.pars.get_process_name())
        self.ref_prov = Provenance(
            process=self.pars.get_process_name(),
            code_version_id=code_version.id,
            parameters=pars,
            upstreams=[],
            is_testing='test_parameter' in pars,
        )
        self.ref_prov.insert_if_needed()

        zpprov = Provenance.get( self.pars.zp_prov_id )
        if zpprov is None:
            raise RuntimeError( f"Failed to find ZeroPoint provenance {self.pars.zp_prov_id}" )
        upstreams = [ Provenance.get( self.pars.zp_prov_id ) ]
        # Make the referencing an upstream of the coadd, because changes made to the referencing
        #  could change the images we chose to put into the coadd.
        upstreams.append( self.ref_prov )

        self.coadd_pipeline.datastore = DataStore()
        abswcs = ( self.coadd_pipeline.coadder.pars.alignment_index=='absolute' )
        self.coadd_provs = self.coadd_pipeline.make_provenance_tree( None,
                                                                     absolute_alignment_wcs=abswcs,
                                                                     upstream_provs=upstreams )



    # ======================================================================

    def make_refset(self, session=None):
        """Create or load an existing RefSet with the required name.

        Sets self.refset.  Will also make all the required provenances
        (using the config) and load them into the database.

        """
        with SmartSession( session ) as dbsession:
            # make sure all the sundry component provenances are in the database
            self.setup_provenances( session=dbsession )

            # make sure the ref_prov is in the database
            self.ref_prov.insert_if_needed( session=dbsession )

        with PGDB( dictcursor=True ) as pgdb:
            # Lock the refset table so we don't have a race condition
            pgdb.execute_nofetch( "LOCK TABLE refsets" )
            # Check to see if the refset already exists
            rows = pgdb.execute( "SELECT * FROM refsets WHERE name=%(name)s", { 'name': self.pars.name } )
            if len(rows) > 0:
                # refset already exists, make sure the provenance is right
                if rows[0]['provenance_id'] != self.ref_prov.id:
                    raise ValueError( f"Refset {self.pars.name} already exists with provenance "
                                      f"{rows[0]['provenance_id']}, which does not match the "
                                      f"ref provenance we're using: {self.ref_prov.id}" )
                self.refset = RefSet()
                self.refset.set_attributes_from_dict( rows[0] )
            else:
                # refset doesn't exist, make it
                self.refset = RefSet( name=self.pars.name, description=self.pars.description,
                                      provenance_id=self.ref_prov.id )
                pgdb.execute_nofetch( "INSERT INTO refsets(_id,name,description,provenance_id) "
                                      "VALUES (%(id)s,%(name)s,%(desc)s,%(prov)s)",
                                      { 'id': self.refset.id, 'name': self.refset.name,
                                        'desc': self.refset.description, 'prov': self.refset.provenance_id } )
                pgdb.commit()


    # ======================================================================

    def parse_arguments( self, image=None, image_zp_prov_id=None, zp_prov_id=None, ra=None, dec=None,
                             minra=None, maxra=None, mindec=None, maxdec=None,
                             target=None, section_id=None,
                             filter=None ):
        """Parse arguments for the RefMaker.

        There are three modes in which RefMaker can operate:

        * If the corner_distance parameter is None, then we're making a
          reference that covers a single point (useful for forced
          photometry, for instance).  In this case, either specify an
          image (in which case its central ra and dec are used), or
          specify ra/dec.

        * If the corner_distance parameter is not None, we're making a
          reference that covers a rectangle on the sky (covering at
          least the overlap_fraction parameter of the rectangle).  In
          this case, either specify an image that defines the rectangle
          on the sky, or specify minra/maxra/mindec/maxdec.  The rectangle
          is aligned to NS/EW.

        Parameters
        ----------
          image: str or None
            The id of the image, or a substring of the filepath of the
            image.  If the substring is not unique (i.e. there are
            multiple images with this substring), an exception will be
            raised.

          image_zp_prov_id: str or None
            Ignored if image is not None.  If not None, then will search
            for a zeropoint of this provenance that goes with iamge, and
            then will use the WCSes "good" limits to figure out the
            min/max ra/dec rather than the corners in the image.

          ra, dec: float or None
            Position to search.   Only makes sense if pars.corner_distance is None

          minra, maxra, mindec, maxdec: float or None
            Area to search.  Only makes sense if pars.corner_disdtance is not None

          target, section_id: string or None
            Optionally, specify a target and section_id that images must
            have to be considered for inclusion in a reference.  Only
            use this if you're using a survey that's very careful about
            setting its target names, and if you always go back to
            exactly the same fields so you know that the same chip is
            always going to be in the same place.

          filter: string or None
            If given, only find images whose filter match this filter

        """

        self.image = image
        if image is not None:
            if any ( i is not None for i in [ ra, dec, minra, maxra, mindec, maxdec ] ):
                raise ValueError( "If you pass image to RefMaker.run, you can't pass any coordinates." )

            with PGDB( dictcursor=True ) as pgdb:
                rows = pgdb.execute( sql.SQL( textwrap.dedent(
                    """\
                    SELECT _id, ra, dec, minra, maxra, mindec, maxdec FROM images
                    WHERE filepath LIKE {perimg} OR _id::text={img}
                    """
                ) ).format( img=str(image), perimg=f'%%{image}%%' ) )
                if len(rows) == 0:
                    raise FileNotFoundError( f"Could not find image {image}" )
                elif len(rows) > 1:
                    raise RuntimeError( f"More than one image matched {image}; be more specific." )
                imgid = rows[0]['_id']
                imgra = rows[0]['ra']
                imgdec = rows[0]['dec']
                imgminra = rows[0]['minra']
                imgmaxra = rows[0]['maxra']
                imgmindec = rows[0]['mindec']
                imgmaxdec = rows[0]['maxdec']

                if image_zp_prov_id is not None:
                    rows = pgdb.query( sql.SQL( textwrap.dedent(
                        """\
                        SELECT w.good_ramin, w.good_ramax, w.good_decmin, w.good_decmax
                        FROM world_coordinates w
                        INNER JOIN zero_points z ON z.wcs_id=w._id
                                                AND z.provenance_id={zpprov}
                        INNER JOIN source_lists s ON w.sources_id=s._id
                        WHERE s.image_id={imageid}
                        """
                    ) ).format( zpprov=image_zp_prov_id, imageid=imgid ) )
                    if len(rows) == 0:
                        raise RuntimeError( f"Failed to find a zeropoint for image "
                                            f"with provenance {image_zp_prov_id}" )
                    if len(rows) > 1:
                        raise RuntimeError( "This should never happen." )
                    imgminra = rows[0]['good_ramin']
                    imgmaxra = rows[0]['good_ramax']
                    imgmindec = rows[0]['good_decmin']
                    imgmaxdec = rows[0]['good_decmax']
                    if ( imgminra > imgmaxra ):
                        # Try to deal with the "ra spanning 0" case.
                        imgra = ( imgminra - 360. + imgmaxra ) / 2.
                        if imgra < 0:
                            imgra += 360.
                    else:
                        imgra = ( imgminra + imgmaxra )  / 2.
                    imgdec = ( imgmindec + imgmaxdec ) / 2.

        if self.pars.corner_distance is None:
            if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                raise ValueError( "For RefMaker corner_distance None, can't specify minra/maxra/mindec/maxdec" )
            if image is not None:
                if ( ra is not None ) or ( dec is not None ):
                    raise ValueError( "For RefMaker corner_distance None, must specify image or ra/dec, not both" )
                ra = imgra
                dec = imgdec
            else:
                if ( ra is None ) or ( dec is None ):
                    raise ValueError( "For RefMaker corner_distance None, must provide either image or both ra & dec" )
        else:
            if ( ra is not None ) or ( dec is not None ):
                raise ValueError( "For RefMaker corner_distance not None, can't specify ra/dec" )
            if image is not None:
                if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                    raise ValueError( "For RefMaker corner_distance not None, must specify image or "
                                      "minra/maxra/mindex/maxdec, not both" )
                minra = imgminra
                maxra = imgmaxra
                mindec = imgmindec
                maxdec = imgmaxdec
            else:
                if any ( i is None for i in [ minra, maxra, mindec, maxdec ] ):
                    raise ValueError( "For RefMaker corner_distance not None, must specify image or "
                                      "all of minra/maxra/mindec/maxdec" )

        self.minra = minra
        self.maxra = maxra
        self.mindec = mindec
        self.maxdec = maxdec
        self.ra = ra
        self.dec = dec
        self.target = target
        self.section_id = section_id
        self.filter = filter

    # ======================================================================

    def identify_reference_images_to_coadd( self, *args, _do_not_parse_arguments=False, **kwargs ):
        """Identify images in the database that could be used to build our reference.

        See parse_arguments for a description of the arguments.

        (Parameter _do_not_parse_arguments is used internally, ignore it
        if calling this from the outside.)

        Returns
        -------
           images, match_pos, match_count

           images: list of Image
             List of images that can be included in the sum.

           match_pos: 2d numpy array
             Each row is [ra,dec] of a position on the summed image.  If
             operating in (ra,dec) mode (rather than min/max ra/dec
             mode), this will be [[ra,dec]].

          match_count: list of int
             Number of images that overlap the corresponding match_pos.

          match_pos_images: list of list
             List of the ids of the images that overlap this match pos

        """
        if not _do_not_parse_arguments:
            self.parse_arguments( *args, **kwargs )

        if self.pars.corner_distance is None:
            match_pos = [ [ self.ra, self.dec ] ]
            match_count = [ 0 ]
            match_pos_images = [ [] ]
            kwargs = { 'ra': self.ra, 'dec': self.dec }
        else:
            if ( self.maxra < self.minra ):
                dra = ( self.maxra + 360. - self.minra ) * self.pars.corner_distance/2.
                ctrra = ( self.maxra+360. + self.minra ) / 2.
                ctrra = ctrra if ctrra >= 0. else ctrra + 360.
            else:
                dra = ( self.maxra - self.minra ) * self.pars.corner_distance/2.
                ctrra = ( self.maxra + self.minra ) / 2.
            ddec = ( self.maxdec - self.mindec ) * self.pars.corner_distance/2.
            ctrdec = ( self.maxdec + self.mindec ) / 2.
            match_pos = np.array( [ [ ctrra + 0.,  ctrdec + 0. ],
                                    [ ctrra - dra, ctrdec - ddec ],
                                    [ ctrra + 0.,  ctrdec - ddec ],
                                    [ ctrra + dra, ctrdec - ddec ],
                                    [ ctrra - dra, ctrdec + 0. ],
                                    [ ctrra + dra, ctrdec + 0. ],
                                    [ ctrra - dra, ctrdec + ddec ],
                                    [ ctrra + 0.,  ctrdec + ddec ],
                                    [ ctrra + dra, ctrdec + ddec ] ] )
            match_count = [ 0 ] * 9
            match_pos_images = [ [] ] * 9
            kwargs = { 'minra': self.minra, 'maxra': self.maxra, 'mindec': self.mindec, 'maxdec': self.maxdec,
                       'overlapfrac': self.pars.coadd_overlap_fraction }

        kwargs['provenance_ids'] = [ self.pars.zp_prov_id ]
        kwargs['provenance_ids_are_zp'] = True
        kwargs['instrument' ] = self.pars.instrument
        kwargs['project'] = self.pars.projects
        kwargs['filter'] = self.filter
        kwargs['min_mjd'] = ( None if self.pars.start_time is None
                              else parse_dateobs( self.pars.start_time, output='mjd' ) )
        kwargs['max_mjd'] = None if self.pars.end_time is None else parse_dateobs( self.pars.end_time, output='mjd' )
        kwargs['max_seeing'] = self.pars.max_seeing
        kwargs['min_lim_mag'] = self.pars.min_lim_mag
        kwargs['min_exp_time'] = self.pars.min_exp_time
        kwargs['return_wcs'] = True
        # TODO : airmass, background

        possible, possible_wcs = Image.find_images( **kwargs )

        existing = []
        for image in possible:
            keep = False
            for i, pos in enumerate(match_pos):
                # Use wcs contains so the "good" corners wll be the criteria
                if possible_wcs[image.id].contains( pos[0], pos[1] ):
                    match_pos_images[i].append( image.id )
                    match_count[i] += 1
                    keep = True
            if keep:
                existing.append( image )

        return existing, match_pos, match_count, match_pos_images


    # ======================================================================

    def run(self, *args, do_not_build=False, identify_even_if_not_building=False, **kwargs ):
        """Look to see if there is an existing reference that matches the specs; if not, optionally build one.

        See parse_arguments for function call parameters.  The remaining
        policy for which images to pick, and what provenance to use to
        find references, is defined by the parameters object of self and
        self.pipeline.

        If do_not_build is true, this becomes a thin front-end for Reference.get_references().

        Will check if a RefSet exists with the same provenance and name, and if it doesn't, will create a new
        RefSet with these properties, to keep track of the reference provenances.

        Will return a Reference, or None in case it doesn't exist and cannot be created
        (e.g., because there are not enough images that pass the criteria).

        """

        self.parse_arguments( *args, **kwargs )
        self.make_refset()

        # look for the reference at the given location in the sky (via ra/dec or target/section_id)
        refsandimgs = Reference.get_references(
            minra=self.minra,
            maxra=self.maxra,
            mindec=self.mindec,
            maxdec=self.maxdec,
            ra=self.ra,
            dec=self.dec,
            target=self.target,
            section_id=self.section_id,
            filter=self.filter,
            provenance_ids=self.ref_prov.id,
        )

        refs, _ = refsandimgs

        # if found a reference, can skip the next part of the code!
        if len(refs) == 1:
            return refs[0]
        elif len(refs) > 1:
            raise RuntimeError( f'Found multiple references with the same provenance '
                                f'{self.ref_prov.id} and location!' )

        if do_not_build and ( not identify_even_if_not_building ):
            return None

        ############### no reference found, need to build one! ################

        ( images, match_pos, match_count, match_pos_images
          ) = self.identify_reference_images_to_coadd( _do_not_parse_arguments=True )

        # Make sure we got enough
        nrequired = [ self.pars.min_number ] * len( match_pos )
        if self.pars.center_min_number is not None:
            # match_count[0] is always for the center position
            nrequired[0] = max( self.pars.min_number, self.pars.center_min_number )

        if len(images) < self.pars.min_number:
            SCLogger.info( f"RefMaker only found {len(images)} images overlapping the desired field, "
                           f"which is less than the minimum of {self.pars.min_number}" )
            return None
        if any( n < minn for n, minn in zip( match_count, nrequired ) ):
            SCLogger.info( f"RefMaker didn't find enough references at at least one point on the image; "
                           f"match_count={match_count}, min_number={self.pars.min_number} "
                           f"({nrequired[0]} at center)." )
            return None

        # If there were *too many* images, then we have to start trimming them out
        if ( self.pars.max_number is not None ) and any( n > self.pars.max_number for n in match_count ):
            SCLogger.debug( f"More images than max_number ({self.pars.max_number}) at some positions, trimming." )
            # Sort images by quality factor
            images = sorted( images, key=lambda x: ( x.lim_mag_estimate -
                                                     self.pars.seeing_quality_factor * x.fwhm_estimate ) )
            # Go from the lowest quality to highest quality images, and remove them if they're not needed
            imdex = 0
            nyank = 0
            while any( n > self.pars.max_number for n in match_count ) and ( imdex < len(images) ):
                # Find out of this image is an excess at any position, AND if we can remove
                #   it without dropping another position below the min.
                if ( any( ( ( images[imdex].id in mpi ) and ( n > self.pars.max_number ) )
                          for n, mpi in zip( match_count, match_pos_images )
                         )
                     and
                     all( ( ( images[imdex].id not in mpi ) or ( n > minn ) )
                          for mpi, n, minn in zip( match_pos_images, match_count, nrequired )
                         )
                    ):
                    for i, mpi in enumerate( match_pos_images ):
                        if images[imdex].id in mpi:
                            mpi.remove( images[imdex].id )
                            match_count[ i ] -= 1
                    del images[imdex]
                    nyank += 1
                else:
                    imdex += 1
            SCLogger.debug( f"Removed {nyank} of {len(images)+nyank} images." )

        # Sort the images and create data stores for all of them
        # Have to pull out all the zeropoint upstream provenances
        #   so the DataStore can find its stuff.

        images = sorted(images, key=lambda x: x.mjd)  # sort the images in chronological order for coaddition
        dses = []
        improv = Provenance.get( images[0].provenance_id )
        zpprov = Provenance.get( self.pars.zp_prov_id )
        if ( len(zpprov.upstreams) != 1 ) or ( zpprov.upstreams[0].process != 'astrocal' ):
            raise RuntimeError( "I don't know how to cope" )
        wcsprov = zpprov.upstreams[0]
        if ( len(wcsprov.upstreams) != 1 ) or ( wcsprov.upstreams[0].process != 'extraction' ):
            raise RuntimeError( "I don't know how to cope" )
        srcprov = wcsprov.upstreams[0]
        if ( len(srcprov.upstreams) != 1 ) or ( srcprov.upstreams[0].id != improv.id ):
            raise RuntimeError( "I don't know how to cope" )
        provtree = ProvenanceTree( { p.process: p for p in [ improv, srcprov, wcsprov, zpprov ] },
                                   upstream_steps={ improv.process: [ 'starting_point' ],
                                                    srcprov.process: [ improv.process ],
                                                    wcsprov.process: [ srcprov.process ],
                                                    zpprov.process: [ wcsprov.process ] } )
        for im in images:
            inst = im.instrument
            if inst != self.pars.instrument:
                raise RuntimeError( f"RefMaker for instrument {self.pars.instrument} got an "
                                    f"image from {inst}" )
            if im.provenance_id != improv.id:
                raise RuntimeError( "This should never happen." )
            ds = DataStore( im )
            ds.edit_prov_tree( provtree )
            ds.sources = ds.get_sources()
            ds.bg = ds.get_background()
            ds.psf = ds.get_psf()
            ds.wcs = ds.get_wcs()
            ds.zp = ds.get_zp()
            prods = {p: getattr(ds, p) for p in ['sources', 'psf', 'bg', 'wcs', 'zp']}
            if any( [p is None for p in prods.values()] ):
                raise RuntimeError(
                    f'DataStore for image {im} is missing some of products {prods} for coaddition! '
                    f'Make sure to produce products using the provenances in ex_provs: '
                    f'{self.ex_provs}'
                )
            dses.append( ds )

        nlsp = '\n        '
        mess = nlsp.join( f'({p[0]:8.4f}, {p[1]:8.4f}) : {c}' for p, c in zip( match_pos, match_count ) )
        if do_not_build:
            # If we get here, it's because identify_even_if_not_building is True, which means
            #   the user probably wants to see the list, so log this as info rather than debug.
            SCLogger.info( f"Overlap statistics:{nlsp}{mess}" )
            SCLogger.info( f"{len(dses)} images that would have been combined for the reference:{nlsp}"
                           f"{nlsp.join( d.image.filepath for d in dses )}" )
            return None
        else:
            SCLogger.debug( f"Overlap statistics:{nlsp}{mess}" )
            SCLogger.debug( f"Combining images to make ref:\n"
                            f"{nlsp.join( d.image.filepath for d in dses )}" )

        alignment_target_datastore = None
        alignment_wcs = None
        if self.coadd_pipeline.coadder.pars.alignment_index == 'other':
            raise NotImplementedError( "Gotta do this." )
        elif self.coadd_pipeline.coadder.pars.alignment_index == 'absolute':
            # Gotta figure out the center ra and dec of the image we build
            # THOGUHT REQUIRED : does this align them where we want on heavily
            #   masked images?
            if self.minra is not None:
                if ( self.minra > self.maxra ):
                    self.minra -= 360.
                ra = ( self.minra + self.maxra ) / 2.
                if ( self.minra < 0. ):
                    self.minra += 360.
                dec = ( self.mindec + self.maxdec ) / 2.
            else:
                ra = self.ra
                dec = self.dec

            # Right... I hope I know what I'm doing.
            alignment_wcs = astropy.wcs.WCS(
                { 'CRPIX1': self.coadd_pipeline.coadder.pars.absolute_width / 2.,
                  'CRPIX2': self.coadd_pipeline.coadder.pars.absolute_height / 2.,
                  'CRVAL1': ra,
                  'CRVAL2': dec,
                  'CDELT1': -self.pars.absolute_pixel_scale / 3600.,
                  'CDELT2': self.pars.absolute_pixel_scale / 3600.,
                  'CTYPE1': "RA---TAN",
                  'CTYPE2': "DEC--TAN",
                  'CUNIT1': 'deg',
                  'CUNIT2': 'deg' } )

        coadd_ds = self.coadd_pipeline.run( dses, prov_tree=self.coadd_provs,
                                            alignment_target_datastore=alignment_target_datastore,
                                            alignment_wcs=alignment_wcs )

        ref = Reference(
            zp_id = coadd_ds.zp.id,
            provenance_id = self.ref_prov.id
        )

        if self.pars.save_new_refs:
            coadd_ds.save_and_commit()
            ref.insert()

        return ref

# ======================================================================


class ArgFormatter( argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter ):
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )


def main():
    parser = argparse.ArgumentParser( 'ref_maker.py',
                                      description="Build a reference",
                                      formatter_class=ArgFormatter,
                                      epilog="Rob write help" )
    parser.add_argument( "-r", "--ra", type=float, default=None,
                         help="RA to make a reference for; decimal degrees.  See description above." )
    parser.add_argument( "-d", "--dec", type=float, default=None,
                         help="RA to make a reference for; decimal degrees.  See description above." )
    parser.add_argument( "-i", "--image", type=str, default=None,
                         help="filepath or uuid of image to make a reference for." )
    parser.add_argument( "-z", "--image-zp-prov-id", type=str, default=None, help="See description above." )
    parser.add_argument( "--minra", type=float, default=None, help="See description above." )
    parser.add_argument( "--maxra", type=float, default=None, help="See description above." )
    parser.add_argument( "--mindec", type=float, default=None, help="See description above." )
    parser.add_argument( "--maxdec", type=float, default=None, help="See description above." )
    parser.add_argument( "-f", "--filter", type=str, required=True, help="Filter name." )
    parser.add_argument( "-n", "--no-build", default=False, action="store_true",
                         help="Don't build a reference if one isn't found" )
    parser.add_argument( "-l", "--list-images", default=False, action="store_true",
                         help=( "List the images that are combined into the ref.  If --no-build is True, "
                                "list the images that would have been combined into the ref even if a "
                                "ref is not built." ) )

    # TODO : add arugments that let us override what's in the config file?  Or just rely on config file?
    # (Probably we want this, so we can do one-offs, but put in warnings or require a --override-config
    # so that we don't do it willy-nilly.)
    # parser.add_argument( "-n", "--name", required=True, help="Name of refset" )
    # parser.add_argument( "-d", "--description", default="",
    #                      help="Description of refset.  Only used if the refset is newly created." )
    # parser.add_argument( "-s", "--start-time", type=str, default=None,
    #                      help=( "YYYY-MM-DDTHH:MM:SS (may omit THH:MM:SS).  Only use images taken "
    #                             "after this time (inclusive)" ) )
    # parser.add_argument( "-e", "--end-time", type=str, default=None,
    #                      help=( "YYYY-MM-DDTHH:MM:SS (may omit THH:MM:SS).  Only use images taken "
    #                             "before this time (inclusive)" ) )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    kwargs['do_not_build'] = kwargs['no_build']
    kwargs['identify_even_if_not_building'] = kwargs['list_images']
    del kwargs['no_build']
    del kwargs['list_images']

    refmaker = RefMaker()

    # TODO : Process arguments that override the config file.  (See TODO above.)

    ref = refmaker.run( **kwargs )

    if ref is None:
        SCLogger.warning( "No ref built or returned." )

    else:
        with PGDB( dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i.filepath, i.filter, i.section_id, i._type, b.noise, p.fwhm_pixels, z.zp,
                       ARRAY_AGG(subim.filepath) AS compimages
                FROM refs r
                INNER JOIN zero_points z ON r.zp_id=z._id
                INNER JOIN world_coordinates w ON z.wcs_id=w._id
                INNER JOIN source_lists s ON w.sources_id=s._id
                INNER JOIN backgrounds b ON b.sources_id=s._id
                INNER JOIN psfs p ON p.sources_id=s._id
                INNER JOIN images i ON s.image_id=i._id
                INNER JOIN image_coadd_component comp ON comp.coadd_image_id=i._id
                INNER JOIN zero_points compz ON compz._id=comp.zp_id
                INNER JOIN world_coordinates compw ON compw._id=compz.wcs_id
                INNER JOIN source_lists comps ON comps._id=compw.sources_id
                INNER JOIN images subim ON subim._id=comps.image_id
                WHERE r._id={refid} AND z.provenance_id={zpprov}
                GROUP BY i.filepath, i.filter, i.section_id, i._type, b.noise, p.fwhm_pixels, z.zp
                """
            ) ).format( refid=ref.id, zpprov=refmaker.coadd_provs['photocal'].id )
            rows = pgdb.execute( q )

        if len( rows ) != 1:
            import pdb; pdb.set_trace()
            raise RuntimeError( "This should not happen." )
        row = rows[0]

        # Signal = 10^(zp-m)/2.5 * 0.9375 [ where 0.9375 is the fraction of a Gaussian in a 1FWHM radius ]
        # Noise = sqrt( π * fhwm² * noise² )
        # So m = zp - 2.5 * log10( S/N * √π * fwhm * noise / 0.9375 )
        limag = row['zp'] - 2.5 * np.log10( 5. * np.sqrt(np.pi) * row['fwhm_pixels'] * row['noise'] / 0.9375 )

        strio = io.StringIO()
        strio.write( f"Combined reference: {row['filepath']}\n" )
        strio.write( f"      section={row['section_id']}, filter={row['filter']}, zp={row['zp']:.2f}, "
                     f"skyσ={row['noise']:.1f}, fwhm={row['fwhm_pixels']:.2f} pix\n" )
        strio.write( f"      5σ limiting magnitude is, maybe, {limag:.2f}\n" )
        nlsp = '\n        '
        strio.write( f"      Combined {len(row['compimages'])} images:\n{nlsp.join(row['compimages'])}" )
        SCLogger.info( strio.getvalue() )

        typ = ImageTypeConverter.to_string( row['_type'] )
        if typ != 'ComSci':
            SCLogger.warning( f"Coadd image has type {typ}, expected ComSci." )


# ======================================================================

if __name__ == "__main__":
    main()
