import argparse

import numpy as np
import psycopg.rows

from astropy.io import fits

from models.base import PsycopgConnection, SmartSession
from models.image import Image
from util.logger import SCLogger
import util.util


class FlatBuilder:
    def __init__( self, combmode="median", normmethod="median", sigma=3., iterations=3 ):
        self.normmethod = normmethod
        self.combmode = combmode
        self.sigma = sigma
        self.iterations = iterations
        self.flat = None
        self.filter = None
        self.telescope = None
        self.instrument = None
        self.section_id = None

    def set_input_files( self, files, isdb=True ):
        if files is None:
            raise ValueError( "Gotta give me files" )
        self.files = util.util.listify( files )
        self.flat = None
        self.isdb = isdb

    def calculate_flat( self, dbcon=None ):
        if len(self.files) < 5:
            raise RuntimeError( f"Really?  You want to build a flat from only {len(self.files)} images?" )

        # Do some checks before all the I/O
        if self.normmethod not in [ "median" ]:
            raise ValueError( f"Unknown normalization method {self.normmethod}" )
        if self.combmode not in [ "median", "sigmaclipmedian" ]:
            raise ValueError( f"Unknown combinaton mode {self.combmode}" )
        if self.combmode == "sigmaclipmedian":
            raise NotImplementedError( "sigmaclipmedian not implemented yet" )

        # Read all the images into one massive numpy array.  This will be memory profilgate,
        #  but hopefully in this day and age we don't care.
        massive_stack = None

        SCLogger.info( f"Trying to build a flat from {len(self.files)} images" )

        if self.isdb:
            ids = []
            with PsycopgConnection(dbcon) as conn:
                cursor = conn.cursor( row_factory=psycopg.rows.dict_row )
                for f in self.files:
                    cursor.execute( "SELECT _id FROM images WHERE filepath=%(f)s or _id::text=%(f)s", { 'f': f } )
                    rows = cursor.fetchall()
                    if len(rows) == 0:
                        raise FileNotFoundError( f"No image with filepath or id {f}" )
                    if len(rows) > 1:
                        raise RuntimeError( f">1 image with filepath or id {f}; this should never happen." )
                    ids.append( rows[0]['_id'] )
            if len(ids) != len(self.files):
                raise RuntimeError( "This should never happen" )

            # TODO : use PsycopgConnection.  Should update UUIDMixin.get_by_id to be able to do this.
            #   But, will want to write tests and such to make sure it really does work.
            with SmartSession() as sess:
                for i, imageid in enumerate( ids ):
                    SCLogger.info( f"Reading {self.files[i]}..." )
                    im = Image.get_by_id( imageid, session=sess )
                    if massive_stack is None:
                        self.filter = im.filter
                        self.telescope = im.telescope
                        self.instrument = im.instrument
                        self.section_id = im.section_id
                        massive_stack = np.empty( ( im.data.shape[0], im.data.shape[1], len(ids) ) )
                    else:
                        if im.data.shape != massive_stack.shape[0:2]:
                            raise RuntimeError( f"First image ({self.files[0]}) has shape {massive_stack.shape[0:2]}, "
                                                f"but {self.files[i]} has shape {im.shape}" )
                        for prop in [ 'filter', 'telescope', 'instrument', 'section_id' ]:
                            meprop = getattr( self, prop )
                            improp = getattr( im, prop )
                            if meprop != improp:
                                SCLogger.warning( f"First image ({self.files[0]}) has {prop} {meprop} but "
                                                  f"{self.files[i]} has {prop} {improp}.  Blindly proceeding." )
                    massive_stack[ :, :, i ] = im.data

        else:
            raise NotImplementedError( "FlatBuilder currently can't handle non-database images." )

        # Normalize each image so they are comparable
        # There's probably a single numpy command to do this, but whatevs
        SCLogger.info( "Normalizing all images..." )
        for i in range(len(self.files)):
            if self.normmethod == 'median':
                massive_stack[ :, :, i ] /= np.median( massive_stack[ :, :, i ] )
            else:
                raise ValueError( f"Unknown normalization method {self.normmethod}" )

        # Combine all the images
        SCLogger.info( "Building flat..." )
        if self.combmode == "median":
            self.flat = np.median( massive_stack, axis=2 )

        SCLogger.info( "...done building flat." )

    def write_flat( self, outfile ):
        if self.flat is None:
            raise RuntimeError( "No flat to write." )

        hdu = fits.PrimaryHDU()
        hdu.header['TELESCOP'] = self.telescope
        hdu.header['INSTRUME'] = self.instrument
        hdu.header['SEC_ID'] = self.section_id
        hdu.header['FILTER'] = self.filter
        hdu.header['COMMENT'] = f"Flat normmethod={self.normmethod}"
        hdu.header['COMMENT'] = f"Flat combmode={self.combmode}"
        if self.combmode == 'sigmaclipmedian':
            hdu.header['COMMENT'] = f"...sigma={self.sigma}, iterations={self.iterations}"
        hdu.header['COMMENT'] = "Files combined:"
        for f in self.files:
            hdu.header['COMMENT'] = f"  {f}"

        hdu.data = self.flat

        SCLogger.info( f"Writing flat to {outfile}..." )
        hdu.writeto( outfile )
        SCLogger.info( "...written." )


# ======================================================================

def main():
    parser = argparse.ArgumentParser( "flatbuilder", description="try to make a flat",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "images", nargs='*', default=[],
                         help="Images to combine (paths, database filepaths, or database ids)" )
    parser.add_argument( "-l", "--file-list", default=None,
                         help=( "Text file with list of images, one per line.  Must not specify any images if you "
                                "specify this" ) )
    parser.add_argument( "-o", "--outfile", default="flat.fits", help="Name of output file" )
    # TODO : combination mode, normalization mode, isdb

    args = parser.parse_args()

    builder = FlatBuilder( combmode="median", normmethod="median" )

    if args.file_list is not None:
        if len(args.images):
            raise ValueError( "Specify images on the command line, or a file with a list of images, not both." )
        files = []
        with open( args.file_list ) as ifp:
            for line in ifp:
                line = line.strip()
                if not ( ( line[0] == '#' ) or ( len(line) == 0 ) ):
                    files.append( line )
    elif len(args.images) == 0:
        raise ValueError( "Must give either images on the command line, or a file with a list of images." )
    else:
        files = args.images

    builder.set_input_files( files )
    builder.calculate_flat()
    builder.write_flat( args.outfile )


# ======================================================================
if __name__ == "__main__":
    main()
