import pathlib
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.orm.session import object_session
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property

from astropy.time import Time
from astropy.io import fits

from util.config import Config
from util.fits import read_fits_image
from util.radec import parse_ra_hms_to_deg, parse_dec_dms_to_deg

from models.base import (
    Base,
    SeeChangeBase,
    UUIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    SmartSession,
    HasBitFlagBadness,
)
from models.instrument import guess_instrument, get_instrument_instance

from models.enums_and_bitflags import (
    ImageFormatConverter,
    ImageTypeConverter,
)

# columns key names that must be loaded from the header for each Exposure
EXPOSURE_COLUMN_NAMES = [
    'ra',
    'dec',
    'mjd',
    'project',
    'target',
    'exp_time',
    'filter',
    'telescope',
    'instrument',
    'airmass',
]

# these are header keywords that are not stored as columns of the Exposure table,
# but are still useful to keep around inside the "info" JSONB column.
EXPOSURE_HEADER_KEYS = ['gain']  # TODO: add more here


class SectionData:
    """A helper class that lazy loads the section data from the database.

    When requesting one of the section IDs it will fetch that data from
    disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filepath, instrument):
        """Create a SectionData.

        Must initialize this object with a filepath
        (or list of filepaths) and an instrument object.
        These two things will control how data is loaded
        from the disk.

        Parameters
        ----------
        filepath: str or list of str
            The filepath of the exposure to load.
            If each section is in a different file, then
            this should be a list of filepaths.
        instrument: Instrument
            The instrument object that describes the
            sections and how to load them from disk.

        """
        self.filepath = filepath
        self.instrument = instrument
        self._data = defaultdict(lambda: None)

    def __getitem__(self, section_id):
        if self._data[section_id] is None:
            self._data[section_id] = self.instrument.load_section_image(self.filepath, section_id)
        return self._data[section_id]

    def __setitem__(self, section_id, value):
        self._data[section_id] = value

    def clear_cache(self):
        self._data = defaultdict(lambda: None)


class SectionHeaders:
    """A helper class that lazy loads the section header from the database.

    When requesting one of the section IDs it will fetch the header
    for that section, load it from disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filepath, instrument):
        """Create a SectionHeaders.

        Must initialize this object with a filepath
        (or list of filepaths) and an instrument object.
        These two things will control how data is loaded
        from the disk.

        Parameters
        ----------
        filepath: str or list of str
            The filepath of the exposure to load.
            If each section is in a different file, then
            this should be a list of filepaths.
        instrument: Instrument
            The instrument object that describes the
            sections and how to load them from disk.

        """
        self.filepath = filepath
        self.instrument = instrument
        self._header = defaultdict(lambda: None)  # each item here is a lazy-loaded fits.Header

    def __getitem__(self, section_id):
        if self._header[section_id] is None:
            self._header[section_id] = self.instrument.read_header(self.filepath, section_id)
        return self._header[section_id]

    def __setitem__(self, section_id, value):
        self.header[section_id] = value

    def clear_cache(self):
        self._header = defaultdict(lambda: None)


class ExposureImageIterator:
    """A class to iterate through the HDUs of an exposure, one for each SensorSection."""

    def __iter__( self, exposure ):
        self.exposure = exposure

        self.instrument = get_instrument_instance( self.exposure.instrument )
        self.section_ids = self.instrument.get_section_ids()
        self.dex = 0
        return self

    def __next__( self ):
        from models.image import Image  # avoid circular import
        if self.dex < len( self.section_ids ):
            img = Image.from_exposure( self.exposure, self.section_ids[ self.dex ] )
            self.dex += 1
            return img
        else:
            raise StopIteration


class Exposure(Base, UUIDMixin, FileOnDiskMixin, SpatiallyIndexed, HasBitFlagBadness):
    """Encapsulates one exposure, which includes many images each on different sensor sections (chips).

    Access to the data is through the data, weight, and flags
    attributes, which are SectionData instances.  You can treat them as
    dictionaries, where you give the sensor section id as key.

    Call clear_cache() to clear any cached data from the SectionData instances in order to free up memory.

    """

    __tablename__ = "exposures"

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
            CheckConstraint( sqltext='NOT(_filter IS NULL AND filter_array IS NULL)',
                             name='exposures_filter_or_array_check' )
        )


    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(ImageTypeConverter.convert('Sci')) ),
        index=True,
        doc=(
            "Type of image. One of: Sci, Diff, Bias, Dark, DomeFlat, SkyFlat, TwiFlat, "
            "or any of the above types prepended with 'Com' for combined "
            "(e.g., a ComSci image is a science image combined from multiple exposures)."
            "The value is saved as SMALLINT but translated to a string when read. "
        )
    )

    @hybrid_property
    def type(self):
        return ImageTypeConverter.convert(self._type)

    @type.expression
    def type(cls):  # noqa: N805
        return sa.case(ImageTypeConverter.dict, value=cls._type)

    @type.setter
    def type(self, value):
        self._type = ImageTypeConverter.convert(value)

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(ImageFormatConverter.convert('fits')) ),
        doc="Format of the file on disk. Should be fits or hdf5. "
            "The value is saved as SMALLINT but translated to a string when read. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this exposure. "
            "The provenance will containe a record of the code version "
            "and the parameters used to obtain this exposure."
        )
    )

    @hybrid_property
    def format(self):
        return ImageFormatConverter.convert(self._format)

    @format.expression
    def format(cls):  # noqa: N805
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(ImageFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = ImageFormatConverter.convert(value)

    info = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
        doc=(
            "Subset of the raw exposure's header. "
            "Only keep a subset of the keywords, "
            "and re-key them to be more consistent. "
            "This will only include global values, "
            "not those associated with a specific section. "
        )
    )

    mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc="Modified Julian date of the start of the exposure (MJD=JD-2400000.5)."
    )

    exp_time = sa.Column(sa.REAL, nullable=False, index=True, doc="Exposure time in seconds. ")

    _filter = sa.Column(sa.Text,
                        nullable=True, #nullable because a constraint checks _filter or filter_array
                        index=True,
                        doc=("Name of the filter used to make this exposure. "
                             "This is the short filter name - conversion "
                             "to the proper name for each instrument happens in its code."))

    airmass = sa.Column(sa.REAL, nullable=True, index=True, doc="Airmass taken from the header of the exposure. ")

    @property
    def filter( self ):
        if self._filter is None:
            return None
        else:
            return self._filter

    @filter.setter
    def filter( self, val ):
        # Once instrument can only be a valid instrument, should add
        # a check here which says "IF there is an instrument, only
        # allow a valid filter for that instrument". Leaving the
        # ability to set any filter at the moment for tests.
        self._filter = val

    @property
    def filter_full(self):
        if self.instrument_object is None:
            raise ValueError( "Exposure must have an instrument to determine full filter.")
        return self.instrument_object.get_full_filter_name( self._filter )

    filter_array = sa.Column(
        ARRAY(sa.Text, zero_indexes=True),
        nullable=True,
        index=True,
        doc=( "Array of filter names, if multiple filters were used. "
              "Should be short filter names - instruments can convert to full. ")
    )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the instrument used to take the exposure. '
    )

    project = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the project (could also be a proposal ID). '
    )

    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the target object or field id. '
    )

    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=True,
        doc='Bitflag for this exposure. Good exposures have a bitflag of 0. '
            'Bad exposures are each bad in their own way (i.e., have different bits set). '
    )

    preproc_bitflag = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=False,
        doc=( 'Bitflag specifying which preprocessing steps have been completed for the images in the '
              'exposure.  Useful for things like the NOIRlab archive where we can download preprocessed '
              'images.  If this is anything other than 0, the code might assume that the exposure also '
              'has weight and flags components.' )
    )

    origin_identifier = sa.Column(
        sa.Text,
        nullable=True,
        index=True,
        doc='Opaque string used by InstrumentOriginExposures to identify this exposure remotely'
    )

    def __init__(self, current_file=None, invent_filepath=True, **kwargs):
        """Initialize the exposure object.

        If the filepath is given (as a keyword argument), it will parse the instrument name
        from the filename.  The header will be read out from the file.

        Parameters
        ----------
        All the properties of Exposure (i.e. columns of the exposures table), plus:

        current_file: Path or str
           The path to the file where the exposure currently is (which
           may or may not be the same as the filepath it will have in
           the database).  If you don't specify this, then the file must
           exist at filepath (either the one you pass or the one that is
           determined automatically if invent_filepath is True).

        invent_filepath: bool
           Will be ignored if you specify a filepath as an argument.
           Otherwise, if this is True, call invent_filepath() to create
           the filepath for this Exposure, based on all the other
           properties.  If this is False, then you must specify filepath
           unless the global property Exposure.nofile is True (but you
           really shouldn't be playing around with that).

        """
        FileOnDiskMixin.__init__(self, **kwargs)
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self._data = None  # the underlying image data for each section
        self._section_headers = None  # the headers for individual sections, directly from the FITS file
        self._weight = None
        self._weight_section_headers = None
        self._flags = None
        self._flags_section_headers = None
        self._header = None  # the global (exposure level) header, directly from the FITS file
        self.type = 'Sci'  # default, can override using kwargs
        self._instrument_object = None
        self._bitflag = 0
        self.preproc_bitflag = 0     # 0 is the database default; make sure object has it when created

        if 'header' in kwargs:
            kwargs['_header'] = kwargs.pop('header')

        # manually set all properties (columns or not, but don't
        # overwrite instance methods) Do this once here, because some of
        # the values are going to be needed by upcoming function calls.
        # We will run this exact code again later so that the keywords
        # can override what's detected from the header.
        self.set_attributes_from_dict( kwargs )
        # NOTE: since above reads directly from header, it will often set
        # a full filter name without an instrument, which must be detected
        # and fixed below before making anything permanent (filepaths,
        # provenances etc).

        # must have Instrument to invent a filename (and initialize Provenance)
        # but if not given, it can be guessed from the filepath
        if self.filepath is None and self.instrument is None:
            raise ValueError( "Exposure.__init__: must give at least a filepath or an instrument" )

        if self.filepath is None:
            # in this case, the instrument must have been given
            self.filter = self.instrument_object.get_short_filter_name( self.filter )
            if self.provenance_id is None:
                prov = self.make_provenance(self.instrument)  # a default provenance for exposures
                self.provenance_id = prov.id

            if invent_filepath:
                self.filepath = self.invent_filepath()
            elif not self.nofile:
                raise ValueError("Exposure.__init__: must give a filepath to initialize an Exposure object. ")

        if self.instrument is None:
            self.instrument = guess_instrument(self.filepath)

        # ensure we are working with short filter from here and below
        self.filter = self.instrument_object.get_short_filter_name( self.filter )

        # this can happen if the instrument is not given, but the filepath is
        if self.provenance_id is None:
            prov = self.make_provenance(self.instrument)  # a default provenance for exposures
            self.provenance_id = prov.id


        # instrument_obj is lazy loaded when first getting it
        if current_file is None:
            current_file = pathlib.Path( FileOnDiskMixin.local_path ) / self.filepath
        if self.instrument_object is not None:
            self.use_instrument_to_read_header_data( fromfile=current_file )

        # Allow passed keywords to override what's detected from the header
        self.set_attributes_from_dict( kwargs )
        self.filter = self.instrument_object.get_short_filter_name( self.filter )

        self.calculate_coordinates()  # galactic and ecliptic coordinates

    @classmethod
    def make_provenance(cls, instrument):
        """Generate a Provenance for this exposure and save it to the database.

        Just calls the instrument's get_exposure_provenance() method.
        """
        return get_instrument_instance( instrument ).get_exposure_provenance()


    @sa.orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self._data = None
        self._section_headers = None
        self._weight = None
        self._weight_section_headers = None
        self._flags = None
        self._flags_section_headers = None
        self._header = None
        self._instrument_object = None
        session = object_session(self)
        if session is not None:
            self.update_instrument(session=session)

    def __setattr__(self, key, value):
        if key == 'ra' and isinstance(value, str):
            value = parse_ra_hms_to_deg(value)
        if key == 'dec' and isinstance(value, str):
            value = parse_dec_dms_to_deg(value)

        super().__setattr__(key, value)

    def use_instrument_to_read_header_data(self, fromfile=None):
        """Use the instrument object to read the header data from the file.

        This will set the column attributes from these values.
        Additional header values will be stored in the header JSONB column.
        """
        # get the header from the file in its raw form as a fits.Header object
        if fromfile is None:
            fromfile = self.get_fullpath()
        raw_header_dictionary = self.instrument_object.read_header( fromfile )

        # read and rename/convert units for all the column attributes:
        critical_info = self.instrument_object.extract_header_info(
            header=raw_header_dictionary,
            names=EXPOSURE_COLUMN_NAMES,
        )

        # verify some attributes match, and besides set the column attributes from these values
        for k, v in critical_info.items():
            if k == 'instrument':
                if self.instrument != v:
                    raise ValueError(f"Header instrument {v} does not match Exposure instrument {self.instrument}")
            elif k == 'telescope':
                if self.telescope != v:
                    raise ValueError(
                        f"Header telescope {v} does not match Exposure telescope {self.telescope}"
                    )
            elif k == 'filter' and isinstance(v, list):
                self.filter_array = self.instrument_object.get_short_filter_name( v )
            elif k == 'filter' and isinstance(v, str):
                self.filter = self.instrument_object.get_short_filter_name( v )
            else:
                setattr(self, k, v)

        # these additional keys go into the header only
        auxiliary_names = EXPOSURE_HEADER_KEYS + self.instrument_object.get_auxiliary_exposure_header_keys()
        self.info = self.instrument_object.extract_header_info(
            header=raw_header_dictionary,
            names=auxiliary_names,
        )

    def check_required_attributes(self):
        """Check that this exposure has all the required attributes."""

        missing = []
        required = EXPOSURE_COLUMN_NAMES
        required.pop('filter')  # check this manually after the loop
        for name in required:
            if getattr(self, name) is None:
                missing.append(name)

        # one of these must be defined:
        if self.filter is None and self.filter_array is None:
            missing.append('filter')

        if len(missing) > 0:
            raise ValueError(f"Missing required attributes: {missing}")

    @property
    def instrument_object(self):
        if self.instrument is not None:
            if self._instrument_object is None or self._instrument_object.name != self.instrument:
                self._instrument_object = get_instrument_instance(self.instrument)

        return self._instrument_object

    @property
    def telescope(self):
        return self.instrument_object.telescope

    @instrument_object.setter
    def instrument_object(self, value):
        self._instrument_object = value

    @property
    def start_mjd(self):
        """Time of the beginning of the exposure (equal to mjd). """
        return self.mjd

    @property
    def mid_mjd(self):
        """Time of the middle of the exposure. """
        if self.mjd is None or self.exp_time is None:
            return None
        return (self.start_mjd + self.end_mjd) / 2.0

    @property
    def end_mjd(self):
        """The time when the exposure ended. """
        if self.mjd is None or self.exp_time is None:
            return None
        return self.mjd + self.exp_time / 86400.0

    @property
    def observation_time(self):
        """Translation of the MJD column to datetime object."""
        if self.mjd is None:
            return None
        else:
            return Time(self.mjd, format='mjd').datetime

    def __repr__(self):

        filter_str = '--'
        if self.filter is not None:
            filter_str = self.filter
        if self.filter_array is not None:
            filter_str = f"[{', '.join(self.filter_array)}]"

        return (
            f"Exposure(id: {self.id}, "
            f"exp: {self.exp_time}s, "
            f"filt: {filter_str}, "
            f"from: {self.instrument}/{self.telescope})"
        )

    def __str__(self):
        return self.__repr__()

    def invent_filepath( self ):
        """Create a filepath (relative to data root) for the exposure based on metadata.

        This is used when saving the exposure to disk

        """

        # Much code redundancy with Image.invent_filepath; move to a mixin?

        inst_name = project = ''

        if self.provenance_id is None:
            raise ValueError("Cannot invent filepath for exposure without provenance.")
        if self.instrument_object is not None:
            inst_name = self.instrument_object.get_short_instrument_name()
        if self.project is not None:
            project = self.project
        prov_hash = self.provenance_id

        t = Time(self.mjd, format='mjd', scale='utc').datetime
        date = t.strftime('%Y%m%d')
        time = t.strftime('%H%M%S')

        filter = self.filter

        ra = self.ra
        ra_int, ra_frac = str(float(ra)).split('.')
        ra_int = int(ra_int)
        ra_int_h = ra_int // 15
        ra_frac = int(ra_frac)

        dec = self.dec
        dec_int, dec_frac = str(float(dec)).split('.')
        dec_int = int(dec_int)
        dec_int_pm = f'p{dec_int:02d}' if dec_int >= 0 else f'm{dec_int:02d}'
        dec_frac = int(dec_frac)

        default_convention = "{inst_name}_{date}_{time}_{filter}_{prov_hash:.6s}"
        cfg = Config.get()
        name_convention = cfg.value( 'storage.exposures.name_convention', default=None )
        if name_convention is None:
            name_convention = default_convention

        filepath = name_convention.format(
            inst_name=inst_name,
            project=project,
            date=date,
            time=time,
            filter=filter,
            ra=ra,
            ra_int=ra_int,
            ra_int_h=ra_int_h,
            ra_frac=ra_frac,
            dec=dec,
            dec_int=dec_int,
            dec_int_pm=dec_int_pm,
            dec_frac=dec_frac,
            prov_hash=prov_hash,
        )

        if self.components is None:
            if self.format == 'fits':
                filepath += ".fits"
            else:
                raise ValueError( f"Unknown format for exposures: {self.format}" )

        return filepath


    def _file_suffix( self, comp=None ):
        if self.format == 'fits':
            return ".fits"
        elif self.format == 'fitsfz':
            return ".fits.fz"
        raise ValueError( f"Don't know suffix for exposure format {self.format}" )


    def save( self, *args, **kwargs ):
        """Save an exposure to the local file store and the archive.

        Two ways to call:

          - one or more positional parameters.  The number of positional
            parameters must match the length of
            self.components, or be one if
            self.components is None.  Each parameter is either
            a binary blob with what should be written to the file, or a
            Path or str pointing to where the file currently exists on
            disk.

          - No positional parameters.  In this case, it will assume that
            file (or files, if filepath_etensions is not None) are
            already in the right place on disk, and it will use
            self.get_fullpath(nofile=True) to find them.

        Keyword parameters are passed on to FileOnDiskMixin.save().  Do *not*
        include an "component" keyword in **kwargs; that is handled internally.

        """

        data = None
        datas = None
        if len(args) > 0:
            if self.components is None:
                if len(args) != 1:
                    raise ValueError( f"components is None but {len(args)}>1 positional parameters supplied" )
                data = args[0]
            else:
                if len(args) != len( self.components ):
                    raise ValueError( f"components has length {len(self.components)}, but "
                                      f"{len(args)} positional parameters were supplied" )
                datas = args
        else:
            if self.components is None:
                data = self.get_fullpath( nofile=True )
            else:
                datas = self.get_fullpath( nofile=True )
                # datas = [ self.get_fullpath( nofile=True, extension=e ) for e in self.components ]

        if self.components is None:
            FileOnDiskMixin.save( self, data, **kwargs )
        else:
            self.md5sum_components = [ None ] * len( self.components )
            for comp, data in zip( self.components, datas ):
                FileOnDiskMixin.save( self, data, component=comp, **kwargs )


    def load(self, section_ids=None):
        # Thought required: if exposures are going to be on the archive,
        #  then we're going to need to call self.get_fullpath() to make
        #  sure the exposure has been downloaded from the archive to
        #  local storage.
        if section_ids is None:
            section_ids = self.instrument_object.get_section_ids()

        if not isinstance(section_ids, list):
            section_ids = [section_ids]

        if not all([isinstance(sec_id, (str, int)) for sec_id in section_ids]):
            raise ValueError("section_ids must be a list of integers. ")

        if self.filepath is not None:
            # Putting this error in to catch if we actually ever actually call this.
            # This is a scary function, since it will load the entire exposure into
            # memory at once.  That may be what somebody wants, but often isn't.
            # If this error is caught, then further thought is required.
            # If the error is never caught, we might be able to delete the whole
            # "load" method.
            raise RuntimeError( "Do you really want to load the whole exposure?" )
            for i in section_ids:
                self.data[i]  # use the SectionData __getitem__ method to load the data
        else:
            raise ValueError("Cannot load data from database without a filepath! ")

    def _comp_filepath( self, which='image' ):
        if which not in [ 'image', 'weight', 'flags' ]:
            raise ValueError( f"Unknown exposure file component {which}" )
        if self.instrument is None:
            raise ValueError("Cannot load data without an instrument! ")

        paths = self.get_fullpath( as_list=True )

        if self.components is None:
            if which == 'image':
                return paths[0]
            else:
                raise ValueError( f"Exposure has no components, so won't have a {which} file" )

        compdex = self.components.index( which )
        if compdex is None:
            raise ValueError( f"Failed to find {which} in {self.components}" )

        return paths[ compdex ]


    @property
    def data(self):
        if self._data is None:
            self._data = SectionData( self._comp_filepath('image'), self.instrument_object )
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, SectionData):
            raise ValueError(f"data must be a SectionData object. Got {type(value)} instead. ")
        self._data = value

    @property
    def section_headers(self):
        if self._section_headers is None:
            self._section_headers = SectionHeaders( self._comp_filepath('image'), self.instrument_object )
        return self._section_headers

    @section_headers.setter
    def section_headers(self, value):
        if not isinstance(value, SectionHeaders):
            raise ValueError(f"section_headers must be a SectionHeaders object. Got {type(value)} instead. ")
        self._section_headers = value

    @property
    def weight(self):
        if self._weight is None:
            self._weight = SectionData( self._comp_filepath('weight'), self.instrument_object )
        return self._weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, SectionData):
            raise ValueError(f"weight must be a SectionData object. Got {type(value)} instead. ")
        self._weight = value

    @property
    def weight_section_headers(self):
        if self._weight_section_headers is None:
            self._weight_section_headers = SectionHeaders( self._comp_filepath('weight'), self.instrument_object )
        return self._weight_section_headers

    @weight_section_headers.setter
    def weight_section_headers(self, value):
        if not isinstance(value, SectionHeaders):
            raise ValueError(f"weight_section_headers must be a SectionHeaders object. Got {type(value)} instead. ")
        self._weight_section_headers = value

    @property
    def flags(self):
        if self._flags is None:
            self._flags = SectionData( self._comp_filepath('flags'), self.instrument_object )
        return self._flags

    @flags.setter
    def flags(self, value):
        if not isinstance(value, SectionData):
            raise ValueError(f"flags must be a SectionData object. Got {type(value)} instead. ")
        self._flags = value

    @property
    def flags_section_headers(self):
        if self._flags_section_headers is None:
            self._flags_section_headers = SectionHeaders( self._comp_filepath('flags'), self.instrument_object )
        return self._flags_section_headers

    @flags_section_headers.setter
    def flags_section_headers(self, value):
        if not isinstance(value, SectionHeaders):
            raise ValueError(f"flags_section_headers must be a SectionHeaders object. Got {type(value)} instead. ")
        self._flags_section_headers = value

    def clear_cache( self ):
        if self._data is not None:
            self._data.clear_cache()
        if self._weight is not None:
            self._weight.clear_cache()
        if self._flags is not None:
            self._flags.clear_cache()

    @property
    def header(self):
        if self._header is None:
            self._header = read_fits_image(self.get_fullpath(), ext=0, output='header')
        if self._header is None:
            self._header = fits.Header()
        return self._header

    @header.setter
    def header(self, value):
        if not isinstance(value, fits.Header):
            raise ValueError(f"data must be a fits.Header object. Got {type(value)} instead. ")
        self._header = value

    def update_instrument(self, session=None):
        """Make sure the instrument object is up-to-date with the current database session.

        This will call the instrument's fetch_sections() method,
        using the given session and the exposure's MJD as dateobs.

        If there are SensorSections for this instrument on the DB,
        and if their validity range is consistent with this exposure's MJD,
        those sections will be loaded to the instrument.
        This must be called before loading any data.

        This function is called automatically when an exposure
        is loaded from the database.

        Parameters
        ----------
        session: sqlalchemy.orm.Session
            The database session to use.
            If None, will open a new session
            and close it at the end of the function.
        """
        if self.instrument is None:
            return
        with SmartSession(session) as session:
            self.instrument_object.fetch_sections(session=session, dateobs=self.mjd)

    @staticmethod
    def _do_not_require_file_to_exist():
        """Modify default behavior from FileOnDiskMixin.

        By default, new Exposure objects are generated
        with nofile=False, which means the file must exist
        at the time the Exposure object is created.
        This is the opposite default from the base class
        FileOnDiskMixin behavior.
        """
        return False

    def get_upstreams(self, session=None):
        """An exposure does not have any upstreams. """
        return []

    def get_downstreams(self, session=None):
        """An exposure has only Image objects as direct downstreams. """
        from models.image import Image

        with SmartSession(session) as session:
            images = session.scalars(sa.select(Image).where(Image.exposure_id == self.id)).all()

        return images
