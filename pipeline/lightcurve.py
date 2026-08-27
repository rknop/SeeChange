import uuid

from pipeline.parameters import Parameters


class ParsLightcurve(Parametrs):
    def __init__( self, **kwargs ):
        super().__init__()

        self.subzp_prov = self.add_par(
            name = 'subzp_prov_search',
            default = None,
            par_types = (str, None),
            docstring = ( "Provenances of the zeropoints of the *subtraction* images to include in the lightcurve, "
                          "produced by the search pipeline.  Set this to None to not use any subtraction images "
                          "in the lightcurve.  BE CAREFUL HERE.  If you give a provenance that has parameters "
                          "inconsistent with what we will naturally determine for the provenance of the subtraction "
                          "images made as part of lightcurve building, you will get a scary heterogeneous "
                          "lightcurve.  Safest is to leave this at None, but that's also the least efficient. "
                          "Know what you're doing and set the parameter accordingly." ),
            critical = True
        )

        self.only_existing_search_subtractions = self.add_par(
            name = "only_existing_search_subtractions",
            default = False,
            par_types = bool,
            docstring = ( "If True, the only look at existing search subtractions to find points to add to "
                          "the lightcurve.  This is fast, but will only produce lightcurves where we have happened "
                          "to do a subtraction" ),

            critical = True
        )

        self.only_existing_subtractions = self.add_par(
            name = "only_existing_subtractions",
            default = False,
            par_types = bool,
            docstring = ( "Don't do any new subtractions, only look at existing subtractions to do forced "
                          "photometry on.  This will include both subtractions that were already done for..."
                          "ROB YOU WERE EDITING HERE" ),
            critical = True
        )


        self.referencing_config = self.add_par(
            name = "referencing_config",
            default = {},
            par_types = dict,
            docstring = ( "A dictionary with referncing config.  Lightcurve will start with the base "
                          "referencing config, which is for searching.  Then, values here will override "
                          "anything that's in the base config before the RefMaker is actually made." ),
            critical = True
        )

        self.full_referencing_config = self.ad_par(
            name = "full_referencing_config",
            default = {},
            par_types = dict,
            docstring = ( "DO NOT SET ANYTHING HERE.  This is here so that the full referencing config "
                          "(built by combining the base referencing config with what's in the referencing_config "
                          "parameter) will be part of the provenance.  The value of this parameter will "
                          "be filled in at runtime, and anything you set will be blown away.  Use referencing_config "
                          "to actually set values." ),
            critical = True
        )

        self.mjd0 = self.add_par(
            name = "mjd0",
            default = None,
            par_types = ( float, None ),
            docstring = ( "The earliest mjd of images to include in the lightcurve." ),
            critical = False
        )

        self.mjd1 = self.add_par(
            name = "mjd1",
            default = None,
            par_types = ( float, None ),
            docstring = ( "The latest mjd of images to include in the lightcurve." ),
            critical = False
        )

        self.filter = self.add_par(
            name = "filter",
            default = "_filter_not_set_this_is_bad_",
            par_types = str,
            docstring = ( "The filter of the lightcurve to build." ),
            critical = False
        )

        self.object_id = self.add_par(
            name = "object_id",
            default = None
            par_types = ( uuid.UUID, str, None ),
            docstring = ( "The id of the object to build a lightcurve for.  Specify either this, object_name, "
                          "or ra and dec." ),
            critical = False
        )

        self.object_name = self.add_par(
            name = "object_name",
            default = None,
            par_types = ( str, None ),
            docstring = ( "The name of the object to build a lightcurve for.  Specify either this, object_id, "
                          "or ra and dec." ),
            critical = False
        )

        self.ra = self.add_par(
            name = "ra",
            default = None
            par_types = ( float, None ),
            docstring = ( "The ra of the object to build a lightcurve for.  Specify exactly one of object_id, "
                          "object_name, or (ra and dec)." ),
            critical = False
        )

        self.dec = self.add_par(
            name = "dec",
            default = None
            par_types = ( float, None ),
            docstring = ( "The ra of the object to build a lightcurve for.  Specify exactly one of object_id, "
                          "object_name, or (ra and dec)." ),
            critical = False
        )
