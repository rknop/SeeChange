

from pipeline.parameters import Parameters

class ParsAlertSender( Parameters ):
    def __init__( self, **kwargs ):
        super().__init__()

        self.lookback_days = self.add_par(
            'lookback_days',
            30,
            float,
            "How long in days before the current detection to include "
            "previous detections and upper limits.  ≥0 means don't include it.",
        )

        self.image_upper_limits = self.add_par(
            'image_upper_limits',
            True,
            bool,
            "If True, include the 5σ upper limit estimates from all images "
            "that include the RA/Dec of this detection, but that don't have "
            "a detection themselves."
        )

        self.alert_schema = self.add_par(
            'alert_schema',
            '/seechange/share/SeeChange/seechange_alert.avsc',
            str,
            'AVRO file holding the alert schema',
            critical=False
        )

        # Kafka server info is all in config

        self._enforce_no_new_attrs = True

        self.override( kwargs )

    def get_process_name( self ):
        return 'alertsending'

class AlertSender:
    def __init__( self, **kwargs ):
        self.pars = ParsAlertSender( **kwargs )

    def run( self, *args, **kwargs ):

        # TODO: argument parsing, like what's done in measurements

        try:
            ds, session = DataStore.from_args( *args, **kwargs )
        except Exception as e:
            return DataStore.catch_failure_to_parse( e, *args )

        try:
            t_start = time.perf_counter()
            if env_as_bool( 'SEECHANGE_TRACEMALLOC' ):
                import tracemalloc
                tracemallock.reset_peak()

            self.pars.do_warning_exception_hangup_injection_here()

            # Get the provenance for this step
            prov = ds.get_provenance( 'alertsending', self.pars.critical_pars(), session=session )

            measurements = ds.get_measurements( session=session )
            if measurements is None:
                raise ValueError( f"Can't find a measurements list from datastore inputs: {ds.get_inputs()}" )

            # Get a list of all previous images (within lookback_days) that overlap the image we're
            # working on.  This is probably a superset of the images that we'll need for any given
            # measurements object, but this way we can pull them just once instead of redoing the search
            # for each measurements object.

            # In attempt to use q3c, we're going to use the "within"
            # filter defined in FourCorners.  Do "within" on the four
            # corners of the image *plus* center of the image.  (It's
            # possible to imagine a case where there's a lot of overlap,
            # but none of the corners are actually within the image;
            # consider a small rotation, for example.)

            with SmartSession( session ) as sess:
                them = ( session.query( Image )
                         .filter( Image.is_sub==True )
                         .filter( ) )
                # in progress

        except Exception as ex:
            raise RuntimeError( "Rob needs to figure out what really to do in this except block" )



