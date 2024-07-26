

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

    @classmethod
    def get_meas_sub_provs( cls, alert_prov, meas_prov=None, sub_prov=None, session=None ):
        """Get the measurement and subtraction provenances associated with an alert provenance.

        Parameters
        ----------
          alert_prov: Provenance
             The 'alertsending' provenance

          meas_prov: Provenance or None
             If not None, will raise an exception if 'measuring' upstream of alert_prov
             doesn't match this.

          sub_prov: Provenance or None
             If not None, will raise an exception if 'subtracting' upstream of alert_prov
             doesn't match this

          session: Session (optional)

        Returns
        -------
          meas_prop, sub_prov: Provenance, Provenance

        """

        with SmartSession( session ) as sess:
            alert_prov = sess.safe_merge( alert_prov )
            meas_prov = sess.safe_merge( meas_prov ) if meas_prov is not None else None
            sub_prov = sess.safe_merge( sub_prov ) if sub_prov is not None else None
            
            found_meas_prov = None
            found_sub_prov = None
            for prov in alert_prov.upstreams:
                if prov.process == 'subtraction':
                    if found_sub_prov is not None:
                        raise RuntimeError( f"Error, alert sending provenance {alert_prov.id} has "
                                            f"multiple subtraction upstreams!" )
                    found_sub_prov = prov
                elif prov.process == 'measuring':
                    if found_meas_prov is not None:
                        raise RuntimeError( f"Error, alert sending provenance {alert_prov.id} has "
                                            f"multiple measuring upstreams!" )
                    found_meas_prov = prov

        if found_sub_prov is None:
            raise RuntimeError( f"Error, could not find subtraction upstream for "
                                f"alert sending provenance {alert_prov.id}" )
        if found_meas_prov is None:
            raise RuntimeError( f"Error, could not find measuring upstream for "
                                f"alert sending provenance {alert_prov.id}" )

        if ( meas_prov is not None ) and ( meas_prov.id != found_meas_prov.id ):
            raise ValueError( f"alert sending provenance {alert_prov.id} has upstream measuring provenance "
                              f"{found_meas_prov.id}, which doesn't match expected {meas_prov.id}" )
        if ( sub_prov is not None ) and ( sub_prov.id != found_sub_prov.id ):
            raise ValueError( f"alert sending provenance {alert_prov.id} has upstream subtracting provenance "
                              f"{sub_meas_prov.id}, which doesn't match expected {sub_prov.id}" )

        return ( found_meas_prov, found_sub_prov )
        

    # This is a class method, not an instance method, so we have the
    #  ability to reconsturct alerts outside of the context of
    #  a DataStore
    @classmethod
    def reconstruct_alert( cls, measurement, cutouts, alert_prov, meas_prov=None, sub_prov=None, resend=False,
                           image_bank=None, alert_schema=None, image_upper_limits=None, lookback_days=None,
                           session=None ):
        config = Config.get()
        if alert_schema is None:
            alert_schema = config.value( 'alertsending.alert_schema' )
        if image_upper_limits is None:
            image_upper_limits = config.value( 'alertsending.image_upper_limits' )
        if lookback_days is None:
            lookback_days = config.value( 'alertsending.lookback_days' )

        meas_prov, sub_prov = get_meas_sub_provs( alert_prov, meas_prov, sub_prov, session=session )
            
        # The structures below must match the scheama in share/SeeChange/*.avsc

        alertobj = None
        with SmartSession( session ) as sess:
            existing = ( session.query( Alert )
                         .filter( Alert.measurement_id==measurement.id )
                         .filter( Alert.provenance_id==provenance.id ) ).all()
            if len( existing ) > 0:
                alertobj = existing[]

            if alertobj is None:
                alertobj = Alert( measurement_id=measurement.id, time_sent=None, provenance_id=alert_prov.id )
                # Formally there's a race condition here where two different processes could be trying
                # to create the same alert at the same time, but in practice it's very unlikely we're goint
                # to have two different processes trying to send out an alert for the same Measurement
                # at the same time.  (Really, the race condition is bigger, if you think about checking
                # and setting the time the alert was sent if resend is False.)  Ignore it as not
                # worth worrying about.
                session.add( alertobj )
                session.flush()    # Make sure the alert gets an ID

            measurement = session.safe_merge( measurement )
            object = measurement.object

            oldmeasurements = []
            upperlimimages = []
            if ( lookback_days is not None ) and ( lookback_days > 0 ):
                oldmeasurements = ( sess.query( Measurement )
                                    .filter( Measurement.object_id == measurement.object_id )
                                    .filter( Measurement.mjd < measurement.mjd )
                                    .filter( Measurement.mjd >= measurements.mjd - lookback_days )
                                    .filter( Measurement.provenance_id == measurement.provenance_id ) ).all()
                

                
            

            
            alert = { 'alert_id': alertobj.id,
                      'object': 
                
                
            
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
            alert_prov = ds.get_provenance( 'alertsending', self.pars.critical_pars(), session=session )

            
            
            measurements = ds.get_measurements( session=session )
            if measurements is None:
                raise ValueError( f"Can't find a measurements list from datastore inputs: {ds.get_inputs()}" )
            if not all( [ m.provenance_id == meas_prov.id for m in measurements ] ):
                raise ValueError( f"Error, provenance of DataStore measurements don't match measuring "
                                  f"upstream of alert sending provenance {alert_prov.id}" )
            
            # Get a list of all previous images (within lookback_days)
            # that could overlap the image we're working on.  This is
            # probably a superset of the images that we'll need for any
            # given measurements object, but this way we can pull them
            # just once instead of redoing the search for each
            # measurements object.

            self.prev_subs = Image.find_potential_overlapping( ds.image, prov_id=sub_prov )
            

        except Exception as ex:
            raise RuntimeError( "Rob needs to figure out what really to do in this except block" )



