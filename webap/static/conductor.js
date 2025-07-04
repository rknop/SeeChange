import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js";


// **********************************************************************

seechange.Conductor = class
{

    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'conductordiv' } );
        this.connector = this.context.connector;
        this.hide_exposure_details_checkbox = null;
    };

    // **********************************************************************

    render()
    {
        let self = this;

        let p, h3, span, hbox, vbox, subhbox, table, tr, td;

        rkWebUtil.wipeDiv( this.div );
        this.frontpagediv = rkWebUtil.elemaker( "div", this.div );

        this.pipelineworkers = new seechange.PipelineWorkers( this.context, this );

        hbox = rkWebUtil.elemaker( "div", this.frontpagediv, { "classes": [ "hbox" ] } );
        this.pollingdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "conductorconfig" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "conductorconfig" ] } );
        h3 = rkWebUtil.elemaker( "h3", vbox, { "text": "Pipeline Config  " } );
        rkWebUtil.button( h3, "Refresh", () => { self.show_config_status() } );
        p = rkWebUtil.elemaker( "p", vbox, { "text": "Run through step " } )
        this.throughstep_select = rkWebUtil.elemaker( "select", p );
        for ( let step of seechange.Conductor.ALL_STEPS ) {
            rkWebUtil.elemaker( "option", this.throughstep_select,
                                { "text": step,
                                  "attributes": { "value": step,
                                                  "id": "throughstep_select",
                                                  "name": "throughstep_select",
                                                  "selected": ( step=='alerting' ) ? 1 : 0 } } );
        }

        hbox.appendChild( this.pipelineworkers.div );

        this.contentdiv = rkWebUtil.elemaker( "div", this.frontpagediv );

        rkWebUtil.elemaker( "hr", this.contentdiv );

        p = rkWebUtil.elemaker( "p", this.contentdiv );
        rkWebUtil.button( p, "Search", () => { self.update_known_exposures(); } );
        p.appendChild( document.createTextNode( " known exposures taken from " ) );
        this.knownexp_mintwid = rkWebUtil.elemaker( "input", p, { "attributes": { "size": 20 } } );
        this.knownexp_mintwid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDateUTC( self.knownexp_mintwid )
        } );
        p.appendChild( document.createTextNode( " to " ) );
        this.knownexp_maxtwid = rkWebUtil.elemaker( "input", p, { "attributes": { "size": 20 } } );
        this.knownexp_maxtwid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDateUTC( self.knownexp_maxtwid )
        } );
        p.appendChild( document.createTextNode( " UTC (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)" ) );

        p = rkWebUtil.elemaker( "p", this.contentdiv );
        this.search_criteria_shown = false;
        this.show_hide_search_criteria = rkWebUtil.button( p, "Show Additional Search Criteria",
                                                           (e) => {
                                                               if ( self.search_criteria_shown ) {
                                                                   self.show_hide_search_criteria.value = (
                                                                       "Show Additional Search Criteria" );
                                                                   self.search_criteria_div.classList.remove(
                                                                       "dispblock" );
                                                                   self.search_criteria_div.classList.add(
                                                                       "dispnone" );
                                                                   self.search_criteria_shown = false;
                                                               } else {
                                                                   self.show_hide_search_criteria.value = (
                                                                       "Hide Addditional Search Criteria" );
                                                                   self.search_criteria_div.classList.remove(
                                                                       "dispnone" );
                                                                   self.search_criteria_div.classList.add(
                                                                       "dispblock" );
                                                                   self.search_criteria_shown = true;
                                                               } } );
        this.search_criteria_div = rkWebUtil.elemaker( "div", p,
                                                       { "classes": [ "midborder", "dispnone" ] } );
        table = rkWebUtil.elemaker( "table", this.search_criteria_div );
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "instrument:" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.search_instrument = rkWebUtil.elemaker( "input", td, { "attributes": { "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "target:" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.search_target = rkWebUtil.elemaker( "input", td, { "attributes": { "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "filter:" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.search_filter = rkWebUtil.elemaker( "input", td, { "attributes": { "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "project:" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.search_project = rkWebUtil.elemaker( "input", td, { "attributes": { "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "exp time ≥" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.min_exp_time = rkWebUtil.elemaker( "input", td, { "attributes": { "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "state:" } );
        td = rkWebUtil.elemaker( "td", tr );
        for ( let i of [ "held", "ready", "claimed", "running", "done" ] ) {
            this['search_state_' + i] =
                rkWebUtil.elemaker( "input", td, { "attributes": { "type": "checkbox",
                                                                   "id": "search_state_" + i,
                                                                   "checked": 1 } } );
            rkWebUtil.elemaker( "label", td, { "text": i, "attributess": { "for": "search_state_" + i } } );
        }
        tr = rkWebUtil.elemaker( "tr", table );
        td = rkWebUtil.elemaker( "td", tr, { "classes": [ "right" ], "text": "claim time ≤" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.max_claim_time = rkWebUtil.elemaker( "input", td, { "attributes": { "size": 20 } } );
        this.max_claim_time.addEventListener( "blur",
                                              (e) => { rkWebUtil.validateWidgetDateUTC( self.max_claim_time ) } );


        this.knownexp_notification_div = rkWebUtil.elemaker( "div", this.contentdiv );
        this.knownexpdiv = rkWebUtil.elemaker( "div", this.contentdiv );

        this.show_config_status();
        this.pipelineworkers.render();
    }

    // **********************************************************************

    show_config_status( edit=false )
    {
        var self = this;

        let p;

        rkWebUtil.wipeDiv( this.pollingdiv )
        rkWebUtil.elemaker( "p", this.pollingdiv,
                            { "text": "Loading status...",
                              "classes": [ "warning", "bold", "italic" ] } )

        if ( edit )
            this.connector.sendHttpRequest( "conductor/status", {}, (data) => { self.edit_config_status(data) } );
        else
            this.connector.sendHttpRequest( "conductor/status", {},
                                            (data) => { self.actually_show_config_status(data) } );
    }

    // **********************************************************************

    actually_show_config_status( data )
    {
        let self = this;

        let table, tr, th, td, p;

        rkWebUtil.wipeDiv( this.pollingdiv );
        rkWebUtil.elemaker( "h3", this.pollingdiv,
                            { "text": "Conductor polling config" } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        rkWebUtil.button( p, "Refresh", () => { self.show_config_status() } );
        p.appendChild( document.createTextNode( "  " ) );
        rkWebUtil.button( p, "Modify", () => { self.show_config_status( true ) } );

        if ( data.pause )
            rkWebUtil.elemaker( "p", this.pollingdiv, { "text": "Automatic updating is paused." } )
        if ( data.hold )
            rkWebUtil.elemaker( "p", this.pollingdiv, { "text": "Newly added known exposures are being held." } )

        let instrument = ( data.instrument == null ) ? "" : data.instrument;
        let minmjd = "(None)";
        let maxmjd = "(None)";
        let minexptime = "(None)";
        let projects = "(Any)";
        if ( data.updateargs != null ) {
            minmjd = data.updateargs.hasOwnProperty( "minmjd" ) ? data.updateargs.minmjd : minmjd;
            maxmjd = data.updateargs.hasOwnProperty( "maxmjd" ) ? data.updateargs.maxmjd : maxmjd;
            minexptime = data.updateargs.hasOwnProperty( "minexptime" ) ? data.updateargs.minexptime : minexptime;
            projects = data.updateargs.hasOwnProperty( "projects" ) ? data.updateargs.projects.join(",") : projects;
        }

        table = rkWebUtil.elemaker( "table", this.pollingdiv );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Instrument" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": data.instrument } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Min MJD" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": minmjd } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Max MJD" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": maxmjd } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Min Exp. Time" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": minexptime } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Projects" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": projects } );

        this.forceconductorpoll_p = rkWebUtil.elemaker( "p", this.pollingdiv );
        rkWebUtil.button( this.forceconductorpoll_p, "Force Conductor Poll", () => { self.force_conductor_poll(); } );

        this.throughstep_select.value = data.throughstep;
        this.partial_pickup_checkbox = ( data.pickuppartial ? 1 : 0 );
    }

    // **********************************************************************

    edit_config_status( data )
    {
        let self = this;

        let table, tr, th, td, p;

        rkWebUtil.wipeDiv( this.pollingdiv );
        rkWebUtil.elemaker( "h3", this.pollingdiv,
                            { "text": "Conductor polling config" } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        rkWebUtil.button( p, "Save Changes", () => { self.update_conductor_config(); } );
        p.appendChild( document.createTextNode( "  " ) );
        rkWebUtil.button( p, "Cancel", () => { self.show_config_status() } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        this.status_pause_wid = rkWebUtil.elemaker( "input", p, { "attributes": { "type": "checkbox",
                                                                                  "id": "status_pause_checkbox" } } );
        if ( data.pause ) this.status_pause_wid.setAttribute( "checked", "checked" );
        rkWebUtil.elemaker( "label", p, { "text": "Pause automatic updating",
                                          "attributes": { "for": "status_pause_checkbox" } } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        this.status_hold_wid = rkWebUtil.elemaker( "input", p, { "attributes": { "type": "checkbox",
                                                                                 "id": "status_hold_checkbox" } } );
        if ( data.hold ) this.status_hold_wid.setAttribute( "checked", "checked" );
        rkWebUtil.elemaker( "label", p, { "text": "Hold newly added exposures",
                                          "attributes": { "for": "status_hold_checkbox" } } );


        let minmjd = "";
        let maxmjd = "";
        let minexptime = "";
        let projects = "";
        if ( data.updateargs != null ) {
            minmjd = data.updateargs.hasOwnProperty( "minmjd" ) ? data.updateargs.minmjd : minmjd;
            maxmjd = data.updateargs.hasOwnProperty( "maxmjd" ) ? data.updateargs.maxmjd : maxmjd;
            minexptime = data.updateargs.hasOwnProperty( "minexptime" ) ? data.updateargs.minexptime : minexptime;
            projects = data.updateargs.hasOwnProperty( "projects" ) ? data.updateargs.projects.join(",") : projects;
        }
        let instrument = ( data.instrument == null ) ? "" : data.instrument;

        table = rkWebUtil.elemaker( "table", this.pollingdiv );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Instrument" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_instrument_wid = rkWebUtil.elemaker( "input", td,
                                                         { "attributes": { "value": instrument,
                                                                           "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Start time" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_minmjd_wid = rkWebUtil.elemaker( "input", td,
                                                     { "attributes": { "value": minmjd,
                                                                       "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " (MJD or YYYY-MM-DD HH:MM:SS)" } )
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "End time" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_maxmjd_wid = rkWebUtil.elemaker( "input", td,
                                                     { "attributes": { "value": maxmjd,
                                                                       "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " (MJD or YYYY-MM-DD HH:MM:SS)" } )
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Min Exp. Time" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_minexptime_wid = rkWebUtil.elemaker( "input", td,
                                                         { "attributes": { "value": minexptime,
                                                                           "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " seconds" } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Projects" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_projects_wid = rkWebUtil.elemaker( "input", td,
                                                       { "attributes": { "value": projects,
                                                                         "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " (comma-separated)" } );
    }


    // **********************************************************************

    update_conductor_config()
    {
        let self = this;

        let instrument = this.status_instrument_wid.value.trim();
        instrument = ( instrument.length == 0 ) ? null : instrument;

        // Parsing is often verbose
        let minmjd = this.status_minmjd_wid.value.trim();
        if ( minmjd.length == 0 )
            minmjd = null;
        else if ( minmjd.search( /^ *([0-9]*\.)?[0-9]+ *$/ ) >= 0 )
            minmjd = parseFloat( minmjd );
        else {
            try {
                minmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( minmjd ) );
            } catch (e) {
                window.alert( e );
                return;
            }
        }

        let maxmjd = this.status_maxmjd_wid.value.trim();
        if ( maxmjd.length == 0 )
            maxmjd = null;
        else if ( maxmjd.search( /^ *([0-9]*\.)?[0-9]+ *$/ ) >= 0 )
            maxmjd = parseFloat( maxmjd );
        else {
            try {
                maxmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( maxmjd ) );
            } catch (e) {
                window.alert( e );
                return;
            }
        }

        let minexptime = this.status_minexptime_wid.value.trim();
        minexptime = ( minexptime.length == 0 ) ? null : parseFloat( minexptime );

        let projects = this.status_projects_wid.value.trim();
        if ( projects.length == 0 )
            projects = null;
        else {
            let tmp = projects.split( "," );
            projects = [];
            for ( let project of tmp ) projects.push( project.trim() );
        }

        let params = {};
        if ( minmjd != null ) params['minmjd'] = minmjd;
        if ( maxmjd != null ) params['maxmjd'] = maxmjd;
        if ( minexptime != null ) params['minexptime'] = minexptime;
        if ( projects != null ) params['projects'] = projects;
        if ( Object.keys(params).length == 0 ) params = null;

        this.connector.sendHttpRequest( "conductor/updateparameters",
                                        { 'instrument': instrument,
                                          'pause': this.status_pause_wid.checked ? 1 : 0,
                                          'hold': this.status_hold_wid.checked ? 1 : 0,
                                          'updateargs': params },
                                        () => self.show_config_status() );
    }

    // **********************************************************************

    force_conductor_poll()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.forceconductorpoll_p );
        rkWebUtil.elemaker( "span", this.forceconductorpoll_p,
                            { "text": "...forcing conductor poll...",
                              "classes": [ "warning", "bold", "italic" ] } );
        this.connector.sendHttpRequest( "conductor/forceupdate", {}, () => self.did_force_conductor_poll() );
    }

    // **********************************************************************

    did_force_conductor_poll()
    {
        let self = this;
        rkWebUtil.wipeDiv( this.forceconductorpoll_p );
        rkWebUtil.button( this.forceconductorpoll_p, "Force Conductor Poll", () => { self.force_conductor_poll(); } );
        this.update_known_exposures();
    }


    // **********************************************************************

    update_known_exposures()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.knownexpdiv );
        let p = rkWebUtil.elemaker( "p", this.knownexpdiv,
                                    { "text": "Loading known exposures...",
                                      "classes": [ "warning", "bold", "italic" ] } );
        let url = "conductor/getknownexposures";
        if ( this.knownexp_mintwid.value.trim().length > 0 ) {
            let minmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( this.knownexp_mintwid.value ) );
            url += "/minmjd=" + encodeURIComponent( minmjd.toString() );
        }
        if ( this.knownexp_maxtwid.value.trim().length > 0 ) {
            let maxmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( this.knownexp_maxtwid.value ) );
            url += "/maxmjd=" + encodeURIComponent( maxmjd.toString() );
        }
        if ( this.search_instrument.value.trim().length > 0 ) {
            url += "/instrument=" + encodeURIComponent( this.search_instrument.value.trim() )
        }
        if ( this.search_target.value.trim().length > 0 ) {
            url += "/target=" + encodeURIComponent( this.search_target.value.trim() );
        }
        if ( this.search_filter.value.trim().length > 0 ) {
            url += "/filter=" + encodeURIComponent( this.search_filter.value.trim() );
        }
        if ( this.search_project.value.trim().length > 0 ) {
            url += "/project=" + encodeURIComponent( this.search_project.value.trim() );
        }
        if ( this.min_exp_time.value.trim().length > 0 ) {
            url += "/minexptime=" + encodeURIComponent( this.min_exp_time.value.trim() );
        }
        if ( this.max_claim_time.value.trim().length > 0 ) {
            url += "/maxclaimtime=" + encodeURIComponent( this.max_claim_time.value.trim() );
        }
        let searchstate = [];
        for ( let i of [ "held", "ready", "claimed", "running", "done" ] ) {
            if ( this['search_state_' + i].checked ) searchstate.push( i );
        }
        if ( searchstate.length > 0 ) {
            url += "/state=" + encodeURIComponent( searchstate.join(",") );
        }
        this.connector.sendHttpRequest( url, {}, (data) => { self.show_known_exposures(data); } );
    }

    // **********************************************************************

    show_known_exposures( data )
    {
        let self = this;

        let tr, td, p, button, span, ttspan, hide_exposure_details;

        this.known_exposures = [];
        this.known_exposures_sort_order = [ '+mjd' ];
        this.known_exposure_checkboxes = {};
        this.known_exposure_rows = {};
        this.known_exposure_state_tds = {};
        // this.known_exposure_checkbox_manual_state = {};

        rkWebUtil.wipeDiv( this.knownexpdiv );

        p = rkWebUtil.elemaker( "p", this.knownexpdiv );

        // Uncomment this (...or move it somewhere) when it's actually properly implemented
        // if ( this.hide_exposure_details_checkbox == null ) {
        //     this.hide_exposure_details_checkbox =
        //         rkWebUtil.elemaker( "input", null,
        //                             { "change": () => { self.show_known_exposures( data ) },
        //                               "attributes": { "type": "checkbox",
        //                                               "id": "knownexp-hide-exposure-details-checkbox" } } );
        // }
        // p.appendChild( this.hide_exposure_details_checkbox );
        // p.appendChild( document.createTextNode( "Hide exposure detail columns    " ) );
        // hide_exposure_details = this.hide_exposure_details_checkbox.checked;

        this.select_all_checkbox = rkWebUtil.elemaker( "input", p,
                                                       { "attributes": {
                                                             "type": "checkbox",
                                                             "id": "knownexp-select-all-checkbox" } } );
        rkWebUtil.elemaker( "label", p, { "text": "Select all",
                                          "attributes": { "for": "knownexp-select-all-checkbox" } } );
        this.select_all_checkbox.addEventListener(
            "change",
            () => {
                for ( let ke of self.known_exposures ) {
                    self.known_exposure_checkboxes[ ke.id ].checked = self.select_all_checkbox.checked;
                }
            } );
        p.appendChild( document.createTextNode( "      Apply to selected: " ) );
        button = rkWebUtil.button( p, "Delete Selected", () => { self.delete_known_exposures() } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Clear Cluster Claim On Selected", () => { self.clear_cluster_claim() } );
        button.classList.add( "hmargin" );
        p.appendChild( document.createTextNode( "      " ) );
        button = rkWebUtil.button( p, "Set", () => { self.set_exposure_state(); } )
        button.classList.add( "hmargin" );
        p.appendChild( document.createTextNode( "selected state to" ) );
        this.set_known_exposure_state_to = rkWebUtil.elemaker( "select", p );
        rkWebUtil.elemaker( "option", this.set_known_exposure_state_to,
                            { "text": "—", "attributes": { "value": "—", "selected": 1 } } );
        rkWebUtil.elemaker( "option", this.set_known_exposure_state_to,
                            { "text": "held", "attributes": { "value": "held" } } );
        rkWebUtil.elemaker( "option", this.set_known_exposure_state_to,
                            { "text": "ready", "attributes": { "value": "ready" } } );
        // Do we want to give the conductor user the ability to set the following three states???
        // I'm inclined towards yes... it's an admin function, and as they say,
        //   with great responsibility comes great power.  Or something.  (If only.)
        rkWebUtil.elemaker( "option", this.set_known_exposure_state_to,
                            { "text": "claimed", "attributes": { "value": "claimed" } } );
        rkWebUtil.elemaker( "option", this.set_known_exposure_state_to,
                            { "text": "running", "attributes": { "value": "running" } } );
        rkWebUtil.elemaker( "option", this.set_known_exposure_state_to,
                            { "text": "done", "attributes": { "value": "done" } } );

        for ( let ke of data.knownexposures ) {
            this.known_exposures.push( ke );
        }

        let rowrenderer = ( ke ) => {
            tr = rkWebUtil.elemaker( "tr", null );
            if ( ke.state == "held" ) tr.classList.add( "heldexposure" );
            else if ( ke.state == "ready" ) tr.classList.add( "readyexposure" );
            td = rkWebUtil.elemaker( "td", tr );
            self.known_exposure_checkboxes[ ke.id ] =
                rkWebUtil.elemaker( "input", td, { "attributes": { "type": "checkbox" } } );
            // (For debugging.)
            // self.known_exposure_checkbox_manual_state[ ke.id ] = 0;
            // self.known_exposure_checkboxes[ ke.id ].addEventListener(
            //     "click", () => {
            //         self.known_exposure_checkbox_manual_state[ ke.id ] =
            //             ( self.known_exposure_checkboxes[ ke.id ].checked ? 1 : 0 );
            //         console.log( "Setting " + ke.id + " to " + self.known_exposure_checkboxes[ ke.id ].checked );
            //     } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.state, "classes": [ "state" + ke.state ] } );
            self.known_exposure_state_tds[ ke.id ] = td;
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.instrument } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.identifier } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.mjd ).toFixed( 5 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.target } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.ra ).toFixed( 5 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.dec ).toFixed( 5 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.gallat ).toFixed( 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.filter } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.exp_time ).toFixed( 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.project } );
            td = rkWebUtil.elemaker( "td", tr );
            span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ], "text": ke.cluster_id } );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } )
            ttspan.innerHTML = "node: " + ke.node_id + "<br>machine: " + ke.machine_id;
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": ( ke.claim_time == null ) ?
                                       "" : rkWebUtil.dateUTCFormat(rkWebUtil.parseDateAsUTC(ke.claim_time)) } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": ( ke.release_time == null ) ?
                                       "" : rkWebUtil.dateUTCFormat(rkWebUtil.parseDateAsUTC(ke.release_time)) } );
            td = rkWebUtil.elemaker( "td", tr );
            if ( ke.exposure_id != null ) {
                let a = rkWebUtil.elemaker( "a", td,
                                            { 'classes': [ "link" ],
                                              'text': ke.filepath,
                                              'click': (e) => {
                                                  self.context.frontpagetabs.selectTab("exposuresearch");
                                                  let el = self.context.exposuresearch.exposurelist;
                                                  el.show_exposure( ke.exposure_id,
                                                                    self.context.provtag_wid.value );
                                              }
                                            } );
            }
            self.known_exposure_rows[ ke.id ] = tr;
            return tr;
        }

        let fields = [ '', 'state', 'instrument', 'identifier', 'mjd', 'target', 'ra', 'dec', 'b',
                       'filter', 'exp_time', 'project', 'cluster', 'claim_time', 'release_time',
                       'exposure' ];
        let nosortfields = [ '', 'state' ];
        let fieldmap = { 'instrument': 'instrument',
                         'identifier': 'identifier',
                         'mjd': 'mjd',
                         'target': 'target',
                         'ra': 'ra',
                         'dec': 'dec',
                         'b': 'gallat',
                         'filter': 'filter',
                         'exp_time': 'exp_time',
                         'project': 'project',
                         'cluster': 'cluster_id',
                         'claim_time': 'claim_time',
                         'release_time': 'release_time',
                         'exposure': 'exposure_id',
                       };
        let tab = new rkWebUtil.SortableTable( this.known_exposures, rowrenderer, fields,
                                               { 'fieldmap': fieldmap,
                                                 'dictoflists': false,
                                                 'nosortfields': nosortfields,
                                                 'initsort': [ '+mjd' ],
                                                 'colorclasses': [ 'bgfade', 'bgwhite' ],
                                                 'colorlength': 3 } );
        this.knownexpdiv.appendChild( tab.table );
    }

    // **********************************************************************

    set_exposure_state()
    {
        let self = this;
        let state = this.set_known_exposure_state_to.value;
        let kexps = []
        for ( let ke of this.known_exposures ) {
            if ( this.known_exposure_checkboxes[ ke.id ].checked )
                kexps.push( ke.id );
        }
        rkWebUtil.wipeDiv( this.knownexp_notification_div );
        rkWebUtil.elemaker( "span", this.knownexp_notification_div,
                            { "text": "Updating state of some known exposures...",
                              "classes": [ "bold", "italic", "warning" ] } );
        if ( kexps.length > 0 ) {
            this.connector.sendHttpRequest( "conductor/setknownexposurestate",
                                            { "state": state, "knownexposure_ids": kexps },
                                            (data) => { self.process_set_exposure_state( data ) } );
        }
    }

    // **********************************************************************

    process_set_exposure_state( data )
    {
        for ( let keid of data.knownexposure_ids ) {
            if ( this.known_exposure_rows.hasOwnProperty( keid ) ) {
                this.known_exposure_rows[keid].classList.remove( ...[ "heldexposure", "readyexposure" ] )
                if ( data.state == "held" ) this.known_exposure_rows[ keid ].classList.add( "heldexposure" );
                else if ( data.state == "ready" ) this.known_exposure_rows[keid].classList.add( "readyexposure" );
                for ( let i of [ "held", "ready", "claimed", "running", "done" ] )
                    this.known_exposure_state_tds[ keid ].classList.remove( "state" + i );
                this.known_exposure_state_tds[ keid ].innerHTML = data.state;
                this.known_exposure_state_tds[ keid ].classList.add( "state" + data.state );
            }
        }
        rkWebUtil.wipeDiv( this.knownexp_notification_div );
    }

    // **********************************************************************

    delete_known_exposures()
    {
        let self = this;

        let todel = [];
        for ( let ke of this.known_exposures ) {
            if ( this.known_exposure_checkboxes[ ke.id ].checked )
                todel.push( ke.id );
        }

        if ( todel.length > 0 ) {
            if ( window.confirm( "Delete " + todel.length.toString() + " known exposures? " +
                                 "(This cannot be undone.)" ) )
                this.connector.sendHttpRequest( "conductor/deleteknownexposures", { 'knownexposure_ids': todel },
                                                (data) => { self.process_delete_known_exposures(data, todel) } );
        }
    }

    // **********************************************************************

    process_delete_known_exposures( data, todel )
    {
        for ( let keid of todel ) {
            // Ugh, n²
            let dex = 0;
            while ( dex < this.known_exposures.length ) {
                if ( this.known_exposures[dex].id == keid )
                    this.known_exposures.splice( dex, 1 );
                else
                    dex += 1;
            }
            this.known_exposure_rows[ keid ].parentNode.removeChild( this.known_exposure_rows[ keid ] );
            delete this.known_exposure_rows[ keid ];
            delete this.known_exposure_checkboxes[ keid ];
            delete this.known_exposure_state_tds[ keid ];
        }
    }

    // **********************************************************************

    clear_cluster_claim()
    {
        let self = this;

        let toclear = [];
        for ( let ke of this.known_exposures) {
            if ( this.known_exposure_checkboxes[ ke.id ].checked )
                toclear.push( ke.id );
        }

        if ( toclear.length > 0 ) {
            if ( window.confirm( "Clear cluster claim on " + toclear.length.toString() + " known exposures?" ) ) {
                rkWebUtil.wipeDiv( this.knownexposidv );
                rkWebUtil.elemaker( "p", this.knownexpdiv,
                                    { "text": "Loading known exposures...",
                                      "classes": [ "warning", "bold", "italic" ] } );
                this.connector.sendHttpRequest( "/conductor/fullyclearclusterclaim",
                                                { 'knownexposure_ids': toclear },
                                                (data) => { self.update_known_exposures() } );
            }
        }
    }
}



// **********************************************************************

seechange.PipelineWorkers = class
{
    constructor( context, conductor )
    {
        this.context = context;
        this.conductor = conductor;
        this.connector = this.context.connector;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'conductorworkers-div',
                                                      'classes': [ 'conductorworkers' ] } )
    };

    // **********************************************************************

    render()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.div );
        let hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox" ] } );
        this.workersdiv = rkWebUtil.elemaker( "div", hbox, {} );
        this.update_workers();
    };

    // **********************************************************************

    update_workers()
    {
        let self = this;
        this.connector.sendHttpRequest( "conductor/getworkers", {}, (data) => { self.show_workers(data); } );
    }

    // **********************************************************************

    show_workers( data )
    {
        let self = this;
        let table, tr, th, td, p, h3;

        rkWebUtil.wipeDiv( this.workersdiv );

        h3 = rkWebUtil.elemaker( "h3", this.workersdiv, { "text": "Known Pipeline Workers  " } );
        rkWebUtil.button( h3, "Refresh", () => { self.update_workers(); } );

        table = rkWebUtil.elemaker( "table", this.workersdiv, { "classes": [ "borderedcells" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "cluster_id" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "node_id" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "last heartbeat" } );

        let grey = 0;
        let coln = 3;
        for ( let worker of data['workers'] ) {
            if ( coln == 0 ) {
                grey = 1 - grey;
                coln = 3;
            }
            coln -= 1;
            tr = rkWebUtil.elemaker( "tr", table );
            if ( grey ) tr.classList.add( "greybg" );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.cluster_id } );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.node_id } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": rkWebUtil.dateUTCFormat(
                                         rkWebUtil.parseDateAsUTC( worker.lastheartbeat ) ) } );
        }
    }

}

// **********************************************************************
// Keep this synced with top_level.py::Pipeline::ALL_STEPS

seechange.Conductor.ALL_STEPS = [ 'preprocessing', 'extraction', 'astrocal', 'photocal', 'subtraction',
                                  'detection', 'cutting', 'measuring', 'scoring', 'alerting' ];


// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { };
