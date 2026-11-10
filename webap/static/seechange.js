import { seechange } from "./seechange_ns.js"
import { rkAuth } from "./rkauth.js"
import { rkWebUtil } from "./rkwebutil.js";
import { ImView } from "./imview.js";
import "./provenancetags.js";
import "./exposuresearch.js";
import "./exposurelist.js";
import "./exposure.js";
import "./diaforcedphot.js"
import "./conductor.js";

// Everything is going in the seechange namespace
//  (which is imported from seechange_ns.js so that
//   we can avoid circular imports).
// Because of how we do this, it's important to
//  import all the files from here, and only
//  to import rkwebutil and seechange_ns in the
//  various other .js files.  (seechange_start.js
//  is a special case.)


// **********************************************************************
// The organization of the webap on the screen, and in classes below,
// is hierarchical.  At the top is Context.  (Above and outside
// the hierarchy is user auth.)  Contextholds a tabbed div with several
// broad "applications", and those applications have tabbed divs inside them.
//
//
// Each class initializes itself with its constructor, and then actually
// renders itself with render_page().
//
//   Context
//     Conductor (admins only)
//     Exposure Search
//        Exposure List
//        Exposure
//     Provenance Tags
//

// **********************************************************************
// **********************************************************************
// **********************************************************************
// The global context

seechange.Context = class
{
    constructor()
    {
        this.authdiv = document.getElementById( "authdiv" );
        this.maindiv = document.getElementById( "pagebody" );
        this.rendered = false;

        // TODO : make this configurable?  Or at least remember how to
        //   detect in javascript what the URL you're running from is.  (In
        //   case the webap is not running as the root ap of the webserver.)
        this.connector = new rkWebUtil.Connector( "/" );
    };


    init()
    {
        let self = this;

        this.auth = new rkAuth( this.authdiv, "",
                                () => { self.render_page(); },
                                () => { window.location.reload(); } );
        this.auth.checkAuth();
    };


    render_page()
    {
        var self = this;
        let p, span, button;

        if ( this.rendered ) console.log( "WARNING: seechange.Context.render_page called more than once." );

        rkWebUtil.wipeDiv( this.authdiv );
        p = rkWebUtil.elemaker( "span", this.authdiv,
                                { "text": "Logged in as " + this.auth.username
                                  + " (" + this.auth.userdisplayname + ") — ",
                                  "classes": [ "italic" ] } );
        span = rkWebUtil.elemaker( "span", p,
                                   { "classes": [ "link" ],
                                     "text": "Log Out",
                                     "click": () => { self.auth.logout( () => { window.location.reload(); } ) }
                                   } );

        this.frontpagetabs = new rkWebUtil.Tabbed( this.maindiv );
        this.frontpagetabs.div.setAttribute( 'id', 'frontpagetabs' );

        if ( this.auth.usergroups.includes( 'admin' ) || this.auth.usergroups.includes( 'root' ) ) {
            this.conductor = new seechange.Conductor( this );
            this.conductor.render();
            this.frontpagetabs.addTab( "conductor", "Conductor", this.conductor.div );
        }

        this.exposuresearch = new seechange.ExposureSearch( this );
        this.exposuresearch.render();
        this.frontpagetabs.addTab( "exposuresearch", "Exposure/Image Search", this.exposuresearch.div );

        this.diaforcedphot = new seechange.DiaForcedPhotSearch( this );
        this.diaforcedphot.render();
        this.frontpagetabs.addTab( "diaforcedphot", "DIA Forced Photometry", this.diaforcedphot.div );

        this.provenancetags = new seechange.ProvenanceTags( this );
        this.provenancetags.render();
        this.frontpagetabs.addTab( "provtags", "Provenance Tags", this.provenancetags.div );


        let div = rkWebUtil.elemaker( "div", null );
        rkWebUtil.elemaker( "p", div, { "text": "42" } );
        this.frontpagetabs.addTab( "gratuitous", "Answer", div );

        this.frontpagetabs.selectTab( "exposuresearch" );

        // This next element is used in our tests as a way of quickly figuring
        //   out that this function has finished.
        rkWebUtil.elemaker( "input", this.frontpagediv, { "id": "seechange_context_render_page_complete",
                                                          "attributes": { "type": "hidden" } } );
    }

    _image_data_getter( url )
    {
        return async () => {
            if ( ! ImView.is_little_endian() ) {
                throw new Error( "This webap doesn't work on a platform that's not little-endian." );
            }
            let rval = await this.connector.httpPromise( url, null, false );
            let dv = new DataView( rval );
            let height = dv.getUint16( 0, true );
            let width = dv.getUint16( 2, true );
            let data = new Float32Array( dv.buffer, 4, ( (dv.byteLength - 4) / 4 ) );
            return { 'data': data, 'width': width, 'height': height }
        }
    }

}

// **********************************************************************

seechange.prov_tag_wids = {};

seechange.ProvTagWid = class
{
    constructor( context=null, idtag=null, inprops={} )
    {
        let self = this;
        this.context = context;

        let props = { 'introtext': 'Use provenance tag: ',
                      'container': null,
                      'parent': null };
        rkWebUtil.paramParser( props, inprops, this );

        if ( idtag == null ) idtag = seechange.prov_tag_wids.length.toString();
        if ( seechange.prov_tag_wids.hasOwnProperty( idtag ) ) {
            window.alert( "Internal error making provenance tag widget.  Rob needs to fix this." );
            return;
        }
        this.idtag = idtag;
        seechange.prov_tag_wids[ idtag ] = this;

        if ( this.container == null ) {
            if ( this.parent == null ) {
                window.alert( "Error, seechange.ProvTagWid parent can't be null" );
                return;
            }
            this.container = rkWebUtil.elemaker( "p", this.parent, { "text": this.introtext } );
        } else {
            if ( this.container.parentNode == null ) {
                window.alert( "Error, seechange.ProvTagWid container parentNode can't be null" );
                return;
            }
            rkWebUtil.wipeDiv( this.container );
        }

        this.provtag_select = rkWebUtil.elemaker( "select", this.container, { 'id': 'provtag_select_' + idtag } );
        rkWebUtil.button( this.container, "Refresh Tags",
                          (e) => { self.refresh_all_provtag_selects(); },
                          [ "marginleftem" ] )
        this.loading_provtags = rkWebUtil.elemaker( "span", this.container, { "classes": [ "warning", "bold" ] } );
        this.loading_provtags.innerHTML = "Loading provtags...";

        this.observer = new MutationObserver( (mutlist, obs) => { self.check_mutation(mutlist, obs); } );
        this.observer.observe( this.container.parentNode, { 'childList': true } );

        this.context.connector.sendHttpRequest( "provtags", {}, (data) => { self.update_provtag_select(data) } );
    }

    get value()
    {
        return this.provtag_select.value;
    }
    
    refresh_all_provtag_selects()
    {
        let self = this;
        for ( let idtag of Object.keys( seechange.prov_tag_wids ) ) {
            seechange.prov_tag_wids[ idtag ].loading_provtags.innerHTML = "Loading provtags...";
        }
        this.context.connector.sendHttpRequest( "provtags", {}, (data) => { self.update_all_provtag_selects(data) } );
    }

    update_all_provtag_selects( data )
    {
        for ( let idtag of Object.keys( seechange.prov_tag_wids ) ) {
            seechange.prov_tag_wids[ idtag ].update_provtag_select( data );
        }
    }

    update_provtag_select( data )
    {
        this.loading_provtags.innerHTML = "";
        let cursel = this.provtag_select.value;
        rkWebUtil.wipeDiv( this.provtag_select );
        for ( let provtag of data.provenance_tags) {
            let option = rkWebUtil.elemaker( "option", this.provtag_select,
                                             { "text": provtag,
                                               "attributes": { "values": provtag } } );
            if ( provtag == cursel )
                option.setAttribute( "selected", "1" );
        }
    }

    check_mutation( mutlist, obs )
    {
        for ( let mutrec of mutlist ) {
            if ( mutrec.removedNodes.length > 0 ) {
                if ( ! this.container.parentNode ) {
                    rkWebUtil.wipeDiv( this.container );
                    this.observer.disconnect();
                    delete seechange.prov_tag_wids( this.idtag );
                    // We're done.  (Done for.)  (Toast.)
                    return;
                }
            }
        }
    }
                                                                 
}



// **********************************************************************
// Make this into a module

export { }

