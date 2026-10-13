import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"
import { SVGPlot } from "./svgplot.js"

// **********************************************************************

seechange.DiaForcedPhotSearch = class
{
    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox" ],
                                                      'id': 'diaforcedphotsearchdiv' } );
        this.ltcv_divs = {};
    }

    render()
    {
        let self = this;
        let hbox, box, vbox, p;

        hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "padex", "xmargin", "mostlyborder" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "padex", "xmargin", "mostlyborder" ] } );
        rkWebUtil.elemaker( "p", vbox, { "text": "Show lightcurve for object:" } );
        p = rkWebUtil.elemaker( "p", vbox )
        this.ltcvid = rkWebUtil.elemaker( "input", p, { "id": "diaforcedphot_ltcvid",
                                                        "attributes": { "type": "text", "size": 36} } );
        p = rkWebUtil.elemaker( "p", vbox )
        rkWebUtil.button( p, "Show Object", () => { self.show_lightcurve_for_object(); } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "padex", "xmargin", "mostlyborder" ] } );
        rkWebUtil.elemaker( "p", vbox, { "text": "(Searching not implemented yet)" } );

        this.ltcv_parent_div = rkWebUtil.elemaker( "div", this.div, { "classes": [ "vbox100p" ] } );
    }


    show_lightcurve_for_object()
    {
        // TODO checkif we already have this in ltcv_divs
        let self = this;
        let provtag = this.context.provtag_wid.value;
        let objid = this.ltcvid.value.trim();
        rkWebUtil.wipeDiv( this.ltcv_parent_div );
        rkWebUtil.elemaker( "h3", this.ltcv_parent_div, { "text": "Loading...",
                                                          "classes": [ "warning", "italic" ] } );
        this.context.connector.sendHttpRequest( "diaforcedphotforobj/" + objid + "/" + provtag, {},
                                                (data) => { self.actually_show_lightcurve( data ); } );
    }


    actually_show_lightcurve( data )
    {
        let ltcv = new seechange.DiaForcedPhot( this.context, data );
        this.ltcv_divs[ data.provtag + "/" + data.objid ] = ltcv.div;
        rkWebUtil.wipeDiv( this.ltcv_parent_div );
        this.ltcv_parent_div.appendChild( ltcv.div );
        ltcv.render();
    }

}

// **********************************************************************
    
seechange.DiaForcedPhot = class
{
    constructor( context, data )
    {
        this.context = context;
        this.data = data;
        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox" ] } );
    }

    render()
    {
        let self = this;
        
        rkWebUtil.elemaker( "h2", this.div, { "text": "DIA Forced Photometry for " + this.data.objname
                                              + " ( " + this.data.objid + " )" } );

        let hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "flexeven" ] } );
        this.ltcvdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                       "flexfitcontent", "minwid40p", "scrolly" ] } );
        this.cutoutsdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                          "flexfitcontent", "minwid40p",
                                                                          "scrolly" ] } );
        
        this.svgplot = new SVGPlot.Plot( { "name": "ltcv-" + this.data.objid,
                                           "divid": "ltcv-svgplot-" + this.data.objid,
                                           "showerrorbar": true,
                                           "xtitle": "MJD",
                                           "ytitle": "Flux (nJy)"
                                         } );
        this.ltcvdiv.appendChild( this.svgplot.topdiv );

        this.apervspsf = rkWebUtil.elemaker( "select", this.svgplot.buttonbox );
        rkWebUtil.elemaker( "option", this.apervspsf, { "text": "PSF photometry",
                                                        "attributes": { "value": "psf", "selected": 1 } } );
        rkWebUtil.elemaker( "option", this.apervspsf, { "text": "Aperture Photometry",
                                                        "attributes": { "value": "aperture" } } );
        this.apervspsf.addEventListener( "change", (e) => { self.select_aper_vs_psf(); } );
        

        let filterorder = [ 'u', 'b', 'v', 'g', 'r', 'i', 'z', 'y' ]
        let filtercolors = { 'u': "#8800cc",
                             'b': "#0000cc",
                             'v': "#88cc00",
                             'g': "#00cc00",
                             'r': "#cc0000",
                             'i': "#cc8800",
                             'z': "#884400",
                             'y': "#442200" }
        let filters = new Set();
        for ( let filt of this.data.filter ) {
            filters.add( filt );
            if ( ! filterorder.includes( filt ) ) {
                filterorder.push( filt );
                filtercolors[filt] = "#000000";  // TODO, be better
            }
        }

        this.filters = []
        this.psfphotids = {};
        this.psfsets = {};
        this.aperphotids = {};
        this.apersets = {};
        this.phottype = "psf";
        for ( let filt of filterorder ) {
            if ( filters.has( filt ) ) {
                this.filters.push( filt )
                this.psfphotids[filt] = [];
                this.aperphotids[filt] = [];
                let psf_x = [];
                let psf_y = [];
                let psf_dy = [];
                let aper_x = [];
                let aper_y = [];
                let aper_dy = [];
                for ( let i in this.data.mjd ) {
                    if ( ! this.data.psfflags[i] ) {
                        psf_x.push( this.data.mjd[i] );
                        psf_y.push( this.data.flux_psf[i] );
                        psf_dy.push( this.data.flux_psf_err[i] );
                        this.psfphotids[filt].push( this.data.diaforcdedphotid );
                    }
                    if ( ! this.data.aperflags_0[i] ) {
                        aper_x.push( this.data.mjd[i] );
                        aper_y.push( this.data.flux_aper_0[i] );
                        aper_dy.push( this.data.flux_aper_0_err[i] );
                        this.aperphotids[filt].push( this.data.diaforcedphotid );
                    }
                }
                this.psfsets[filt] = new SVGPlot.Dataset( { 'x': psf_x, 'y': psf_y, 'dy': psf_dy } );
                this.apersets[filt] = new SVGPlot.Dataset( { 'x': aper_x, 'y': aper_y, 'dy': aper_dy } );
                this.svgplot.addDataset( this.psfsets[filt] );
            }
        }

        this.svgplot.redraw();
    }

    select_aper_vs_psf()
    {
        let which = this.apervspsf.value;
        if ( which == this.phottype ) return;

        if ( this.phottype == "psf" ) {
            for ( let filt in this.psfsets ) {
                this.svgplot.removeDataset( this.psfsets[filt] );
            }
        } else {
            for ( let filt in this.apersets ) {
                this.svgplot.removeDataset( this.apersets[filt] );
            }
        }

        this.phottype == which;
        if ( this.phottype == "psf" ) {
            for ( let filt in this.psfsets ) {
                this.svgplot.addDataset( this.psfsets[filt] );
            }
        } else {
            for ( let filt in this.apersets ) {
                this.svgplot.addDataset( this.apersets[filt] );
            }
        }

        this.svgplot.redraw();
    }

}
