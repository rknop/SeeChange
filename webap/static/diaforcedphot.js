import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"
import { SVGPlot } from "./svgplot.js"
import { ImView } from "./imview.js"

// **********************************************************************

seechange.DiaForcedPhotSearch = class
{
    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox", "grow1" ],
                                                      'id': 'diaforcedphotsearchdiv' } );
        this.ltcv_divs = {};
    }

    render()
    {
        let self = this;
        let hbox, box, vbox, p;

        hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "padex", "xmargin", "mostlyborder",
                                                                   "nogrow" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "padex", "xmargin", "mostlyborder" ] } );
        rkWebUtil.elemaker( "p", vbox, { "text": "Show lightcurve for object:" } );
        p = rkWebUtil.elemaker( "p", vbox )
        this.ltcvid = rkWebUtil.elemaker( "input", p, { "id": "diaforcedphot_ltcvid",
                                                        "attributes": { "type": "text", "size": 36} } );
        p = rkWebUtil.elemaker( "p", vbox )
        rkWebUtil.button( p, "Show Object", () => { self.show_lightcurve_for_object(); } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "padex", "xmargin", "mostlyborder" ] } );
        rkWebUtil.elemaker( "p", vbox, { "text": "(Searching not implemented yet)" } );

        this.ltcv_parent_div = rkWebUtil.elemaker( "div", this.div, { "classes": [ "vbox", "grow1" ] } );
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
    constructor( context, data, params )
    {
        this.context = context;
        this.data = data;

        let defaultparams = {
            "clipsize": 25,
            "oversamp": 5,
            "zscale_new_contrast": 0.1,
            "zscale_sub_contrast": 0.1
        }
        Object.assign( this, defaultparams );
        Object.assign( this, params );

        
        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox", "grow1" ] } );
    }

    render()
    {
        let self = this;
        
        rkWebUtil.elemaker( "h2", this.div, { "text": "DIA Forced Photometry for " + this.data.objname
                                              + " ( " + this.data.objid + " )" } );

        let hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "flexeven", "grow1" ] } );
        this.ltcvdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                       "flexfitcontent", "scrolly",
                                                                       "vbox", "grow1" ] } );
        this.cutoutsdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                          "flexfitcontent", "minwid40p",
                                                                          "vbox", "grow0", "scrolly" ] } );

        // ****************************************
        // ltcvdiv
        
        this.svgplot = new SVGPlot.Plot( { "name": "ltcv-" + this.data.objid,
                                           "divid": "ltcv-svgplot-" + this.data.objid,
                                           "showerrbar": true,
                                           "manuallimits": true,
                                           "xtitle": "MJD",
                                           "ytitle": "Flux (nJy)"
                                         } );
        this.svgplot.addClickListener( (info) => { self.click_on_svgplot(info) } );
        this.ltcvdiv.appendChild( this.svgplot.topdiv );

        this.apervspsf = rkWebUtil.elemaker( "select", this.svgplot.buttonbox,
                                             { "change": (e) => { self.select_aper_vs_psf(); } } );
        rkWebUtil.elemaker( "option", this.apervspsf, { "text": "PSF photometry",
                                                        "attributes": { "value": "psf", "selected": 1 } } );
        rkWebUtil.elemaker( "option", this.apervspsf, { "text": "Aperture Photometry",
                                                        "attributes": { "value": "aperture" } } );

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
        this.aperphotids = {};
        this.psfphotdexen = {};
        this.aperphotdexen = {};
        this.psfsets = {};
        this.apersets = {};
        this.phottype = "psf";
        for ( let filt of filterorder ) {
            if ( filters.has( filt ) ) {
                this.filters.push( filt )
                this.psfphotids[filt] = [];
                this.aperphotids[filt] = [];
                this.psfphotdexen[filt] = [];
                this.aperphotdexen[filt] = [];
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
                        this.psfphotids[filt].push( this.data.diaforcedphotid[i] );
                        this.psfphotdexen[filt].push( i );
                    }
                    if ( ! this.data.aperflags_0[i] ) {
                        aper_x.push( this.data.mjd[i] );
                        aper_y.push( this.data.flux_aper_0[i] );
                        aper_dy.push( this.data.flux_aper_0_err[i] );
                        this.aperphotids[filt].push( this.data.diaforcedphotid[i] );
                        this.aperphotdexen[filt].push( i );
                    }
                }
                this.psfsets[filt] = new SVGPlot.Dataset( { 'x': psf_x, 'y': psf_y, 'dy': psf_dy, linewid: 0 } );
                this.apersets[filt] = new SVGPlot.Dataset( { 'x': aper_x, 'y': aper_y, 'dy': aper_dy, linewid: 0 } );
                this.svgplot.addDataset( this.psfsets[filt] );
            }
        }

        this.svgplot.redraw();

        // ****************************************
        // cutouts div

        this.clipboxes = {};
        this.clipcanvases = {};
        this.clipctx = {};
        let tab = rkWebUtil.elemaker( "table", this.cutoutsdiv );
        for ( let which of [ "new", "ref", "sub" ] ) {
            let tr = rkWebUtil.elemaker( "tr", tab, { "classes": [ "vcenter" ] } );
            rkWebUtil.elemaker( "th", tr, { "text": which } );
            let td = rkWebUtil.elemaker( "td", tr );
            this.clipboxes[which] = rkWebUtil.elemaker( "div", td,
                                                        { "classes": [ "mostlyborder", "padex",
                                                                       "nogrow", "noshrink" ] } );
            this.clipboxes[which].style.width = ( this.clipsize * this.oversamp ).toString() + "px";
            this.clipboxes[which].style.height = ( this.clipsize * this.oversamp ).toString() + "px";
            this.clipcanvases[which] = rkWebUtil.elemaker( "canvas", this.clipboxes[which],
                                                           { "attributes": { "width": this.clipsize * this.oversamp,
                                                                             "height": this.clipsize * this.oversamp
                                                                           },
                                                             "classes": [ "imview_canvas" ] } );
            this.clipcanvases[which].style.width = ( this.clipsize * this.oversamp ).toString() + "px";
            this.clipcanvases[which].style.height = ( this.clipsize * this.oversamp ).toString() + "px";
            this.clipctx[which] = this.clipcanvases[which].getContext( "2d" );
        }

        tab = rkWebUtil.elemaker( "table", this.cutoutsdiv, { "classes": [ "borderless", "margintopem",
                                                                           "font75" ] } );
        this.cliptabcells = {};
        for ( let prop of [ "mjd", "filter", "flux_psf", "flux_aper_0", "psfflags", "x", "y" ] ) {
            let tr = rkWebUtil.elemaker( "tr", tab );
            rkWebUtil.elemaker( "th", tr, { "classes": [ "right" ], "text": prop + ": " } );
            this.cliptabcells[prop] = rkWebUtil.elemaker( "td", tr );
        }
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

        this.phottype = which;
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

    click_on_svgplot( info )
    {
        let self = this;
        let i =( ( this.phottype == 'psf' )
                 ? this.psfphotdexen[ this.filters[ info.setdex ] ][ info.pointdex ]
                 : this.aperphotdexen[ this.filters[ info.setdex ] ][ info.pointdex ] );
        let photid = this.data.diaforcedphotid[i];
        let promise = this.context.connector.httpPromise( "diaforcedphotclips/" + photid, null, false );
        promise.then( (data) => self.show_thumbnails(data) );

        let tmp = { "fixedprecision":2, "nullok": true };
        
        this.cliptabcells["mjd"].innerHTML = this.data.mjd[i].toFixed( 5 );
        this.cliptabcells["filter"].innerHTML = this.data.filter[i];
        this.cliptabcells["flux_psf"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_psf[i], 4, tmp )
                                                    + " ± " +
                                                    rkWebUtil.floatToString( this.data.flux_psf_err[i], 4, tmp )
                                                  );
        this.cliptabcells["flux_aper_0"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_aper_0[i], 4, tmp )
                                                       + " ± " +
                                                       rkWebUtil.floatToString( this.data.flux_aper_0_err[i], 4, tmp )
                                                     );
        // flux_aper_n will be identical to flux_aper_0 by construction
        // this.cliptabcells["flux_aper_n"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_aper_n[i], 4, tmp )
        //                                                + " ± " +
        //                                                rkWebUtil.floatToString( this.data.flux_aper_n_err[i], 4, tmp )
        //                                              );
        this.cliptabcells["psfflags"].innerHTML = "0x" + this.data.psfflags[i].toString(16);
        this.cliptabcells["x"].innerHTML = this.data.x[i].toFixed( 2 );
        this.cliptabcells["y"].innerHTML =this.data.y[i].toFixed( 2 );
    }
    
    float_to_byte( f, min, max )
    {
        if ( isNan(f) ) return 0;
        else if ( f <= min ) return 0;
        else if ( f >= max ) return 255;
        else return Math.round( ( f - min ) * 255 / ( max - min ) )
    }

    
    show_thumbnails( data )
    {
        let dv = new DataView( data );
        this.clipsize = dv.getUint16( 0, true );
        let clipsdq = this.clipsize * this.clipsize;
        this.newdata = new Float32Array( clipsq );
        this.refdata = new Float32Array( clipsq );
        this.subdata = new Float32Array( clipsq );
        // Is there not a faster way to do this?
        for ( let i = 0 ; i < clipsq ; ++i ) {
            this.newdata[i] = dv.getFloat32( 2 + 4 + i, true );
            this.refdata[i] = dv.getFloat32( 2 + 4 * clipsq + 4 * i, true );
            this.subdata[i] = dv.getFloat32( 2 + 8 * clipsq + 4 * i, true );
        }
        let tmp = ImView.zscale_limits( this.newdata, this.zscale_new_contrast );
        this.newmin = tmp[0];
        this.newmax = tmp[1];
        tmp = ImView.zscale_limits( this.subdata, this.zscale_sub_contrast );
        this.submin = tmp[0];
        this.submax = tmp[1];

        let newdisp = this.clipctx['new'].createImageData( this.ciposize, this.clipsize );
        let refdisp = this.clipctx['ref'].createImageData( this.ciposize, this.clipsize );
        let subdisp = this.clipctx['sub'].createImageData( this.ciposize, this.clipsize );
        let off = 0;
        let bval = 0;
        for ( let y = 0 ; y < this.clipsize ; ++y ) {
            for ( let x = 0; x < this.clipsize ; ++x ) {
                let dex = y * this.clipsize + x;
                bval = this.float_to_byte( this.newdata[dex], this.newmin, this.newmax );
                newdisp[ off ] = bval;
                newdisp[ off + 1 ] = bval;
                newdisp[ off + 2 ] = bval;
                newdisp[ off + 3 ] = 255;  // alpha
                bval = this.float_to_byte( this.refdata[dex], this.newmin, this.newax );
                refdisp[ off ] = bval;
                refdisp[ off + 1 ] = bval;
                refdisp[ off + 2 ] = bval;
                refdisp[ off + 3 ] = 255;
                bval = this.float_to_byte( this.subdata[dex], this.submin, this.submax );
                subdisp[ off ] = bval;
                subdisp[ off + 1 ] = bval;
                subdisp[ off + 2 ] = bval;
                subdisp[ off + 3 ] = 255;
                off += 4;
            }
        }
        this.clipctx['new'].putImageData( newdisp, 0, 0 );
        this.clipctx['ref'].putImageData( refdisp, 0, 0 );
        this.clipctx['sub'].putImageData( subdisp, 0, 0 );
    }
        
    
}
