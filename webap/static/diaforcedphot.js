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
        // For convenience
        this.datadex = {};
        for ( let i = 0 ; i < data.diaforcedphotid.length ; ++i ) {
            this.datadex[ data.diaforcedphotid[i] ] = i;
        }

        let defaultparams = {
            "clipsize": 25,
            "oversamp": 5,
            "zscale_new_contrast": 0.1,
            "zscale_sub_contrast": 0.1
        }
        Object.assign( this, defaultparams );
        Object.assign( this, params );

        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox", "grow1" ] } );
        this.clipcache = {};
        this.clippromise = {};
        this.clips_clipsize = null;
        this.clips_photid = null;
        this.imviewcache = {};
        this.imagepromise = {};
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
                                                                          "flexfitcontent",
                                                                          "vbox", "grow0", "scrolly" ] } );
        this.imagediv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "margin", "padex",
                                                                        "flexfitcontent", "grow0", "scrolly" ] } );

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
                let psf_pointdata = [];
                let aper_pointdata = [];
                for ( let i in this.data.mjd ) {
                    if ( ! this.data.psfflags[i] ) {
                        psf_x.push( this.data.mjd[i] );
                        psf_y.push( this.data.flux_psf[i] );
                        psf_dy.push( this.data.flux_psf_err[i] );
                        this.psfphotids[filt].push( this.data.diaforcedphotid[i] );
                        this.psfphotdexen[filt].push( i );
                        psf_pointdata.push( { 'id': this.data.diaforcedphotid[i], 'dex': i } );
                    }
                    if ( ! this.data.aperflags_0[i] ) {
                        aper_x.push( this.data.mjd[i] );
                        aper_y.push( this.data.flux_aper_0[i] );
                        aper_dy.push( this.data.flux_aper_0_err[i] );
                        this.aperphotids[filt].push( this.data.diaforcedphotid[i] );
                        this.aperphotdexen[filt].push( i );
                        aper_pointdata.push( { 'id': this.data.diaforcedphotid[i], 'dex': i } );
                    }
                }
                this.psfsets[filt] = new SVGPlot.Dataset( { 'x': psf_x, 'y': psf_y, 'dy': psf_dy,
                                                            'pointdata': psf_pointdata, linewid: 0 } );
                this.apersets[filt] = new SVGPlot.Dataset( { 'x': aper_x, 'y': aper_y, 'dy': aper_dy,
                                                             'pointdata': aper_pointdata, linewid: 0 } );
                this.svgplot.addDataset( this.psfsets[filt] );
            }
        }

        this.svgplot.redraw();

        // ****************************************
        // cutouts div

        this.clipboxes = {};
        this.clipcanvases = {};
        let dispsize = Math.round( this.clipsize * this.oversamp );
        let tab = rkWebUtil.elemaker( "table", this.cutoutsdiv );
        for ( let which of [ "new", "ref", "sub" ] ) {
            let tr = rkWebUtil.elemaker( "tr", tab, { "classes": [ "vcenter" ] } );
            rkWebUtil.elemaker( "th", tr, { "text": which } );
            let td = rkWebUtil.elemaker( "td", tr );
            this.clipboxes[which] = rkWebUtil.elemaker( "div", td, { "classes": [ "mostlyborder", "padex" ] } );
            this.clipboxes[which].style.poition = "relative";
            this.clipboxes[which].style.width = dispsize.toString() + "px";
            this.clipboxes[which].style.height = dispsize.toString() + "px";
            this.clipcanvases[which] = rkWebUtil.elemaker( "canvas", this.clipboxes[which],
                                                           { "attributes": { "width": dispsize,
                                                                             "height": dispsize
                                                                           },
                                                             "classes": [ "clipcanvas" ],
                                                             "click": (e) => { self.click_on_thumbnail(which) }
                                                           } );
            this.clipcanvases[which].style.width = dispsize.toString() + "px";
            this.clipcanvases[which].style.height = dispsize.toString() + "px";
        }

        tab = rkWebUtil.elemaker( "table", this.cutoutsdiv, { "classes": [ "borderless", "margintopem",
                                                                           "font75" ] } );
        this.cliptabcells = {};
        for ( let prop of [ "mjd", "filter", "flux_psf", "flux_aper_0", "flux_aper_n", "psfflags", "x", "y" ] ) {
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
        let photid = info.pointdata.id;
        let i = info.pointdata.dex;

        let clipsdone = false;
        if ( this.clipcache.hasOwnProperty( this.clipsize ) ) {
            if ( this.clipcache[ this.clipsize ].hasOwnProperty( photid ) ) {
                this.show_thumbnails( this.clipcache[ this.clipsize ][ photid ] );
                clipsdone = true;
            }
        }
        if ( ! clipsdone ) {
            let promise = null;
            if ( this.clippromise.hasOwnProperty( this.clipsize ) ) {
                if ( this.clippromise[ this.clipsize ].hasOwnProperty( photid ) ) {
                    promise = this.clippromise[ this.clipsize ][ photid ];
                }
            }
            if ( promise === null ) {
                promise = this.context.connector.httpPromise( "diaforcedphotclips/" + photid + "/" + this.clipsize,
                                                              null, false );
                if ( ! this.clippromise.hasOwnProperty( this.clipsize ) ) {
                    this.clippromise[ this.clipsize ] = {};
                }
                this.clippromise[ this.clipsize ][ photid ] = promise;
            }
            promise.then( (data) => self.receive_thumbnails(data, photid, self.clipsize) );
        }

        let tmp = { "digits": 4, "fixedprecision":2, "expprecision": 3, "nullok": true };

        this.cliptabcells["mjd"].innerHTML = this.data.mjd[i].toFixed( 5 );
        this.cliptabcells["filter"].innerHTML = this.data.filter[i];
        this.cliptabcells["flux_psf"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_psf[i], tmp )
                                                    + " ± " +
                                                    rkWebUtil.floatToString( this.data.flux_psf_err[i], tmp )
                                                  );
        this.cliptabcells["flux_aper_0"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_aper_0[i], tmp )
                                                       + " ± " +
                                                       rkWebUtil.floatToString( this.data.flux_aper_0_err[i], tmp )
                                                     );
        this.cliptabcells["flux_aper_n"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_aper_n[i], tmp )
                                                       + " ± " +
                                                       rkWebUtil.floatToString( this.data.flux_aper_n_err[i], tmp )
                                                     );
        this.cliptabcells["psfflags"].innerHTML = "0x" + this.data.psfflags[i].toString(16);
        this.cliptabcells["x"].innerHTML = this.data.x[i].toFixed( 2 );
        this.cliptabcells["y"].innerHTML = this.data.y[i].toFixed( 2 );
    }

    receive_thumbnails( data, photid, clipsize )
    {
        if ( ! ImView.is_little_endian() ) {
            window.alert( "This webap doesn't work on a platform that's not little-endian." );
            return;
        }
        let dv = new DataView( data );
        if ( clipsize != dv.getUint16( 0, true ) ) {
            window.alert( "Expected clipsize " + clipsize + " doesn't match what we got back: " +
                          dv.getUint16( 0, true ) );
            return;
        }
        let clipsq = clipsize * clipsize;
        let newdata = new Float32Array( dv.buffer, 4, clipsq );
        let refdata = new Float32Array( dv.buffer, 4 + 4 * clipsq, clipsq );
        let subdata = new Float32Array( dv.buffer, 4 + 8 * clipsq, clipsq );
        let tmp = ImView.zsc_limits( newdata, this.zscale_new_contrast );
        let newmin = tmp[0];
        let newmax = tmp[1];
        tmp = ImView.zsc_limits( subdata, this.zscale_sub_contrast );
        let submin = tmp[0];
        let submax = tmp[1];

        if ( ! this.clipcache.hasOwnProperty( clipsize ) ) {
            this.clipcache[ clipsize ] = {};
        }

        this.clipcache[ clipsize ][ photid ] = {
            'photid': photid,
            'clipsize': clipsize,
            'new': newdata,
            'ref': refdata,
            'sub': subdata,
            'newmin': newmin,
            'newmax': newmax,
            'submin': submin,
            'submax': submax
        };

        this.show_thumbnails( this.clipcache[ clipsize ][ photid ] );
    }

    show_thumbnails( clips )
    {
        this.clips_photid = clips['photid'];
        this.clips_clipsize = clips['clipsize'];
        let newdata = clips['new'];
        let refdata = clips['ref'];
        let subdata = clips['sub'];
        let clipsize = clips['clipsize'];

        // THOUGHT REQUIRED.  If clipsize has changed since the canvases were
        //   made, this will be broken.
        let dispsize = Math.round( clipsize * this.oversamp );
        let newctx = this.clipcanvases['new'].getContext( "2d", { 'alpha': false } );
        let refctx = this.clipcanvases['ref'].getContext( "2d", { 'alpha': false } );
        let subctx = this.clipcanvases['sub'].getContext( "2d", { 'alpha': false } );
        // I *think* this initialized all the data to 0 (based on Uint8ClampedArray docs)
        let newdisp = newctx.createImageData( dispsize, dispsize );
        let refdisp = refctx.createImageData( dispsize, dispsize );
        let subdisp = subctx.createImageData( dispsize, dispsize );
        // Remember that astronomical images think the lower-left is (0, 0),
        //    whereas browsers think the upper-left is (0, 0)
        for ( let y = 0 ; y < clipsize ; ++y ) {
            let starty = Math.round( (clipsize - y - 1) * this.oversamp );
            let nexty = Math.round( (clipsize - y) * this.oversamp );
            if ( nexty <= starty ) continue;
            if ( nexty > dispsize ) nexty = dispsize;
            for ( let x = 0 ; x < clipsize ; ++x ) {
                let startx = Math.round( x * this.oversamp );
                let nextx = Math.round( (x+1) * this.oversamp );
                if ( nextx <= startx ) continue;
                if ( nextx > dispsize ) nextx = dispsize;

                let dex = y * clipsize + x;
                let newbval = ImView.float_to_byte( newdata[dex], clips.newmin, clips.newmax );
                let refbval = ImView.float_to_byte( refdata[dex], clips.newmin, clips.newmax );
                let subbval = ImView.float_to_byte( subdata[dex], clips.submin, clips.submax );

                for ( let dispy = starty ; dispy < nexty ; ++dispy ) {
                    for ( let dispx = startx ; dispx < nextx ; ++dispx ) {
                        let off = 4 * ( ( dispy * dispsize ) + dispx );
                        newdisp.data[ off     ] = newbval;
                        newdisp.data[ off + 1 ] = newbval;
                        newdisp.data[ off + 2 ] = newbval;
                        newdisp.data[ off + 3 ] = 255;  // alpha
                        refdisp.data[ off     ] = refbval;
                        refdisp.data[ off + 1 ] = refbval;
                        refdisp.data[ off + 2 ] = refbval;
                        refdisp.data[ off + 3 ] = 255;
                        subdisp.data[ off     ] = subbval;
                        subdisp.data[ off + 1 ] = subbval;
                        subdisp.data[ off + 2 ] = subbval;
                        subdisp.data[ off + 3 ] = 255;
                    }
                }
            }
        }
        newctx.putImageData( newdisp, 0, 0 );
        refctx.putImageData( refdisp, 0, 0 );
        subctx.putImageData( subdisp, 0, 0 );
    }

    click_on_thumbnail( which )
    {
        if ( ( this.clips_clipsize === null ) || ( this.clips_photid === null ) ) return;

        let self = this;

        rkWebUtil.wipeDiv( this.imagediv );
        let p = rkWebUtil.elemaker( "p", this.imagediv, { "classes": [ "warning", "bold", "italic" ],
                                                          "text": "Loading image data..." } );

        let dex = this.datadex[ this.clips_photid ];
        let imgid = null;
        if ( which == "new" ) {
            imgid = this.data.newid[ dex ];
        } else if ( which == "ref" ) {
            imgid = this.data.refid[ dex ];
        } else {
            imgid = this.data.subid[ dex ];
        }
        if ( imgid === null ) {
            rkWebUtil.wipeDiv( "p" );
            rkWebUtil.elemaker( "text", p, { "classes": [ "error", "bold", "italic" ],
                                             "text": "Requested image data doesn't exist." } );
            return;
        }

        if ( this.imviewcache.hasOwnProperty( imgid ) ) {
            rkWebUtil.wipeDiv( this.imagediv );
            this.imagediv.appendChild( this.imviewcache[ imgid ].div );
        }
        else {
            let promise = null;
            if ( this.imagepromise.hasOwnProperty( imgid ) ) promise = this.imagepromise[ imgid ];
            if ( promise === null ) {
                promise = this.context.connector.httpPromise( "image_data/" + imgid, null, false );
                this.imagepromise[ imgid ] = promise;
            }
            promise.then( (data) => { self.create_imview( imgid, data ) } );
        }
    }

    create_imview( imgid, data ) {
        let self = this;

        rkWebUtil.wipeDiv( this.imagediv );

        let dv = new DataView( data );
        let height = dv.getUint16( 0, true );
        let width = dv.getUint16( 2, true );
        if ( ! ImView.is_little_endian() ) {
            window.alert( "This webap doesn't work on a platform that's not little-endian." );
            return;
        }
        let imagedata = new Float32Array( dv.buffer, 4, ( ( dv.byteLength - 4 ) / 4 ) );
        let imview = new ImView( { "data": imagedata,
                                   "width": width,
                                   "height": height,
                                   "parent": this.imagediv,
                                   "dispwidth": 600,    // ... make this configurable?
                                   "dispheight": 600
                                 } );
        this.imviewcache[ imgid ] = imview;
    }

}
