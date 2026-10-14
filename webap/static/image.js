import { rkWebUtil } from "./rkwebutil.js";
import { ImView } from "./imview.js";
import { seechange } from "./seechange_ns.js";

// **********************************************************************

seechange.Image = class
{
    constructor( context, info, inparams={} )
    {
        let self = this;
        let p, fncomps;
        this.params = { 'nomeas_cutouts': false,
                        'provtag': null,
                        'box_width': 10,
                        'start_image': null
                      };
        Object.assign( this.params, inparams );
        
        this.context = context;
        this.info = info;
        this.issub = this.info.is_sub;

        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "hbox" ] } );
        fncomps = this.info.filepath.match( /\/?([^\/]+)$/ );
        this.filename = (fncomps === null) ? "????" : fncomps[1];
        rkWebUtil.elemaker( "h2", this.div, "Image: " + fncomps[1] + "( " + this.info['_id'] + " )" );

        this.imview_div = rkWebUtil.elemaker( "div", this.div, { "classes": [ "vbox", "marginrightem" ] } );
        this.info_div = rkWebUtil.elemaker( "div", this.div, { "classes": [ "marginrightem" ] } );
        this.cutouts_div = rkWebUtil.elemaker( "div", this.div );
        
        this.shown_imview = null;
        this.shown_info = null;
        if ( this.issub ) {
            this.infopromises = { 'new': null, 'ref': null };
            this.imviews = { 'new': null, 'ref': null, 'sub': null };
            this.imview_promises = { 'new': null, 'ref': null, 'sub': null };
            this.select_imview( 'sub' );
            this.imview_div = rkWebUtil.elemaker( "div", this.imview_div );
            this.cutouts_promise = null;
            this.squares_rendered = { 'new': false, 'ref': false, 'sub': false };
            this.zoomx0 = null;
            this.zoomy0 = null;
            this.zoomscale = null;

            p = rkWebUtil.elemaker( "p", this.imview_div );
            this.newrefsub = rkWebUtil.elemaker( "select", p,
                                                 { "attributes": {
                                                     "change": (e) => {
                                                         self.load_info(which);
                                                         self.select_imview(self.newrefsub.value);
                                                     }
                                                 } } );
            let n = rkWebUtil.elemaker( "option", this.newrefsub, { "text": "New", attributes: { "value": "new" } } );
            let r = rkWebUtil.elemaker( "option", this.newrefsub, { "text": "Ref", attributes: { "value": "ref" } } );
            let s = rkWebUtil.elemaker( "option", this.newrefsub, { "text": "Sub", attributes: { "value": "sub" } } );
            let start_image = this.params.start_image;
            if ( start_image === null ) {
                if ( this.issub ) start_image = 'sub';
                else start_image = 'image';
            }
            if ( start_image == 'sub' ) {
                s.setAttribute( 'selected', 1 );
                this.show_info_view( this.info, 'sub' );
            }
            else if ( start_image == 'ref' ) {
                r.setAttribute( 'selected', 1 );
                this.load_info_view( 'ref' );
            }
            else if ( start_image == 'new' ) {
                n.setAttribute( 'selected', 1 );
                this.load_info_view( 'new' );
            }
            else if ( start_image == 'image' ) {
                n.setAttribute( 'selected', 1 );
                this.load_info_view( this.info, 'image' );
            }
            else {
                window.alert( "Unknown start_image " + start_image + ", can't show image info, many things broken." )
                return;
            }
        }
        else {
            this.infoview = this.show_info_view( this.info, 'image' );
            this.imview = null;
            this.imview_promise = null;
            this.select_imview();
        }
    }

    load_info_view( which )
    {
        let self = this;
        
        if ( ( which == 'sub' ) || ( which == 'image' ) ) {
            this.show_info_view( this.info, which );
        }
        else {
            if ( this.infopromises[which] === null ) {
                url = "image_info/";
                if ( which == 'new' ) {
                    url += this.info['new_id'];
                }
                else if ( which == 'ref' ) {
                    url += this.info['wapred_ref_id'];
                }
                else {
                    window.alert( "Invalid thing to load_info_view: " + which );
                    return;
                }
                this.infopromises[which] = this.context.connector.httpPromise( url );
            }
            this.infopromises[which].then( (info) => { self.show_info_view( info, which ); } );
        }
    }
    
    show_info_view( info, which='image' )
    {
        rkWebUtil.wipeDiv( this.info_div );
        if ( which == 'new' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "New Image" } );
        }
        else if ( which == 'ref' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "Warped Ref Image" } );
        }
        else if ( which == 'sub' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "Sub Image" } );
        }
        else if ( which != 'image' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "Something has gone wrong." } );
        }
            
        let propvalues = { 'mjd': seechange.nullorfixed( info['mjd'], 5),
                           'ra': seechange.nullorfixed( info['ra'], 5 ),
                           'dec': seechange.nullorfixed( info['dec'], 5 ),
                           'filter': info['filter'],
                           'exptime': seechange.nullorfixed( info['exp_time'], 1 ),
                           'instrument': info['instrument'],
                           'airmass': seechange.nullorfixed( info['airmass'], 2 ),
                           'seeing (pix)': seechange.nullorfixed( info['fwhm_estimate'], 2 ),
                           'zp': seechange.nullorfixed( info['zero_point_estimate'], 2 ),
                           'm_lim': seechange.nullorfixed( info['lim_mag_estimate'], 1 ),
                           'bkg': seechange.nullorexpo( info['bkg_mean_estimate'], 4 ),
                           'σ_bkg': seechange.nullorexpo( info['bkg_rms_esitmate'], 4 ),
                           'provid': info['provenance_id'],
                           'filename': this.filename,
                           'id': info['_id'],
                         }
        let table = rkWebUtil.elemaker( "table", this.info_div );
        for ( let prop in propvalues ) {
            let tr = rkWebUtil.elemaker( "tr", table );
            let td = rkWebUtil.elemaker( "td", tr, { "text": prop, "classes": [ "bold", "alignright" ] } );
            if ( prop == 'filename' ) {
                td = rkWebUtil.elemaker( "td", tr, { "text": propvalues[prop], "classes": [ "alignleft",
                                                                                            "tooltipsource" ] } );
                rkWebUtil.elemaker( "span", td, { "text": this.info.filepath, "classes": [ "tooltiptext" ] } );
            } else {
                td = rkWebUtil.elemaker( "td", tr, { "text": propvalues[prop], "classes": [ "alignleft" ] } );
            }
        }
    }
    
    select_imview( which=null )
    {
        let self = this;
        let imview = null;
        let imgtorequest = null;
        if ( this.issub ) {
            imview = this.imviews[ which ];
            if ( which == 'new' ) {
                imgtorequest = this.info.new_id;
            }
            else if ( which == 'ref' ) {
                imgtorequest = this.info.warped_ref_id;
            }
            else {
                imgtorequest = this.info._id;
            }
        }
        else {
            which = 'image';
            imview = this.imview;
            imgtorequest = this.info._id;
        }

        if ( this.shown_imview == which ) return;

        rkWebUtil.wipeDiv( this.imview_div );
        if ( imview !== null ) {
            this.imview_div.appendChild( imview );
            this.shown_imview = which;
            return;
        }
        else {
            if ( this.issub ) {
                if ( this.imview_promises[which] == null ) {
                    this.imview_promises[which] = this.context.connector.httpPromise( "image_data/" + imgtorequest,
                                                                                      null, false );
                }
                this.imview_promises[which].then( (data) => { self.create_imview( data, imgtorequest, which ); } );
            }
            else {
                if ( this.imview_promise == null ) {
                    this.imview_promise = this.context.connector.httpPromise( "image_data/" + imgtoreuest,
                                                                              null, false );
                }
                this.imview_promise.then( (data) => { self.create_imview( data, imgtorequest, which ); } );
            }
        }
    }
    
    create_imview( data, imgtorequest, which )
    {
        let self = this;

        let dv = new DataView( data );
        let height = dv.getUint16( 0, true );
        let width = dv.getUint16( 2, true );
        let imagedata = new DataView( data, 4 );
        let imview = new ImView( { "data": imagedata,
                                   "width": width,
                                   "height": height,
                                   "parent": this.imview_div,
                                   "dispwidth": 600,    // ... make this configurable?
                                   "dispheight": 600
                                 },
                                 true
                               );
        if ( this.zoomx0 !== null ) {
            imview.x0 = this.zoomx0;
            imview.y0 = this.zoomy0;
            imview.scale = this.zoomscale;
        }
        this.imview_div.appendChild( imview.div );
        imview.render_image();
        this.shown_imview = which;

        if ( this.issub ) {
            imview.clickcallback = (x, y) => { self.clicked_on_image(x, y); };
            this.imviews[which] = imview;
            let but = rkWebUtil.button( imview.zoom_buttonbox, "Copy Zoom", () => { self.copy_zoom(which); } );
            but.classList.add( "marginleftex" );
            if ( this.params.provtag !== null ) {
                this.load_cutouts( (data) => { self.show_detections_on_image(which, data); }, which );
            }
        } else {
            this.imview = imview;
        }
    }

    copy_zoom( which )
    {
        this.zoomx0 = this.imviews[which].x0;
        this.zoomy0 = this.imviews[which].y0;
        this.zoomscale = this.imviews[which].scale;
        for ( let im in [ 'new', 'ref', 'sub' ] ) {
            if ( im != which ) {
                if ( this.imviews[im] != null ) {
                    this.imviews[im].x0 = this.zoomx0;
                    this.imviews[im].y0 = this.zoomy0;
                    this.imviews[im].scale = this.zoomsca;e
                    this.imviews[im].render_image();
                }
            }
        }
    }

    load_cutouts( callback=null, which=null )
    {
        // Should only ever be called if the image is a sub image
        let self = this;
        if ( this.cutouts_promise === null ) {
            let url = ( "png_cutouts_for_sub_image/" + this.info.id_ + "/" + this.params.provtag +
                        "/1/" + (this.params.nomeas_cutouts ? "1" : "0") );
            this.cutouts_promise = this.context.connector.httpPromise( url );
        }
        if ( ( which != null ) && ( ! this.squares_rendered[which] ) ) {
            this.cutouts_promise.then( (data) => { self.show_detections_on_image(which, data); } );
            // Setting it here and not in the promise result so the callback won't be
            //   executed multiple times
            self.squareds_rendered[which] = true;
        }            
        if ( callback !== null ) {
            this.cutouts_promise.then( callback );
        }
    }
    
    show_detections_on_image( which, data )
    {
        // This should only ever be called for a sub image
        let imview = this.imviews[which];
        for ( let dex in data.cutouts.x ) {
            imview.addsquare( data.cutouts.x[dex], data.cutouts.y[dex], this.params.box_width,
                              "green", data.cutouts.source_index );
        }
        imview.render_image();
    }

    clicked_on_image( which, x, y )
    {
        let self=this;
        this.load_cutouts( (data) => { self.process_clicked_on_image(x, y, data); }, which );
    }
    
    process_clicked_on_image( x, y, data )
    {
        console.log( "Clicked on image at " + x + ", " + y );
    }

}

// **********************************************************************
// Make this into a module

export { }
