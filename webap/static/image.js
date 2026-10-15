import { rkWebUtil } from "./rkwebutil.js";
import { ImView } from "./imview.js";
import { seechange } from "./seechange_ns.js";

// **********************************************************************

seechange.Image = class extends( rkWebUtil.listener )
{
    constructor( context, info, params={} )
    {
        super();
        let default_params = { 'nomeas_cutouts': false,
                               'provtag': null,
                               'box_width': 10,
                               'oversample': 5,
                               'start_image': null,
                               'show_detections': true
                             };
        Object.assign( this, default_params );
        Object.assign( this, params );

        this._addsignal( 'image' );
        this._addsignal( 'detections' );

        let self = this;
        let p, fncomps;

        this.context = context;
        this.info = info;
        this.issub = this.info.is_sub;
        fncomps = this.info.filepath.match( /\/?([^\/]+)$/ );
        this.filename = (fncomps === null) ? "????" : fncomps[1];

        this._imagediv = null;
        this.imview_div = null;
        this.info_div = null;
        this.thumbnails_div = null;
        this.infopromises = {};
        this.imview_promises = {};
        this.imview_promise = null;
        
        this.detections_wrapper = rkWebUtil.elemaker( 'div', null );
        this.detections_shown = 'neither';
        this.detectionsdiv_sans = null;
        this.detectionsdiv_avec = null;
        this.cutouts_promise = null;
    }

    get imagediv()
    {
        return this.get_imagediv();
    }

    get_imagediv() {
        if ( this._imagediv === null ) this.make_image_div();
        return this._imagediv;
    }
    
    show_imagediv()
    {
        let div = this.get_imagediv();
        this.emit_signal( 'image', { 'imagediv': div } );
        return div;
    }

    get detectionsdiv()
    {
        if ( this.detections_shown == 'avec' ) {
            return get_detectionsdiv( false );
        }
        else {
            return get_detectionsdiv( true );
        }
    }

    get_detectionsdiv( sans_measurements=true )
    {
        if ( ! this.issub ) return null;

        if ( sans_measurements ) {
            if ( this.detectionsdiv_sans === null ) this.make_detections_div( true, false );
            if ( this.detections_shown != 'sans' ) {
                rkWebUtil.wipeDiv( this.detections_wrapper );
                this.detections_wrapper.appendChild( this.detectionsdiv_sans );
                this.detections_shown = 'sans';
            }
        }
        else {
            if ( this.detectionsdiv_avec === null ) this.make_detections_div( false, false );
            if ( this.detections_shown != 'avec' ) {
                rkWebUtil.wipeDiv( this.detections_warpper );
                this.detections_wrapper.appendChild( this.detectionsdiv_avec );
                this.detections_shown = 'avec';
            }
        }
        return this.detections_wrapper;
    }
        
    show_detectionsdiv( sans_measurements=true )
    {
        let div = this.get_detections_div( sans_measurements );
        this.emit_signal( 'detections', { 'detectionsdiv': div } );
        return this.detections_wrapper;
    }

        

    // **********************************************************************
    // Image div

    make_image_div( remake=false )
    {
        let self = this;
        
        if ( ( this._imagediv !== null ) && ( ! remake ) ) return;

        this._imagediv = rkWebUtil.elemaker( "div", null );

        rkWebUtil.elemaker( "h2", this._imagediv,
                            { "text": "Image: " + this.filename + "( " + this.info._id + " )" } );
        
        let hbox = rkWebUtil.elemaker( "div", this._imagediv, { "classes": [ "hbox" ] } );

        this.imview_div = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "marginrightem" ] } );
        this.info_div = rkWebUtil.elemaker( "div", hbox, { "classes": [ "marginrightem" ] } );
        this.thumbnails_div = rkWebUtil.elemaker( "div", hbox );
        
        this.shown_imview = null;
        this.shown_info = null;
        if ( this.issub ) {
            this.infopromises = { 'new': null, 'ref': null };
            this.imviews = { 'new': null, 'ref': null, 'sub': null };
            this.imview_promises = { 'new': null, 'ref': null, 'sub': null };
            this.select_imview( 'sub' );
            this.thumbnails_promise = null;
            this.squares_rendered = { 'new': false, 'ref': false, 'sub': false };
            this.zoomx0 = null;
            this.zoomy0 = null;
            this.zoomscale = null;

            let p = rkWebUtil.elemaker( "p", this.imview_div );
            this.newrefsub = rkWebUtil.elemaker( "select", p,
                                                 { "change": (e) => {
                                                     let which = self.newrefsub.value;
                                                     self.load_info_view( which );
                                                     self.select_imview( which );
                                                 } } );
            let n = rkWebUtil.elemaker( "option", this.newrefsub, { "text": "New", attributes: { "value": "new" } } );
            let r = rkWebUtil.elemaker( "option", this.newrefsub, { "text": "Ref", attributes: { "value": "ref" } } );
            let s = rkWebUtil.elemaker( "option", this.newrefsub, { "text": "Sub", attributes: { "value": "sub" } } );
            let start_image = this.start_image;

            // Make a new div underneath the dropdown we just made so the dropdown won't be wiped out
            this.imview_div = rkWebUtil.elemaker( "div", this.imview_div );

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
                let url = "image_info/";
                if ( which == 'new' ) {
                    url += this.info['new_id'];
                }
                else if ( which == 'ref' ) {
                    url += this.info['warped_ref_id'];
                }
                else {
                    window.alert( "Invalid thing to load_info_view: " + which );
                    return;
                }
                this.infopromises[which] = this.context.connector.httpPromise( url );
            }
            rkWebUtil.wipeDiv( this.info_div );
            rkWebUtil.elemaker( "p", this.info_div, { "text": "Loading image info...",
                                                      "classes": [ "warning", "bold", "italic" ] } );
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
                           'type': seechange.imagetypes[ info['_type'] ],
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
            this.imview_div.appendChild( imview.div );
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
                    this.imview_promise = this.context.connector.httpPromise( "image_data/" + imgtorequest,
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
            if ( ( this.provtag !== null ) && this.show_detections ) {
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
        for ( let im of [ 'new', 'ref', 'sub' ] ) {
            if ( im != which ) {
                if ( this.imviews[im] != null ) {
                    this.imviews[im].x0 = this.zoomx0;
                    this.imviews[im].y0 = this.zoomy0;
                    this.imviews[im].scale = this.zoomscale;
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
            let url = ( "png_cutouts_for_sub_image/" + this.info._id + "/" + this.provtag +
                        "/1/" + (this.nomeas_cutouts ? "1" : "0") );
            this.cutouts_promise = this.context.connector.httpPromise( url );
        }
        if ( ( which != null ) && ( ! this.squares_rendered[which] ) ) {
            this.cutouts_promise.then( (data) => { self.show_detections_on_image(which, data); } );
            // Setting it here and not in the promise result so the callback won't be
            //   executed multiple times
            self.squares_rendered[which] = true;
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
            imview.addsquare( data.cutouts.x[dex], data.cutouts.y[dex], this.box_width,
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

    // **********************************************************************
    // Detections div

    make_detections_div( sans=true, remake=false )
    {
        if ( ! this.is_sub ) {
            throw Error( "Can't make detections div when an image isn't a subtraction" );
        }

        rkWebUtil.wipeDiv( this.detections_wrapper );
        rkWebUtil.elemaker( 'h3', this.detections_wrapper, { "text": "Loading detections...",
                                                             "classes": [ "warning", "bold", "italic" ] } );
        
        let self = this;

        if ( sans ) {
            if ( ( this.detectionsdiv_sans !== null ) && ( ! remake ) ) return;
            if ( this.detections_sans_promise !== null ) {
                let url = "png_cutouts_for_sub_image/" + this.info._id + "/" + this.provtag + "/1/0";
                this.detections_sans_promise = this.context.connector.httpPromise( url );
            }
            this.detections_sans_promise.then( (data) => { self._actually_make_detections_div( true, data ); } );
        } else {
            if ( ( this.detectionsdiv_avec !== null ) && ( ! remake ) ) return;
            if ( this.detections_avec_promise !== null ) {
                let url = "png_cutoust_for_sub_image/" + this.info._id + "/" + this.provtag + "/1/1";
                this.detections_avec_promise = this.context.connector.httpPromise( url );
            }
            this.detections_avec_promise.then( (data) => { self.actually_make_detections_div( false, data ); } );
        }
    }

    actually_make_detections_div( sans_measurements, data )
    {
        let div, table, tr, th, td, img, span, ttspan;

        rkWebUtil.wipeDiv( this.detections_wrapper );
        if ( sans_measurements ) {
            this.detectionsdiv_sans = rkWebUtil( "div", this.detections_wrapper );
            div = this.detectionsdiv_sans;
        } else {
            this.detectionsdiv_avec = rkWebUtil( "div", this.detections_wrapper );
            div = this.detectionsdiv_avec;
        }

        table = rkWebUtil.elemaker( "table", div, { "id": "detectionstable" } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "new" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "ref" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "sub" } );

        // Sorting is now done server-side... TODO, think about this
        // // TODO : sort by r/b, make sort configurable
        // let dexen = [...Array(data.cutouts.sub_id.length).keys()];
        // dexen.sort( (a, b) => {
        //     if ( ( data.cutouts['flux'][a] == null ) && ( data.cutouts['flux'][b] == null ) ) return 0;
        //     else if ( data.cutouts['flux'][a] == null ) return 1;
        //     else if ( data.cutouts['flux'][b] == null ) return -1;
        //     else if ( data.cutouts['flux'][a] > data.cutouts['flux'][b] ) return -1;
        //     else if ( data.cutouts['flux'][a] < data.cutouts['flux'][b] ) return 1;
        //     else return 0;
        // } );

        // for ( let i of dexen ) {
        for ( let i in data.cutouts.sub_id ) {
            tr = rkWebUtil.elemaker( "tr", table );
            td = rkWebUtil.elemaker( "td", tr );
            if ( data.cutouts.objname[i] != null ) {
                let text = "Object: " + data.cutouts.objname[i];
                if ( data.cutouts.is_test[i] ) text += " [TEST]";
                td.appendChild( document.createTextNode( text ) );
            }
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['new_png'][i],
                                          "width": oversample * data.cutouts['w'][i],
                                          "height": oversample * data.cutouts['h'][i],
                                          "alt": "new" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['ref_png'][i],
                                          "width": oversample * data.cutouts['w'][i],
                                          "height": oversample * data.cutouts['h'][i],
                                          "alt": "ref" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['sub_png'][i],
                                          "width": oversample * data.cutouts['w'][i],
                                          "height": oversample * data.cutouts['h'][i],
                                          "alt": "sub" } } );

            // *** The info td, which is long
            td = rkWebUtil.elemaker( "td", tr );
            // row; chip
            rkWebUtil.elemaker( "b", td, { "text": "chip: " } );
            td.appendChild( document.createTextNode( data.cutouts.section_id[i] ) );
            rkWebUtil.elemaker( "br", td );
            // row: source index, with good/bad
            rkWebUtil.elemaker( "b", td, { "text": "source index: " } )
            td.appendChild( document.createTextNode( data.cutouts.source_index[i] + "  " ) );
            span = rkWebUtil.elemaker( "b", td, { "text": ( data.cutouts.is_bad[i] ?
                                                            "fails cuts" : "passes cuts" ) } );
            span.classList.add( "tooltipcolorlesssource" );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
            ttspan.innerHTML = ( "<p>major_width: " + seechange.nullorfixed( data.cutouts.major_width[i], 2 ) + "<br>" +
                                 "minor_width: " + seechange.nullorfixed( data.cutouts.minor_width[i], 2 ) + "<br>" +
                                 "nbadpix: " + data.cutouts.nbadpix[i] + "<br>" +
                                 "negfrac: " + data.cutouts.negfrac[i] + "<br>" +
                                 "negfluxfrac: " + data.cutouts.negfluxfrac[i] + "<br>" +
                                 "Gauss fit pos: " + seechange.nullorfixed( data.cutouts['gfit_x'][i], 2 )
                                 + " , " + seechange.nullorfixed( data.cutouts['gfit_y'][i], 2 ) +
                                 "</p>" )
            span.classList.add( data.cutouts.is_bad[i] ? "bad" : "good" );
            rkWebUtil.elemaker( "br", td );
            // row: ra/dec
            rkWebUtil.elemaker( "b", td, { "text": "(α, δ): " } );
            td.appendChild( document.createTextNode( "(" +
                                                     seechange.nullorfixed( data.cutouts['measra'][i], 5 )
                                                     + " , " +
                                                     seechange.nullorfixed( data.cutouts['measdec'][i], 5 )
                                                     + ")" ) );
            rkWebUtil.elemaker( "br", td );
            // row: x, y from cutouts; this is where it was originally detected.
            rkWebUtil.elemaker( "b", td, { "text": "det. (x, y): " } );
            td.appendChild( document.createTextNode( "(" +
                                                     seechange.nullorfixed( data.cutouts['cutout_x'][i], 2 )
                                                     + " , " +
                                                     seechange.nullorfixed( data.cutouts['cutout_y'][i], 2 )
                                                     + ")" ) );
            rkWebUtil.elemaker( "br", td );
            // row: x, y from measurement table.
            rkWebUtil.elemaker( "b", td, { "text": "meas. (x, y): " } );
            td.appendChild( document.createTextNode( "(" +
                                                     seechange.nullorfixed( data.cutouts['x'][i], 2 )
                                                     + " , " +
                                                     seechange.nullorfixed( data.cutouts['y'][i], 2 )
                                                     + ")" ) );
            rkWebUtil.elemaker( "br", td );
            // row: flux
            rkWebUtil.elemaker( "b", td, { "text": "Flux: " } );
            td.appendChild( document.createTextNode( seechange.nullorfixed( data.cutouts["flux"][i], 0 )
                                                     + " ± " +
                                                     seechange.nullorfixed( data.cutouts["dflux"][i], 0 )
                                                     + "  " +
                                                     + ( ( ( data.cutouts['aperrad'][i] == null ) ||
                                                           ( data.cutouts['aperrad'][i] <= 0 ) ) ?
                                                         '(psf)' :
                                                         ( "(aper r=" +
                                                           seechange.nullorfixed( data.cutouts['aperrad'][i], 2 )
                                                           + " px)" ) ) ) );
            rkWebUtil.elemaker( "br", td );
            // row: mag
            rkWebUtil.elemaker( "b", td, { "text": "Mag: " } );
            td.appendChild( document.createTextNode( seechange.nullorfixed( data.cutouts['mag'][i], 2 )
                                                     + " ± " +
                                                     seechange.nullorfixed( data.cutouts['dmag'][i], 2 ) ) );
            rkWebUtil.elemaker( "br", td );
            // row; R/B
            span = rkWebUtil.elemaker( "span", td );
            if ( ( data.cutouts['rb'][i] == null ) || ( data.cutouts['rb'][i] < data.cutouts['rbcut'][i] ) )
                span.classList.add( 'bad' );
            else
                span.classList.add( 'good' );
            rkWebUtil.elemaker( "b", span, { "text": "R/B: " } );
            span.appendChild( document.createTextNode( seechange.nullorfixed( data.cutouts['rb'][i], 3 ) ) );

        }
    }
   
}

// **********************************************************************
// Make this into a module

export { }
