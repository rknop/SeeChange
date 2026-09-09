import models.object
from models.base import PGDB
from models.provenance import Provenance
from pipeline.lightcurve import Lightcurve



def test_lightcurve( sim_lightcurve_persistent_sources, sim_lightcurve_reference, sim_lightcurve_news ):
    srcs = sim_lightcurve_persistent_sources
    ref, refds = sim_lightcurve_reference
    newdsen = sim_lightcurve_news

    try:
        # Create an object
        objprov = Provenance( process='positioning' )
        obj = models.object.Object( name='test_lightcurve_object', ra=srcs[2]['ra'], dec=srcs[2]['dec'],
                                    is_bad=False, provenance_id=objprov.id )
        obj.insert()

        # Lightcurve builder
        ltcv = Lightcurve( zp_prov = newdsen[0].zp.provenance_id,
                           crop_image = [150, 150],
                           mjd0 = 60025.,
                           mjd1 = 60060.,
                           instrument='DemoInstrument',
                           object_name='test_lightcurve_object',
                           subtraction={ 'refset': 'sim_lightcurve_reference',
                                         'alignment': { 'min_matched': 6 } }
                          )
        ltcv.run()

        import pdb; pdb.set_trace()
        pass

    finally:
        # with PGDB() as pgdb:
        #     pgdb.execute( "DELETE FROM objects WHERE name='test_lightcurve_object'" )
        #     pgdb.commit()
        pass
