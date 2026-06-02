import pytest

from models.provenance import Provenance
from pipeline.scoring import Scorer


def test_rbbot( decam_datastore_through_measurements ):
    ds = decam_datastore_through_measurements
    scorer = Scorer( algorithm='RBbot-quiet-shadow-131-cut0.55' )
    # Need to update the DataStore's provenance tree because
    #   it was created with a different scorer algorithm
    scoreprov = Provenance( process='scoring',
                            parameters=scorer.pars.get_critical_pars(),
                            upstreams=[ ds.prov_tree['measuring'] ]
                           )
    ds.prov_tree['scoring'] = scoreprov

    expected_scores = [ 0.433, 0.388, 0.548, 0.447, 0.792, 0.618, 0.593, 0.487, 0.593, 0.449 ]
    scorer.run( ds )
    for scobj, expect in zip( ds.deepscores, expected_scores ):
        assert scobj.score == pytest.approx( expect, abs=0.002 )
