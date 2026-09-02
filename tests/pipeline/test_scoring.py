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

    expected_scores = [0.624, 0.880, 0.439, 0.388, 0.493, 0.561, 0.457, 0.791, 0.626, 0.519, 0.488, 0.608, 0.452]
    scorer.run( ds )
    for scobj, expect in zip( ds.deepscores, expected_scores ):
        assert scobj.score == pytest.approx( expect, abs=0.002 )
