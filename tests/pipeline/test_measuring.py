
def test_measuring( sim_lightcurve_one_complete_ds ):
    _ref, _refds, ds, pip = sim_lightcurve_one_complete_ds

    # This is a little ugly.
    #
    # Have the full datastore so that all the necessary data products
    #   and such are there.  But, to test measuring, we want to be able
    #   to inject things that look in certain ways to make sure the
    #   measurements come out right.  That will mean modifing the
    #   images, faking the detections to have the right positions,
    #   rerunning cutting, and then running measuring.
    #
    # First, clean out all data products from detections on so we can
    #   make our mess.

    ds.deepscore_set.delete_from_disk_and_database()
    ds.deepscore_set = None
    ds.deepscores = None
    ds.measurement_set.delete_from_disk_and_database()
    ds.measurement_set = None
    ds.measurements = None
    ds.cutouts.delete_from_disk_and_database()
    ds.cutouts = None
    ds.detections.delete_from_disk_and_database()
    ds.detections = None
