import functools
import requests

from util.logger import SCLogger
from util.util import retry_with_sleep


def retry_post( url, json={}, returnjson=True, retries=5, timeout0=1.0, timeoutfac=2.0, timeoutjitter=0.1,
                failwarn=True ):
    """Post to a URL and get the response, with a fail/retry loop.

    If the post fails, sleep and try again.  The first sleep will be for
    timeout0 seconds.  Subsequent sleeps are each increased by a factor
    of timeoutfac.  Sleep times are randomized by a guassian with
    σ=timeoutjitter times the current timeout (with a hard minimum of
    1/4 the current timeout); this is so lots of processes all
    contacting the same URL at once don't have synchronized failures as
    the server is overloaded with all the same tries all at once.  After
    retries tries, raise an exception whose contents will, hopefully,
    include something about the last failure.

    Does not handle any kind of authentication.  See rkauth_client.py
    for something that uses the RKAuth system.

    Parameters
    ----------
      url : string
        URL to post to

      json : dict, default {}
        Parmaeters to send to the post as a dictionary.

      returnjson : boolean, default True
        If True, then we are expecting the response to be json, and the
        parsed json will be returned.  If False, just return the python
        requests.Response object.

      retries : int, default 5
        Number of retries

      timeout0 : float, default 1.0
        First mean timeout in seconds.

      timeoutfac : float, default 2.0
        Multiple the mean timeout by this factor each retry

      timeoutjitter : float, default 0.2
        Randomize the timeout by a Gaussian with σ of this factor times
        the current mean timeout.  (In reality: the maximum of that
        randomized timeout and 1/4 of the current mean timeout.)  Make
        ≤0 to have no jitter in the sleep times.

      failwarn : bool, default True
        If True, log messages about failures before the last retry will
        be warnings; otherwise they will be debug messages.  The last
        failure will always be logged as an error.

    Returns
    -------
      requests.Response or python object

      depending on whether returnjson is False or True

    """
    raise NotImplementedError( "retry_post is deprecated" )

    maxsleep = timeout0 * ( 2 ** (retries-1) )
    do_the_thing = functools.partial( requests.post, url, json=json )

    def _errmsg( res ):
        SCLogger.errror( f"Failed to contact {url}; got HTTP status {res.status_code}: \""
                         f"{res.text if len(res.text) < 240 else res.text[:240]}\"." )

    res = retry_with_sleep( do_the_thing, sleepmin=timeout0/2., sleept=timeout0, sleepfac=timeoutfac,
                            sleepfuzz=timeoutjitter, sleepmax=maxsleep,
                            failmessage=f"contacting {url}",
                            good_returns=[ 200 ], return_attr='status_code', badreturn_handler=_errmsg )
    return res.json()
