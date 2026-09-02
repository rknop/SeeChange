import pytest
import os
import argparse
import pathlib
import random
import functools
import logging
import time
import re

from util.logger import SCLogger
import util.util
from util.util import listify, ensure_file_does_not_exist, retry_with_sleep, reconstruct_commandline

# TODO : tests of most of the stuff in util...!  (Issue #384)


def test_listify():
    assert listify( None ) is None
    assert listify( ( None, ) ) == [ None ]
    assert listify( "test" ) == [ "test" ]
    assert listify( bytes([1, 2, 3]) ) == [ bytes([1, 2, 3]) ]
    assert listify( 1 ) == [ 1 ]
    assert listify( [ "a", "b", "c" ] ) == [ "a", "b", "c" ]
    assert listify( [ 1, 2, 3 ] ) == [ 1, 2, 3 ]
    assert listify( ( 1, 2, 3 ) ) == [ 1, 2, 3 ]
    listofset = listify( { 1, 2, 3 } )
    assert isinstance( listofset, list )
    assert set( listofset ) == { 1, 2, 3 }

    # Make sure require_string works right
    assert listify( "test", require_string=True ) == [ "test" ]
    assert listify( ( "a", "b", "c" ), require_string=True ) == [ "a", "b", "c" ]

    with pytest.raises( TypeError ):
        _ = listify( 1, require_string=True )
    with pytest.raises( TypeError ):
        _ = listify( [ 1, 2, 3], require_string=True )
    with pytest.raises( TypeError ):
        _ = listify( [ "a", 1 ], require_string=True )


def test_ensure_file_does_not_exist():
    fname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    fpath = pathlib.Path( fname )
    assert not fpath.exists()

    try:
        ensure_file_does_not_exist( fname )
        ensure_file_does_not_exist( fpath )

        fpath.mkdir()
        with pytest.raises( FileExistsError, match='.*exists but is not a regular file' ):
            ensure_file_does_not_exist( fname )
        with pytest.raises( FileExistsError, match='.*exists but is not a regular file' ):
            ensure_file_does_not_exist( fpath )
        fpath.rmdir()

        with open( fpath, "w" ) as ofp:
            ofp.write( "Hello, world\n" )

        with pytest.raises( FileExistsError, match='.*exists and delete is False' ):
            ensure_file_does_not_exist( fname )
        with pytest.raises( FileExistsError, match='.*exists and delete is False' ):
            ensure_file_does_not_exist( fpath )

        ensure_file_does_not_exist( fname, delete=True )
        assert not fpath.exists()
        with open( fpath, "w" ) as ofp:
            ofp.write( "Hello, world\n" )
        ensure_file_does_not_exist( fpath, delete=True )
        assert not fpath.exists()

    finally:
        if fpath.exists():
            if fpath.is_file():
                fpath.unlink()
            else:
                fpath.rmdir()


def test_retry_with_sleep( caplog ):
    caplog.set_level( logging.INFO, logger=SCLogger.instance()._logger.name )

    def _failfast():
        raise RuntimeError( "I failed because you told me to." )

    def _failntimes( s ):
        s['count'] -= 1
        if s['count'] >= 0:
            raise RuntimeError( f"Failed because count={s['count']+1}" )
        else:
            return 42

    def _succeed():
        return 42


    def _check_fail( t1, ex=None ):
        assert len( caplog.records ) == 5
        for i, t in enumerate( [ '0.12', '0.25', '0.50', '1.00' ] ):
            mat = re.search( r"Failed beating my head against a wall after (\d+) tries, will sleep "
                             r".* \(nominally (\d\.\d\d)s\) and try again.  Exception: I failed because "
                             r"you told me to.", caplog.records[i].msg )
            assert mat is not None
            assert int( mat.group(1) ) == i + 1
            assert mat.group(2) == t
            assert caplog.records[i].levelname == 'WARNING'
        mat = re.search( r"Repeated failures beating my head against a wall after (\d\.\d\d)s "
                         r"and (\d+) tries, giving up.  Last exception: I failed because you told me to.",
                         caplog.records[4].msg )
        tsleep = float(mat.group(1))
        # Expected sleeps : 0.125s + 0.25s + 0.5s + 1.0s = 1.875s  [5 tries]
        # With fuzz, say it should be ±~10%
        assert ( tsleep >= 1.8 ) and ( tsleep <= 2.0 )
        assert tsleep == pytest.approx( t1 - t0, rel=0.05 )
        assert int( mat.group(2) ) == 5
        assert caplog.records[4].levelname == 'ERROR'
        if ex is not None:
            mat = re.search( r"Failed beating my head against a wall after ([0-9]?\.[0-9]+)s and ([0-9]+) tries.",
                             str(ex) )
            assert mat is not None
            assert float(mat.group(1)) == pytest.approx( tsleep, abs=0.01 )
            assert int( mat.group(2) ) == 5

    # First, try ultimate failure

    t0 = time.monotonic()
    try:
        retry_with_sleep( _failfast, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                          failmessage="beating my head against a wall", exception_on_fail=True, randseed=42 )
    except Exception as ex:
        _check_fail( time.monotonic(), ex )

    # Next, try ultimate failure with a specific return
    caplog.clear()
    t0 = time.monotonic()
    res = retry_with_sleep( _failfast, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                            failmessage="beating my head against a wall", exception_on_fail=False, randseed=42 )
    _check_fail( time.monotonic() )
    assert res is None

    caplog.clear()
    t0 = time.monotonic()
    res = retry_with_sleep( _failfast, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                            failmessage="beating my head against a wall", exception_on_fail=False,
                            retval_on_fail='omg', randseed=42 )
    _check_fail( time.monotonic() )
    assert res == 'omg'


    # Next, try success
    caplog.clear()
    t0 = time.monotonic()
    res = retry_with_sleep( _succeed, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                            failmessage="succeeding", randseed=42 )
    assert time.monotonic() - t0 < 0.1
    assert res == 42
    assert len(caplog.records) == 0


    # Try success on the third try (after sleeping 0.125 and 0.25 seconds)
    caplog.clear()
    thing = { 'count': 3 }
    dothething = functools.partial( _failntimes, thing )
    t0 = time.monotonic()
    res = retry_with_sleep( dothething, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                            failmessage="getting the answer", randseed=42 )
    assert res == 42
    assert len(caplog.records) == 4
    for i, t in enumerate( [ '0.12', '0.25', '0.50' ] ):
        mat = re.search( r"Failed getting the answer after (\d+) tries, will sleep "
                         r".* \(nominally (\d.\d\d)s\) and try again.  Exception: "
                         r"Failed because count=(\d+)",
                         caplog.records[i].msg )
        assert mat is not None
        assert int( mat.group(1) ) == i + 1
        assert mat.group(2) == t
        assert int( mat.group(3) ) == 3 - i
        assert caplog.records[i].levelname == 'WARNING'
    mat = re.search( r"Succeeded getting the answer after (\d.\d\d)s and (\d+) tries.", caplog.records[3].msg )
    assert mat is not None
    assert float( mat.group(1) ) == pytest.approx( 0.875, rel=0.3 )
    assert int( mat.group(2) ) == 4
    assert caplog.records[3].levelname == 'INFO'


    # Test accept_exceptions
    with pytest.raises( RuntimeError, match="Failed because count=1" ):
        thing = { 'count': 1 }
        dothething = functools.partial( _failntimes, thing )
        retry_with_sleep( dothething, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                          failmessage="getting the answer", randseed=42,
                          accept_exceptions=ValueError )

    with pytest.raises( RuntimeError, match="Failed because count=1" ):
        thing = { 'count': 1 }
        dothething = functools.partial( _failntimes, thing )
        retry_with_sleep( dothething, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                          failmessage="getting the answer", randseed=42,
                          accept_exceptions=(ValueError,TypeError) )

    thing = { 'count': 1 }
    dothething = functools.partial( _failntimes, thing )
    res = retry_with_sleep( dothething, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                            accept_exceptions=RuntimeError )
    assert res == 42

    thing = { 'count': 1 }
    dothething = functools.partial( _failntimes, thing )
    res = retry_with_sleep( dothething, sleepmin=0.1, sleept=0.125, sleepfac=2, sleepmax=1.0, sleepfuzz=0.1,
                            accept_exceptions=(RuntimeError,ValueError) )
    assert res == 42


def test_reconstruct_commandline():

    rndstr = "".join( random.choices( "ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=10 ) )
    rndval = "".join( random.choices( "abcdefghijklmnopqrstuvwxyz", k=10 ) )

    try:
        os.environ[ rndstr ] = rndval

        cmds = [ { "cmd": [],
                   "nopositional": True,
                   "expected": {}
                  },
                 { "cmd": [ "64738" ],
                   "expected": { 'positional': '"64738"' }
                  },
                 { "cmd": [ "64738", "-q", "qux", "--string", "bar", "--bool", "--zeroormore" ],
                   "expected": { 'positional': '"64738"',
                                 '-q': '"qux"',
                                 '--string':  '"bar"',
                                 '--bool': None,
                                 '--zeroormore': None }
                  },
                 { "cmd": [ "64738", "-q", "qux", "-s", "bar", "-b", "-z" ],
                   "expected": { 'positional': '"64738"',
                                 '-q': '"qux"',
                                 '--string': '"bar"',
                                 '--bool': None,
                                 '--zeroormore': None }
                  },
                 { "cmd": [ "--also-string", "bar" ],
                   "nopositional": True,
                   "expected": { '--string': '"bar"' }
                  },
                 { "cmd": [ "-d", "13", "64738" ],
                   "expected": { 'positional': '"64738"',
                                 '--default': '"13"' }
                  }
                ]

        envses = [ ( None, "" ),
                   ( [],  "" ),
                   ( [ "HOME", rndstr ], f'  HOME="/home/seechange"\n  {rndstr}="{rndval}"' )
                  ]

        # omg it's a four-loop
        for cmd in cmds:
            nopositional = ( "nopositional" in cmd.keys() ) and cmd["nopositional"]
            for showcommand in ( False, True ):
                for showdefaults in ( False, True ):
                    for envs, envexpected in envses:
                        parser = argparse.ArgumentParser()
                        if not nopositional:
                            parser.add_argument( "positional" )
                        parser.add_argument( "-d", "--default", type=int, default=42 )
                        parser.add_argument( "-n", "--no-default", type=int )
                        parser.add_argument( "-s", "--string", "--also-string", default="foo" )
                        parser.add_argument( "-q" )
                        parser.add_argument( "-b", "--bool", action="store_true" )
                        parser.add_argument( "-z", "--zeroormore", nargs="*", default=['a', 'b', 'c'] )
                        parser.add_argument( "-o", "--oneormore", nargs="+" )
                        args = parser.parse_args( cmd["cmd"] )
                        if envs is None:
                            cmdstr = reconstruct_commandline( parser, args, showdefaults=showdefaults,
                                                              executable="testing" if showcommand else None )
                        else:
                            cmdstr = reconstruct_commandline( parser, args, showdefaults=showdefaults,
                                                              executable="testing" if showcommand else None,
                                                              envs=envs )

                        expected = ( '\nKey environment variables:\n'
                                     '  SEECHANGE_CONFIG="/seechange/tests/seechange_config_test.yaml"' )

                        if len(envexpected) > 0:
                            expected += f"\n{envexpected}"

                        expected += "\n"

                        if ( len(cmd["expected"]) > 0 ) or showcommand or showdefaults:
                            expected += "\nCommand line:\n"
                            slashy = ""
                            spaces = "  "
                            if showcommand:
                                expected += "  testing"
                                slashy = " \\\n"
                                spaces = "    "
                            for action in parser._actions:
                                optstr = util.util._reconstruct_commandline_canonical_option( action )
                                if optstr == "--help":
                                    continue
                                optstr = optstr if optstr is not None else "positional"
                                if ( ( optstr in cmd["expected"] ) or
                                     ( showdefaults and ( action.default is not None ) and
                                       ( not isinstance( action.nargs, int ) or ( action.nargs > 0 ) )
                                      )
                                    ):
                                    if optstr in cmd["expected"]:
                                        expval = cmd['expected'][optstr]
                                    else:
                                        if isinstance( action.default, list ):
                                            expval = ' '.join( f'"{v}"' for v in action.default )
                                        else:
                                            expval = f'"{action.default}"'
                                    if optstr == "positional":
                                        expected += f"{slashy}{spaces}{expval}"
                                    else:
                                        expected += f"{slashy}{spaces}{optstr}"
                                        if expval is not None:
                                            expected += " " + expval
                                    slashy = " \\\n"

                            expected += "\n"

                        SCLogger.debug( f"Command string:\n{cmdstr}" )
                        assert cmdstr == expected
    finally:
        if rndstr in os.environ:
            del os.environ[ rndstr ]
