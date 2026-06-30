import functools
import argparse
import re
import subprocess
import concurrent.futures


def dorun( command, timeout=None ):
    try:
        result = subprocess.run( command.split(), capture_output=True, timeout=timeout )
        if result.returncode != 0:
            return ( False, result.stdout.decode('utf-8'), result.stderr.decode('utf-8') )
        else:
            return ( True, None, None )
    except Exception as ex:
        return ( False, None, str(ex) )


def main():
    parser = argparse.ArgumentParser( 'parallel_runner.py' )
    parser.add_argument( "commandfile", help="File with commands to run, one per line." )
    parser.add_argument( "-n", "--num-subprocs", default=4, type=int, help="Number of subprocesses to run at once" )
    parser.add_argument( "-t", "--timeout", default=600., type=float, help="Timeout after this much time." )
    args = parser.parse_args()

    with open( args.commandfile ) as ifp:
        lines = [ line.strip() for line in ifp.readlines() ]

    commentline = re.compile( r'^\s*#' )
    blankline = re.compile( r'^\s*$' )
    lines = [ line for line in lines if ( not commentline.search(line) ) and ( not blankline.search(line) ) ]

    execer = concurrent.futures.ProcessPoolExecutor( max_workers=args.num_subprocs,
                                                     max_tasks_per_child=1 )
    runner = functools.partial( dorun, timeout=args.timeout )
    results = execer.map( runner, lines )

    failline = []
    failstdout = []
    failstderr = []
    for line, res in zip( lines, results ):
        if res[0]:
            print( f"Success: {line}" )
        else:
            print( f"FAILURE: {line}" )
            failline.append( line )
            failstdout.append( res[1] )
            failstderr.append( res[2] )

    for line, out, err in zip( failline, failstdout, failstderr ):
        print( "\n======================================================================" )
        print( f"stdout for failure of {line}\n{out}\n" )
        print( f"\nstderr for failure of {line}\n{err}\n" )


# ======================================================================

if __name__ == "__main__":
    main()
