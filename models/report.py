import datetime
import time
import inspect
import textwrap
import re

from psycopg import sql
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as sqlUUID

from models.base import Base, SeeChangeBase, UUIDMixin, PGDB
from models.provenance import Provenance

from util.logger import SCLogger
from util.util import parse_dateobs, asUUID


class Report(Base, UUIDMixin):
    """A report on the status of a pipeline."""
    __tablename__ = 'reports'

    pipeline_name = sa.Column(
        sa.Text,
        nullable = False,
        index = True,
        doc = "Name of the pipeline (the _name property of the class)"
    )

    starting_class = sa.Column(
        sa.Text,
        nullable = False,
        index = True,
        doc = "Class of the starting point of the pipeline."
    )

    starting_point = sa.Column(
        sqlUUID,
        nullable = False,
        index = True
        doc = "ID of the object that is the starting point for the pipeline."
    )

    final_prov_id = sa.Column(
        sa.Text,
        sa.ForeignKey( 'provenances._id', ondelete='CASCADE', name='reports_final_prov_id_fkey' ),
        nullable = False,
        index = True,
        doc = "The final provenance expected for this run of this pipeline."
    )

    process_steps = sa.Column(
        sa.ARRAY( sa.Text ),
        nulalble = False,
        index = False,
        server_default = "[]::text[]"
        doc = "Steps of the pipeline.",
    )
    
    process_provid = sa.Column(
        JSONB,
        nullable = True,
        index = False,
        doc = ( "Dictionary of process→provenance_id for the provenances used in pipeline process.  "
                "The keys of this dictionary should be the same as the elements of process_steps.  "
                "Both are here because you can't count on JSONB dictionaries being ordered." )
    )

    start_time = sa.Column(
        sa.DateTime,
        nullable = False,
        index = False,
        doc = "When did the pipeline say it started."
    )

    end_time = sa.Column(
        sa.DateTime
        nullable = True,
        index = False,
        doc = "When the pipeline ended, if it told us it ended."
    )

    success = sa.Column(
        sa.Boolean,
        nullable = True,
        index = False,
        doc = "Was the pipeline successful?  NULL = dunno, or dunno yet."
    )

    # These next two are not a composite foreign key because
    #   PipelineWorkers only holds active workers, and we want to be
    #   able to see reports for workers that have exited.
    cluster_id = sa.Column(
        sa.Text,
        nullable = True,
        index = False,
        doc = "ID of the cluster that ran this section (see PipelineWorker). "
    )

    node_id = sa.Column(
        sa.Text,
        nullable = True,
        index = False,
        doc = "ID of the node where the worker/process ran this section (see PipelineWorker). "
    )

    error_step = sa.Column(
        sa.Text,
        nullable = True,
        index = False,
        doc = "Name of the processing step where an error occurred. "
    )

    error_type = sa.Column(
        sa.Text,
        nullable = True,
        index = False,
        doc= "Type of error that was raised during processing. "
    )

    error_message = sa.Column(
        sa.Text,
        nullable = True,
        index = False,
        doc = "Error message that was raised during processing. "
    )

    warnings = sa.Column(
        sa.ARRAY( sa.Text ),
        nullable = False,
        index = False,
        server_default = "[]::text[]",
        doc=( "Warnings that were raised during processing.  Each warning begins with the processing step name, "
              "followed by a colon and the the warning message." )
    )

    process_memory = sa.Column(
        JSONB,
        nullable = False,
        index = False,
        server_default = '{}',
        doc = ( "Memory usage of the process during processing. Each key in the dictionary is for a "
                "processing step, and the value is the memory usage in megabytes." )
    )

    process_runtime = sa.Column(
        JSONB,
        nullable = False,
        server_default = '{}',
        doc =( "Runtime of the process during processing. Each key in the dictionary is for a "
               "processing step, and the value is the runtime in seconds." )
    )

    progress_steps_bitflag = sa.Column(
        sa.Integer,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=False,
        doc=( "Bitflag recording what processing steps have already been applied to this section. "
              "Bit n corresponds to 0-offset index n in the process_steps array.  We're assuming there "
              "will never be more than 31 steps in any pipeline...." )
    )

    def __init__( self, pipeline_name, starting_class, starting_point, final_prov_id,
                  process_steps, process_provid, start_time=None, **kwargs
                 ):
        if any ( x is None for x in ( pipeline_name, starting_class, starting_point, final_prov_id ) ):
            raise ValueError( "None of pipeline_name, starting_class, or starting_point can be None" )
        final_prov_id = final_prov_id.id if isinstance( final_prov_id, Provenance ) else final_prov_id
        process_steps = [] if process_steps is None else process_steps
        process_provid = {} if process_provid is None else process_provid
        start_time = datetime.datetime.now( datetime.UTC ) if start_time is None else parse_dateobs( start_time )
        starting_point = asUUID( starting_point )
        if inspect.isclass( starting_class ):
            starting_class = starting_class.__name__
        elif not isinstance( starting_class, str ):
            raise TypeError( f"starting_class must either be a class or the name of a class" )

        super().__init__( pipeline_name=pipeline_name, starting_class=starting_class,
                          starting_point=starting_point, final_prov_id=final_prov_id,
                          process_steps=process_steps, process_provid=process_provid, start_time=start_time,
                          **kwargs )

        self.last_udpate = datetime.datetime.now( datetime.UTC )
        self.errored_out = False
        self.finished = False


    def finish( self, pgdb=None ):
        self.finished = True
        self.end_time = datetime.datetime.now( datetime.UTC )
        with PGDB( pgdb ) as pgdb:
            pgdb.execute( sql.SQL( "UPDATE reports SET end_time={t} WHERE _id={selfid}" )
                          .format( selfid=self.id, t=self.end_time ) )
            pgdb.commit()


    def error_out( self, step, exception, pgdb=None ):
        if self.errored_out or self.finished:
            raise RuntimeError( "Can't error out after erroring out or finishing." )

        self.errored_out = True
        self.error_step = step
        self.error_type = exception.__name__
        self.error_message = str(exception)
        self.end_time = datetime.datetime.now( datetime.UTC )
        with PGDB( pgdb ) as pgdb:
            pgdb.execute( sql.SQL( "UPDATE reports SET error_step={errstep}, error_type={errtype}, "
                                   "error_message={errmsg}, end_time={tf}, success=False "
                                   "WHERE id={selfid}"
                                  ).format( selfid=self.id, errstep=step, errtype=self.error_type,
                                            errmsg=self.error_message, end_time=self.end_time )
                         )
            pgdb.commit()


    def complete_step( self, step, memory=None, pgdb=None ):
        if self.errored_out or self.finished:
            raise RuntimeError( "Can't complete a step after erroring out or finishing!" )

        try:
            bit = self.process_steps.index( step )
        except ValueError:
            raise ValueError( f"Unknown process step {step}" )
        
        now = datetime.datetime.now( datetime.UTC )
        dt = ( now - self.last_update ).total_seconds()
        self.last_update = now

        with PGDB( pgdb ) as pgdb:
            # ...maybe should lock the report table here, but not going to, because
            #  I think that's overdoing it.  Each instance of a running pipeline is going
            #  to get a new Report object with a new id, so in general there should not
            #  be collisions.
            me = Report.get_by_id( me.id, pgdb=pgdb )
            if ( me is None ):
                raise RuntimeError( "Error completing step, report not saved to database!" )
            self.process_memory = me.process_memory
            self.process_runtime = me.process_runtime
            self.process_steps_bitflag = me.progress_steps_bitflag
            
            self.process_runtime[step] = dt
            self.progress_steps_bitflag |= 1 << bit
            if memory is not None:
                self.process_memory[step] = memory

            pgdb.execute( sql.SQL(
                "UPDATE reports SET process_memory={mem}, process_runtime={t}, progress_steps_bitflag={bits} "
                "WHERE _id={selfid}"
            ).format( selfid=self.id, mem=self.process_memory, t=self.process_runtime,
                      bits=self.progress_steps_bitflag ) )
            pgdb.commit()

    def warning( self, step, message, pgdb=None ):
        if self.errored_out or self.finished:
            raise RuntimeError( "Can't add warnings after erroring out or finishing" )

        self.warnings = [] if self.warnings is None else self.warnings
        newwarning = f"{step}: {message}"
        self.warnings.append( newwarning )
        with PGDB( pgdb ) as pgdb:
            pgdb.execute( sql.SQL(
                "UPDATE reports SET warnings=array_append(warnings, {newwarning}) WHERE _id={selfid}"
            ).format( selfid=self.id, newwarning=newwarining ) )
            pgdb.commit()
            
        
    @classmethod
    def get_reports( cls, pipeline_name, starting_class, starting_point, final_prov_id=None, pgdb=None ):
        with PGDB( pgdb, dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT * FROM reports
                WHERE pipeline_name={pip}
                  AND starting_class={cls}
                  AND starting_point={startid}
                  {fprov}
                ORDER BY start_time
                """
            ) ).format( pip=pipeline_name, cls=starting_class, startid=starting_point,
                        fprov=( sql.SQL( "AND final_prov_id={provid}" ).format( provid=final_prov_id )
                                if final_prov_id is not None else sql.SQL( "" ) ) )
            rows = pgdb.execute( q )

        return [ cls.create( **r ) for r in rows ]

    @classmethod
    def get_reports_for_exposure( cls, pipeline_name, exposureid, final_prov_id=None, pgdb=None ):
        if isinstance( exposureid, Exposure ):
            exposureid = exposureid.id
        exposureid = asUUID( exposureid )

        with PGDB( pgdb, dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT r.*, i.section_id FROM reports r
                INNER JOIN images i ON r.starting_point=i._id
                INNER JOIN exposure e ON i.exposure_id=e._id
                WHERE e._id={expid}
                  AND r.pipeline_name={pip}
                  {fprov}
                ORDER BY i.section_id, r.start_time
                """
            ) ).format( expid=exposureid, pip=pipeline_name,
                        fprov=( sql.SQL( "AND final_prov_id={provid} " ).format( provid=final_prov_id )
                                if final_prov_id is not None else sql.SQL( "" ) ) )
            rows = pgdb.execute( q )

        reports = {}
        for row in rows:
            if row['section_id'] not in reports:
                reports['section_id'] = []
            del row['section_id']
            reports['section_id'].append( cls.create( **row ) )

        return reports
                         
