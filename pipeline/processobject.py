from util.config import NotDefined


class ProcessObject:

    _pars_class = None
    _process_name = None
    
    def __init__( self, pipeline=None, ignore_missing_config_values=True, **kwargs ):
        """Initialize a process object or a pipeline.

        All process objects should have this class as a superclass, and
        call super().__init__( **kwargs ) at the right time.  (Thought
        and understanding required to know what the right time is.)

        They must define static variables _pars_class and _process_name,
        and isntance method run().

        """

        if self.__class__.__name__ == "ProcessObject":
            raise RuntimeError( "ProcessObject is abstract, only make subclasses." )

        if ( self._pars_class is None ) or ( self._process_name is None ):
            raise RuntimeError( f"{self.__class__.__name__} definition is broken; needs to define "
                                f"_pars_class and _process_name." )
        
        from pipeline.pipeline import Pipeline
        if not isinstance( pipeline, ( Pipeline, None ) ):
            raise TypeError( f"pipeline must be None or a Pipeline object, not a {type(pipeline)}" )
        
        self.pars = self._pars_class()
        if pipeline is not None:
            self.pars.merge_configs( after=[ self._process_name,
                                             getattr( pipeline.pars, self._process_name ),
                                             kwargs
                                            ],
                                     ignore_missing_config_values=ignore_missing_config_values )
        else:
            self.pars.merge_configs( after=[ self._process_name, kwargs ],
                                     ignore_missing_config_values=ignore_missing_config_values )

        
    def __call__( self, *args, **kwargs ):
        self.run( *args, **kwargs )

    def run( self, *args, **kwargs ):
        raise NotImplementedError( f"{self.__class__.__name__} needs to implement run" )
