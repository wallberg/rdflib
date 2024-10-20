"""
Code for tying SPARQL Engine into RDFLib

These should be automatically registered with RDFLib

"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from rdflib.graph import Graph
from rdflib.plugins.sparql.algebra import translateQuery, translateUpdate
from rdflib.plugins.sparql.evaluate import eval_query
from rdflib.plugins.sparql.parser import parse_query, parse_update
from rdflib.plugins.sparql.sparql import Query, Update
from rdflib.plugins.sparql.update import eval_update
from rdflib.query import Processor, Result, UpdateProcessor
from rdflib.term import Identifier


def prepare_query(
    query_string: str,
    init_ns: Optional[Mapping[str, Any]] = None,
    base: Optional[str] = None,
) -> Query:
    """
    Parse and translate a SPARQL Query
    """
    if init_ns is None:
        init_ns = {}
    ret = translateQuery(parse_query(query_string), base, init_ns)
    ret._original_args = (query_string, init_ns, base)
    return ret


def prepare_update(
    update_string: str,
    init_ns: Optional[Mapping[str, Any]] = None,
    base: Optional[str] = None,
) -> Update:
    """
    Parse and translate a SPARQL Update
    """
    if init_ns is None:
        init_ns = {}
    ret = translateUpdate(parse_update(update_string), base, init_ns)
    ret._original_args = (update_string, init_ns, base)
    return ret


def process_update(
    graph: Graph,
    update_string: str,
    init_bindings: Optional[Mapping[str, Identifier]] = None,
    init_ns: Optional[Mapping[str, Any]] = None,
    base: Optional[str] = None,
) -> None:
    """
    Process a SPARQL Update Request
    returns Nothing on success or raises Exceptions on error
    """
    eval_update(
        graph, translateUpdate(parse_update(update_string), base, init_ns), init_bindings
    )


class SPARQLResult(Result):
    def __init__(self, res: Mapping[str, Any]):
        Result.__init__(self, res["type_"])
        self.vars = res.get("vars_")
        # type error: Incompatible types in assignment (expression has type "Optional[Any]", variable has type "MutableSequence[Mapping[Variable, Identifier]]")
        self.bindings = res.get("bindings")  # type: ignore[assignment]
        self.askAnswer = res.get("askAnswer")
        self.graph = res.get("graph")


class SPARQLUpdateProcessor(UpdateProcessor):
    def __init__(self, graph):
        self.graph = graph

    def update(
        self,
        str_or_query: Union[str, Update],
        init_bindings: Optional[Mapping[str, Identifier]] = None,
        init_ns: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        .. caution::

           This method can access indirectly requested network endpoints, for
           example, query processing will attempt to access network endpoints
           specified in ``SERVICE`` directives.

           When processing untrusted or potentially malicious queries, measures
           should be taken to restrict network and file access.

           For information on available security measures, see the RDFLib
           :doc:`Security Considerations </security_considerations>`
           documentation.
        """

        if isinstance(str_or_query, str):
            str_or_query = translateUpdate(parse_update(str_or_query), initNs=init_ns)

        return eval_update(self.graph, str_or_query, init_bindings)


class SPARQLProcessor(Processor):
    def __init__(self, graph):
        self.graph = graph

    # NOTE on type error: this is because the super type constructor does not
    # accept base argument and thie position of the DEBUG argument is
    # different.
    # type error: Signature of "query" incompatible with supertype "Processor"
    def query(  # type: ignore[override]
        self,
        str_or_query: Union[str, Query],
        init_bindings: Optional[Mapping[str, Identifier]] = None,
        init_ns: Optional[Mapping[str, Any]] = None,
        base: Optional[str] = None,
        debug: bool = False,
    ) -> Mapping[str, Any]:
        """
        Evaluate a query with the given initial bindings, and initial
        namespaces. The given base is used to resolve relative URIs in
        the query and will be overridden by any BASE given in the query.

        .. caution::

           This method can access indirectly requested network endpoints, for
           example, query processing will attempt to access network endpoints
           specified in ``SERVICE`` directives.

           When processing untrusted or potentially malicious queries, measures
           should be taken to restrict network and file access.

           For information on available security measures, see the RDFLib
           :doc:`Security Considerations </security_considerations>`
           documentation.
        """

        if isinstance(str_or_query, str):
            str_or_query = translateQuery(parse_query(str_or_query), base, init_ns)

        return eval_query(self.graph, str_or_query, init_bindings, base)
