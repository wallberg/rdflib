"""
These method recursively evaluate the SPARQL Algebra

evalQuery is the entry-point, it will setup context and
return the SPARQLResult object

evalPart is called on each level and will delegate to the right method

A rdflib.plugins.sparql.sparql.QueryContext is passed along, keeping
information needed for evaluation

A list of dicts (solution mappings) is returned, apart from GroupBy which may
also return a dict of list of dicts

"""

from __future__ import annotations

import collections
import itertools
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pyparsing import ParseException

from rdflib.graph import Graph
from rdflib.plugins.sparql import CUSTOM_EVALS, parser
from rdflib.plugins.sparql.aggregates import Aggregator
from rdflib.plugins.sparql.evalutils import (
    _ebv,
    _eval,
    _fill_template,
    _join,
    _minus,
    _val,
)
from rdflib.plugins.sparql.parserutils import CompValue, value
from rdflib.plugins.sparql.sparql import (
    AlreadyBound,
    FrozenBindings,
    FrozenDict,
    Query,
    QueryContext,
    SPARQLError,
)
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable

if TYPE_CHECKING:
    from rdflib.paths import Path

import json

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    orjson = None  # type: ignore[assignment, unused-ignore]
    _HAS_ORJSON = False

_Triple = Tuple[Identifier, Identifier, Identifier]


def eval_bgp(
    ctx: QueryContext, bgp: List[_Triple]
) -> Generator[FrozenBindings, None, None]:
    """
    A basic graph pattern
    """

    if not bgp:
        yield ctx.solution()
        return

    s, p, o = bgp[0]

    _s = ctx[s]
    _p = ctx[p]
    _o = ctx[o]

    # type error: Item "None" of "Optional[Graph]" has no attribute "triples"
    # type Argument 1 to "triples" of "Graph" has incompatible type "Tuple[Union[str, Path, None], Union[str, Path, None], Union[str, Path, None]]"; expected "Tuple[Optional[Node], Optional[Node], Optional[Node]]"
    for ss, sp, so in ctx.graph.triples((_s, _p, _o)):  # type: ignore[union-attr, arg-type]
        if None in (_s, _p, _o):
            c = ctx.push()
        else:
            c = ctx

        if _s is None:
            # type error: Incompatible types in assignment (expression has type "Union[Node, Any]", target has type "Identifier")
            c[s] = ss  # type: ignore[assignment]

        try:
            if _p is None:
                # type error: Incompatible types in assignment (expression has type "Union[Node, Any]", target has type "Identifier")
                c[p] = sp  # type: ignore[assignment]
        except AlreadyBound:
            continue

        try:
            if _o is None:
                # type error: Incompatible types in assignment (expression has type "Union[Node, Any]", target has type "Identifier")
                c[o] = so  # type: ignore[assignment]
        except AlreadyBound:
            continue

        for x in eval_bgp(c, bgp[1:]):
            yield x


def eval_extend(
    ctx: QueryContext, extend: CompValue
) -> Generator[FrozenBindings, None, None]:
    # TODO: Deal with dict returned from evalPart from GROUP BY

    for c in eval_part(ctx, extend.p):
        try:
            e = _eval(extend.expr, c.forget(ctx, _except=extend._vars))
            if isinstance(e, SPARQLError):
                raise e

            yield c.merge({extend.var: e})

        except SPARQLError:
            yield c


def eval_lazy_join(
    ctx: QueryContext, join: CompValue
) -> Generator[FrozenBindings, None, None]:
    """
    A lazy join will push the variables bound
    in the first part to the second part,
    essentially doing the join implicitly
    hopefully evaluating much fewer triples
    """
    for a in eval_part(ctx, join.p1):
        c = ctx.thaw(a)
        for b in eval_part(c, join.p2):
            yield b.merge(a)  # merge, as some bindings may have been forgotten


def eval_join(ctx: QueryContext, join: CompValue) -> Generator[FrozenDict, None, None]:
    # TODO: Deal with dict returned from evalPart from GROUP BY
    # only ever for join.p1

    if join.lazy:
        return eval_lazy_join(ctx, join)
    else:
        a = eval_part(ctx, join.p1)
        b = set(eval_part(ctx, join.p2))
        return _join(a, b)


def eval_union(ctx: QueryContext, union: CompValue) -> List[Any]:
    branch1_branch2 = []
    for x in eval_part(ctx, union.p1):
        branch1_branch2.append(x)
    for x in eval_part(ctx, union.p2):
        branch1_branch2.append(x)
    return branch1_branch2


def eval_minus(ctx: QueryContext, minus: CompValue) -> Generator[FrozenDict, None, None]:
    a = eval_part(ctx, minus.p1)
    b = set(eval_part(ctx, minus.p2))
    return _minus(a, b)


def eval_left_join(
    ctx: QueryContext, join: CompValue
) -> Generator[FrozenBindings, None, None]:
    # import pdb; pdb.set_trace()
    for a in eval_part(ctx, join.p1):
        ok = False
        c = ctx.thaw(a)
        for b in eval_part(c, join.p2):
            if _ebv(join.expr, b.forget(ctx)):
                ok = True
                yield b
        if not ok:
            # we've cheated, the ctx above may contain
            # vars bound outside our scope
            # before we yield a solution without the OPTIONAL part
            # check that we would have had no OPTIONAL matches
            # even without prior bindings...
            p1_vars = join.p1._vars
            if p1_vars is None or not any(
                _ebv(join.expr, b)
                for b in eval_part(ctx.thaw(a.remember(p1_vars)), join.p2)
            ):
                yield a


def eval_filter(
    ctx: QueryContext, part: CompValue
) -> Generator[FrozenBindings, None, None]:
    # TODO: Deal with dict returned from evalPart!
    for c in eval_part(ctx, part.p):
        if _ebv(
            part.expr,
            c.forget(ctx, _except=part._vars) if not part.no_isolated_scope else c,
        ):
            yield c


def eval_graph(
    ctx: QueryContext, part: CompValue
) -> Generator[FrozenBindings, None, None]:
    if ctx.dataset is None:
        raise Exception(
            "Non-conjunctive-graph doesn't know about "
            + "graphs. Try a query without GRAPH."
        )

    ctx = ctx.clone()
    graph: Union[str, Path, None, Graph] = ctx[part.term]
    prev_graph = ctx.graph
    if graph is None:
        for graph in ctx.dataset.contexts():
            # in SPARQL the default graph is NOT a named graph
            if graph == ctx.dataset.default_context:
                continue

            c = ctx.push_graph(graph)
            c = c.push()
            graph_solution = [{part.term: graph.identifier}]
            for x in _join(eval_part(c, part.p), graph_solution):
                x.ctx.graph = prev_graph
                yield x

    else:
        if TYPE_CHECKING:
            assert not isinstance(graph, Graph)
        # type error: Argument 1 to "get_context" of "ConjunctiveGraph" has incompatible type "Union[str, Path]"; expected "Union[Node, str, None]"
        c = ctx.push_graph(ctx.dataset.get_context(graph))  # type: ignore[arg-type]
        for x in eval_part(c, part.p):
            x.ctx.graph = prev_graph
            yield x


def eval_values(
    ctx: QueryContext, part: CompValue
) -> Generator[FrozenBindings, None, None]:
    for r in part.p.res:
        c = ctx.push()
        try:
            for k, v in r.items():
                if v != "UNDEF":
                    c[k] = v
        except AlreadyBound:
            continue

        yield c.solution()


def eval_multiset(ctx: QueryContext, part: CompValue):
    if part.p.name == "values":
        return eval_values(ctx, part)

    return eval_part(ctx, part.p)


def eval_part(ctx: QueryContext, part: CompValue) -> Any:
    # try custom evaluation functions
    for name, c in CUSTOM_EVALS.items():
        try:
            return c(ctx, part)
        except NotImplementedError:
            pass  # the given custome-function did not handle this part

    if part.name == "BGP":
        # Reorder triples patterns by number of bound nodes in the current ctx
        # Do patterns with more bound nodes first
        triples = sorted(
            part.triples, key=lambda t: len([n for n in t if ctx[n] is None])
        )

        return eval_bgp(ctx, triples)
    elif part.name == "Filter":
        return eval_filter(ctx, part)
    elif part.name == "Join":
        return eval_join(ctx, part)
    elif part.name == "LeftJoin":
        return eval_left_join(ctx, part)
    elif part.name == "Graph":
        return eval_graph(ctx, part)
    elif part.name == "Union":
        return eval_union(ctx, part)
    elif part.name == "ToMultiSet":
        return eval_multiset(ctx, part)
    elif part.name == "Extend":
        return eval_extend(ctx, part)
    elif part.name == "Minus":
        return eval_minus(ctx, part)

    elif part.name == "Project":
        return eval_project(ctx, part)
    elif part.name == "Slice":
        return eval_slice(ctx, part)
    elif part.name == "Distinct":
        return eval_distinct(ctx, part)
    elif part.name == "Reduced":
        return eval_reduced(ctx, part)

    elif part.name == "OrderBy":
        return eval_order_by(ctx, part)
    elif part.name == "Group":
        return eval_group(ctx, part)
    elif part.name == "AggregateJoin":
        return eval_aggregate_join(ctx, part)

    elif part.name == "SelectQuery":
        return eval_select_query(ctx, part)
    elif part.name == "AskQuery":
        return eval_ask_query(ctx, part)
    elif part.name == "ConstructQuery":
        return eval_construct_query(ctx, part)

    elif part.name == "ServiceGraphPattern":
        return eval_service_query(ctx, part)

    elif part.name == "DescribeQuery":
        return eval_describe_query(ctx, part)

    else:
        raise Exception("I dont know: %s" % part.name)


def eval_service_query(ctx: QueryContext, part: CompValue):
    res = {}
    match = re.match(
        "^service <(.*)>[ \n]*{(.*)}[ \n]*$",
        # type error: Argument 2 to "get" of "CompValue" has incompatible type "str"; expected "bool"  [arg-type]
        part.get("service_string", ""),  # type: ignore[arg-type]
        re.DOTALL | re.I,
    )

    if match:
        service_url = match.group(1)
        service_query = _build_query_string_for_service_call(ctx, match.group(2))

        query_settings = {"query": service_query, "output": "json"}
        headers = {
            "accept": "application/sparql-results+json",
            "user-agent": "rdflibForAnUser",
        }
        # GET is easier to cache so prefer that if the query is not to long
        if len(service_query) < 600:
            response = urlopen(
                Request(service_url + "?" + urlencode(query_settings), headers=headers)
            )
        else:
            response = urlopen(
                Request(
                    service_url,
                    data=urlencode(query_settings).encode(),
                    headers=headers,
                )
            )
        if response.status == 200:
            if _HAS_ORJSON:
                json_dict = orjson.loads(response.read())
            else:
                json_dict = json.loads(response.read())
            variables = res["vars_"] = json_dict["head"]["vars"]
            # or just return the bindings?
            res = json_dict["results"]["bindings"]
            if len(res) > 0:
                for r in res:
                    # type error: Argument 2 to "_yieldBindingsFromServiceCallResult" has incompatible type "str"; expected "Dict[str, Dict[str, str]]"
                    for bound in _yield_bindings_from_service_call_results(ctx, r, variables):  # type: ignore[arg-type]
                        yield bound
        else:
            raise Exception(
                "Service: %s responded with code: %s", service_url, response.status
            )


"""
    Build a query string to be used by the service call.
    It is supposed to pass in the existing bound solutions.
    Re-adds prefixes if added and sets the base.
    Wraps it in select if needed.
"""


def _build_query_string_for_service_call(ctx: QueryContext, service_query: str) -> str:
    try:
        parser.parse_query(service_query)
    except ParseException:
        # This could be because we don't have a select around the service call.
        service_query = "SELECT REDUCED * WHERE {" + service_query + "}"
        # type error: Item "None" of "Optional[Prologue]" has no attribute "namespace_manager"
        for p in ctx.prologue.namespace_manager.store.namespaces():  # type: ignore[union-attr]
            service_query = "PREFIX " + p[0] + ":" + p[1].n3() + " " + service_query
        # re add the base if one was defined
        # type error: Item "None" of "Optional[Prologue]" has no attribute "base"
        base = ctx.prologue.base  # type: ignore[union-attr]
        if base is not None and len(base) > 0:
            service_query = "BASE <" + base + "> " + service_query
    sol = [v for v in ctx.solution() if isinstance(v, Variable)]
    if len(sol) > 0:
        variables = " ".join([v.n3() for v in sol])
        variables_bound = " ".join([ctx.get(v).n3() for v in sol])
        service_query = (
            service_query + "VALUES (" + variables + ") {(" + variables_bound + ")}"
        )
    return service_query


def _yield_bindings_from_service_call_results(
    ctx: QueryContext, r: Dict[str, Dict[str, str]], variables: List[str]
) -> Generator[FrozenBindings, None, None]:
    res_dict: Dict[Variable, Identifier] = {}
    for var in variables:
        if var in r and r[var]:
            var_binding = r[var]
            var_type = var_binding["type"]
            if var_type == "uri":
                res_dict[Variable(var)] = URIRef(var_binding["value"])
            elif var_type == "literal":
                res_dict[Variable(var)] = Literal(
                    var_binding["value"],
                    datatype=var_binding.get("datatype"),
                    lang=var_binding.get("xml:lang"),
                )
            # This is here because of
            # https://www.w3.org/TR/2006/NOTE-rdf-sparql-json-res-20061004/#variable-binding-results
            elif var_type == "typed-literal":
                res_dict[Variable(var)] = Literal(
                    var_binding["value"], datatype=URIRef(var_binding["datatype"])
                )
            elif var_type == "bnode":
                res_dict[Variable(var)] = BNode(var_binding["value"])
            else:
                raise ValueError(f"invalid type {var_type!r} for variable {var!r}")
    yield FrozenBindings(ctx, res_dict)


def eval_group(ctx: QueryContext, group: CompValue):
    """
    http://www.w3.org/TR/sparql11-query/#defn_algGroup
    """
    # grouping should be implemented by evalAggregateJoin
    return eval_part(ctx, group.p)


def eval_aggregate_join(
    ctx: QueryContext, agg: CompValue
) -> Generator[FrozenBindings, None, None]:
    # import pdb ; pdb.set_trace()
    p = eval_part(ctx, agg.p)
    # p is always a Group, we always get a dict back

    group_expr = agg.p.expr
    res: Dict[Any, Any] = collections.defaultdict(
        lambda: Aggregator(aggregations=agg.A)
    )

    if group_expr is None:
        # no grouping, just COUNT in SELECT clause
        # get 1 aggregator for counting
        aggregator = res[True]
        for row in p:
            aggregator.update(row)
    else:
        for row in p:
            # determine right group aggregator for row
            k = tuple(_eval(e, row, False) for e in group_expr)
            res[k].update(row)

    # all rows are done; yield aggregated values
    for aggregator in res.values():
        yield FrozenBindings(ctx, aggregator.get_bindings())

    # there were no matches
    if len(res) == 0:
        yield FrozenBindings(ctx)


def eval_order_by(
    ctx: QueryContext, part: CompValue
) -> Generator[FrozenBindings, None, None]:
    res = eval_part(ctx, part.p)

    for e in reversed(part.expr):
        reverse = bool(e.order and e.order == "DESC")
        res = sorted(
            res, key=lambda x: _val(value(x, e.expr, variables=True)), reverse=reverse
        )

    return res


def eval_slice(ctx: QueryContext, slice: CompValue):
    res = eval_part(ctx, slice.p)

    return itertools.islice(
        res,
        slice.start,
        slice.start + slice.length if slice.length is not None else None,
    )


def eval_reduced(
    ctx: QueryContext, part: CompValue
) -> Generator[FrozenBindings, None, None]:
    """apply REDUCED to result

    REDUCED is not as strict as DISTINCT, but if the incoming rows were sorted
    it should produce the same result with limited extra memory and time per
    incoming row.
    """

    # This implementation uses a most recently used strategy and a limited
    # buffer size. It relates to a LRU caching algorithm:
    # https://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used_.28LRU.29
    max = 1
    # TODO: add configuration or determine "best" size for most use cases
    # 0: No reduction
    # 1: compare only with the last row, almost no reduction with
    #    unordered incoming rows
    # N: The greater the buffer size the greater the reduction but more
    #    memory and time are needed

    # mixed data structure: set for lookup, deque for append/pop/remove
    mru_set = set()
    mru_queue: Deque[Any] = collections.deque()

    for row in eval_part(ctx, part.p):
        if row in mru_set:
            # forget last position of row
            mru_queue.remove(row)
        else:
            # row seems to be new
            yield row
            mru_set.add(row)
            if len(mru_set) > max:
                # drop the least recently used row from buffer
                mru_set.remove(mru_queue.pop())
        # put row to the front
        mru_queue.appendleft(row)


def eval_distinct(
    ctx: QueryContext, part: CompValue
) -> Generator[FrozenBindings, None, None]:
    res = eval_part(ctx, part.p)

    done = set()
    for x in res:
        if x not in done:
            yield x
            done.add(x)


def eval_project(ctx: QueryContext, project: CompValue):
    res = eval_part(ctx, project.p)
    return (row.project(project.PV) for row in res)


def eval_select_query(
    ctx: QueryContext, query: CompValue
) -> Mapping[str, Union[str, List[Variable], Iterable[FrozenDict]]]:
    res: Dict[str, Union[str, List[Variable], Iterable[FrozenDict]]] = {}
    res["type_"] = "SELECT"
    res["bindings"] = eval_part(ctx, query.p)
    res["vars_"] = query.PV
    return res


def eval_ask_query(ctx: QueryContext, query: CompValue) -> Mapping[str, Union[str, bool]]:
    res: Dict[str, Union[bool, str]] = {}
    res["type_"] = "ASK"
    res["askAnswer"] = False
    for x in eval_part(ctx, query.p):
        res["askAnswer"] = True
        break

    return res


def eval_construct_query(
    ctx: QueryContext, query: CompValue
) -> Mapping[str, Union[str, Graph]]:
    template = query.template

    if not template:
        # a construct-where query
        template = query.p.p.triples  # query->project->bgp ...

    graph = Graph()

    for c in eval_part(ctx, query.p):
        graph += _fill_template(template, c)

    res: Dict[str, Union[str, Graph]] = {}
    res["type_"] = "CONSTRUCT"
    res["graph"] = graph

    return res


def eval_describe_query(ctx: QueryContext, query) -> Dict[str, Union[str, Graph]]:
    # Create a result graph and bind namespaces from the graph being queried
    graph = Graph()
    # type error: Item "None" of "Optional[Graph]" has no attribute "namespaces"
    for pfx, ns in ctx.graph.namespaces():  # type: ignore[union-attr]
        graph.bind(pfx, ns)

    to_describe = set()

    # Explicit IRIs may be provided to a DESCRIBE query.
    # If there is a WHERE clause, explicit IRIs may be provided in
    # addition to projected variables. Find those explicit IRIs and
    # prepare to describe them.
    for iri in query.PV:
        if isinstance(iri, URIRef):
            to_describe.add(iri)

    # If there is a WHERE clause, evaluate it then find the unique set of
    # resources to describe across all bindings and projected variables
    if query.p is not None:
        bindings = eval_part(ctx, query.p)
        to_describe.update(*(set(binding.values()) for binding in bindings))

    # Get a CBD for all resources identified to describe
    for resource in to_describe:
        # type error: Item "None" of "Optional[Graph]" has no attribute "cbd"
        ctx.graph.cbd(resource, target_graph=graph)  # type: ignore[union-attr]

    res: Dict[str, Union[str, Graph]] = {}
    res["type_"] = "DESCRIBE"
    res["graph"] = graph

    return res


def eval_query(
    graph: Graph,
    query: Query,
    init_bindings: Optional[Mapping[str, Identifier]] = None,
    base: Optional[str] = None,
) -> Mapping[Any, Any]:
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
    main = query.algebra

    init_bindings = dict((Variable(k), v) for k, v in (init_bindings or {}).items())

    ctx = QueryContext(
        graph, init_bindings=init_bindings, dataset_clause=main.datasetClause
    )

    ctx.prologue = query.prologue

    return eval_part(ctx, main)
