from __future__ import annotations

import math
import sys
from typing import Set, Tuple

from rdflib import Graph, Literal
from rdflib.namespace import Namespace
from rdflib.plugins.sparql.processor import process_update
from rdflib.term import Node


def triple_set(graph: Graph) -> Set[Tuple[Node, Node, Node]]:
    return set(graph.triples((None, None, None)))


class TestSPARQLParser:
    def test_insert_recursionlimit(self) -> None:
        # These values are experimentally determined
        # to cause the RecursionError reported in
        # https://github.com/RDFLib/rdflib/issues/1336
        resource_count = math.ceil(sys.getrecursionlimit() / (33 - 3))
        self.do_insert(resource_count)

    def test_insert_large(self) -> None:
        self.do_insert(200)

    def do_insert(self, resource_count: int) -> None:
        EGV = Namespace("http://example.org/vocab#")  # noqa: N806
        EGI = Namespace("http://example.org/instance#")  # noqa: N806
        prop0, prop1, prop2 = EGV["prop0"], EGV["prop1"], EGV["prop2"]
        g0 = Graph()
        for index in range(resource_count):
            resource = EGI[f"resource{index}"]
            g0.add((resource, prop0, Literal(index)))
            g0.add((resource, prop1, Literal("example resource")))
            g0.add((resource, prop2, Literal(f"resource #{index}")))

        g0ntriples = g0.serialize(format="ntriples")
        g1 = Graph()

        assert triple_set(g0) != triple_set(g1)

        process_update(g1, f"INSERT DATA {{ {g0ntriples!s} }}")

        assert triple_set(g0) == triple_set(g1)
