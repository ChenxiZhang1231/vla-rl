# Copyright 2025, Shanghai Innovation Institute.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import copy
from typing import List, Set, Dict, Tuple

# Assuming node.py, task_graph.py, task_loader.py are accessible
from siirl.workers.dag import Node, NodeType, TaskGraph
from siirl.workers.dag.task_loader import generate_structural_signature, get_all_downstream_nodes_recursive, get_all_ancestors, find_all_paths, split_single_structure, split_by_fan_out_to_exits, split_by_reconverging_paths, discover_and_split_parallel_paths


# Helper to get a set of structural signatures for a list of graphs
def get_signatures(graphs: List[TaskGraph]) -> List[str]:
    """Generates a sorted list of structural signatures for a list of graphs."""
    return sorted([generate_structural_signature(g) for g in graphs])


class TestTaskLoaderInternals(unittest.TestCase):
    """Tests for internal helper functions in task_loader.py."""

    def setUp(self):
        # Simple graph: A -> B -> C, A -> D
        #      A
        #     / \
        #    B   D
        #    |
        #    C
        self.node_a = Node("A", NodeType.DATA_LOAD)
        self.node_b = Node("B", NodeType.COMPUTE, dependencies=["A"])
        self.node_c = Node("C", NodeType.COMPUTE, dependencies=["B"])
        self.node_d = Node("D", NodeType.COMPUTE, dependencies=["A"])
        self.graph1 = TaskGraph("g1")
        self.graph1.add_nodes([self.node_a, self.node_b, self.node_c, self.node_d])
        self.graph1.build_adjacency_lists()  # Crucial for many helpers

        # Linear graph: L1 -> L2 -> L3
        self.ln1 = Node("L1", NodeType.DATA_LOAD)
        self.ln2 = Node("L2", NodeType.COMPUTE, dependencies=["L1"])
        self.ln3 = Node("L3", NodeType.MODEL_TRAIN, dependencies=["L2"])
        self.linear_graph = TaskGraph("linear")
        self.linear_graph.add_nodes([self.ln1, self.ln2, self.ln3])
        self.linear_graph.build_adjacency_lists()

        # Empty graph
        self.empty_graph = TaskGraph("empty")

    def test_generate_structural_signature(self):
        sig1 = generate_structural_signature(self.graph1)

        graph1_reordered_nodes = TaskGraph("g1_reordered")
        # Add nodes in different order but same structure
        graph1_reordered_nodes.add_node(copy.deepcopy(self.node_d))
        graph1_reordered_nodes.add_node(copy.deepcopy(self.node_c))
        graph1_reordered_nodes.add_node(copy.deepcopy(self.node_b))
        graph1_reordered_nodes.add_node(copy.deepcopy(self.node_a))
        sig2 = generate_structural_signature(graph1_reordered_nodes)
        self.assertEqual(sig1, sig2, "Signatures should match for structurally identical graphs regardless of node add order.")

        sig_empty = generate_structural_signature(self.empty_graph)
        self.assertTrue("empty_structure" in sig_empty)

        sig_linear = generate_structural_signature(self.linear_graph)
        self.assertTrue("L1" in sig_linear and "L2" in sig_linear and "L3" in sig_linear)
        self.assertTrue("e(L1->L2)" in sig_linear)

    def test_get_all_downstream_nodes_recursive(self):
        # For self.graph1 (A -> B -> C, A -> D)
        # Downstream of A should be A, B, C, D
        downstream_a = get_all_downstream_nodes_recursive(self.graph1, "A")
        self.assertSetEqual(downstream_a, {"A", "B", "C", "D"})

        # Downstream of B should be B, C
        downstream_b = get_all_downstream_nodes_recursive(self.graph1, "B")
        self.assertSetEqual(downstream_b, {"B", "C"})

        # Downstream of C should be C
        downstream_c = get_all_downstream_nodes_recursive(self.graph1, "C")
        self.assertSetEqual(downstream_c, {"C"})

        # Downstream of D should be D
        downstream_d = get_all_downstream_nodes_recursive(self.graph1, "D")
        self.assertSetEqual(downstream_d, {"D"})

        # Non-existent node
        downstream_non_existent = get_all_downstream_nodes_recursive(self.graph1, "Z")
        self.assertSetEqual(downstream_non_existent, set())

        # Empty graph
        downstream_empty = get_all_downstream_nodes_recursive(self.empty_graph, "A")
        self.assertSetEqual(downstream_empty, set())

    def test_get_all_ancestors(self):
        # For self.graph1 (A -> B -> C, A -> D)
        ancestors_c = get_all_ancestors(self.graph1, "C")  # Ancestors of C are B, A
        self.assertSetEqual(ancestors_c, {"A", "B"})

        ancestors_b = get_all_ancestors(self.graph1, "B")  # Ancestors of B is A
        self.assertSetEqual(ancestors_b, {"A"})

        ancestors_d = get_all_ancestors(self.graph1, "D")  # Ancestors of D is A
        self.assertSetEqual(ancestors_d, {"A"})

        ancestors_a = get_all_ancestors(self.graph1, "A")  # A has no ancestors
        self.assertSetEqual(ancestors_a, set())

        # Non-existent node
        ancestors_non_existent = get_all_ancestors(self.graph1, "Z")
        self.assertSetEqual(ancestors_non_existent, set())

        # Empty graph
        ancestors_empty = get_all_ancestors(self.empty_graph, "A")
        self.assertSetEqual(ancestors_empty, set())

    def test_find_all_paths(self):
        # For self.graph1 (A -> B -> C, A -> D)
        paths_a_c = find_all_paths(self.graph1, "A", "C")
        self.assertListEqual(paths_a_c, [["A", "B", "C"]])

        paths_a_d = find_all_paths(self.graph1, "A", "D")
        self.assertListEqual(paths_a_d, [["A", "D"]])

        paths_a_b = find_all_paths(self.graph1, "A", "B")
        self.assertListEqual(paths_a_b, [["A", "B"]])

        paths_b_d = find_all_paths(self.graph1, "B", "D")  # No path
        self.assertListEqual(paths_b_d, [])

        paths_a_a = find_all_paths(self.graph1, "A", "A")  # Path to self
        self.assertListEqual(paths_a_a, [["A"]])

        # Diamond graph: S -> T1 -> E, S -> T2 -> E
        s, t1, t2, e = (Node("S", NodeType.COMPUTE), Node("T1", NodeType.COMPUTE, dependencies=["S"]), Node("T2", NodeType.COMPUTE, dependencies=["S"]), Node("E", NodeType.COMPUTE, dependencies=["T1", "T2"]))
        diamond_graph = TaskGraph("diamond")
        diamond_graph.add_nodes([s, t1, t2, e])
        paths_s_e = find_all_paths(diamond_graph, "S", "E")
        self.assertEqual(len(paths_s_e), 2)
        self.assertIn(["S", "T1", "E"], paths_s_e)
        self.assertIn(["S", "T2", "E"], paths_s_e)

        # Non-existent start/end
        self.assertListEqual(find_all_paths(self.graph1, "Z", "C"), [])
        self.assertListEqual(find_all_paths(self.graph1, "A", "Z"), [])


class TestTaskSplittingLogic(unittest.TestCase):
    """Tests for individual splitting strategy functions."""

    def assertGraphStructure(self, graph: TaskGraph, expected_node_ids: Set[str], expected_dependencies: Dict[str, List[str]], msg_prefix=""):
        self.assertSetEqual(set(graph.nodes.keys()), expected_node_ids, f"{msg_prefix} Node ID mismatch")
        graph.build_adjacency_lists()  # Ensure rev_adj is up-to-date
        for node_id, deps in expected_dependencies.items():
            self.assertIn(node_id, graph.rev_adj, f"{msg_prefix} Node {node_id} not in rev_adj")
            self.assertListEqual(sorted(graph.rev_adj[node_id]), sorted(deps), f"{msg_prefix} Dependencies for {node_id} mismatch")

    def test_split_single_structure_reconvergence(self):
        # Graph: A->B, F->C, then B,C -> D -> E (reconverge at D)
        a = Node("A", NodeType.DATA_LOAD)
        b = Node("B", NodeType.COMPUTE, dependencies=["A"])
        f = Node("F", NodeType.DATA_LOAD)
        c = Node("C", NodeType.COMPUTE, dependencies=["F"])
        d = Node("D", NodeType.COMPUTE, dependencies=["B", "C"])
        e_node = Node("E", NodeType.MODEL_TRAIN, dependencies=["D"])

        src_graph = TaskGraph("reconverge_src")
        src_graph.add_nodes([a, b, f, c, d, e_node])

        parallel_branches = [["A", "B"], ["F", "C"]]  # Node lists for branches up to merge point
        merge_node_id = "D"

        subgraphes = split_single_structure(src_graph, parallel_branches, merge_node_id, "base_idx")
        self.assertEqual(len(subgraphes), 2)

        # Expected subgraph 1: A->B->D->E
        # Expected subgraph 2: F->C->D->E
        subgraph1 = TaskGraph("exp1")
        subgraph1.add_nodes(
            [
                copy.deepcopy(a),
                copy.deepcopy(b),
                Node("D", NodeType.COMPUTE, dependencies=["B"]),  # D's deps adjusted
                Node("E", NodeType.MODEL_TRAIN, dependencies=["D"]),
            ]
        )
        subgraph2 = TaskGraph("exp2")
        subgraph2.add_nodes(
            [
                copy.deepcopy(f),
                copy.deepcopy(c),
                Node("D", NodeType.COMPUTE, dependencies=["C"]),  # D's deps adjusted
                Node("E", NodeType.MODEL_TRAIN, dependencies=["D"]),
            ]
        )
        expected_sig_1 = generate_structural_signature(subgraph1)
        expected_sig_2 = generate_structural_signature(subgraph2)

        actual_sigs = get_signatures(subgraphes)
        self.assertIn(expected_sig_1, actual_sigs)
        self.assertIn(expected_sig_2, actual_sigs)

    def test_split_by_fan_out_to_exits(self):
        # Graph: S -> A -> E1 (exit1)
        #        S -> B -> E2 (exit2)
        s = Node("S", NodeType.DATA_LOAD)
        a = Node("A", NodeType.COMPUTE, dependencies=["S"])
        e1 = Node("E1", NodeType.MODEL_TRAIN, dependencies=["A"])  # Exit 1
        b = Node("B", NodeType.COMPUTE, dependencies=["S"])
        e2 = Node("E2", NodeType.MODEL_TRAIN, dependencies=["B"])  # Exit 2

        src_graph = TaskGraph("fanout_src")
        src_graph.add_nodes([s, a, e1, b, e2])

        subgraphs = split_by_fan_out_to_exits(src_graph, 1)
        self.assertEqual(len(subgraphs), 2, "Should split into two subgraphs for distinct exits.")

        # Expected subgraph 1: S->A->E1
        # Expected subgraph 2: S->B->E2
        # Need to reconstruct expected TaskGraphs to get their signatures

        # Graph S->A->E1
        exp_g1 = TaskGraph("exp_g1")
        exp_g1.add_nodes([copy.deepcopy(s), Node("A", NodeType.COMPUTE, dependencies=["S"]), Node("E1", NodeType.MODEL_TRAIN, dependencies=["A"])])
        exp_g1_sig = generate_structural_signature(exp_g1)

        # Graph S->B->E2
        exp_g2 = TaskGraph("exp_g2")
        exp_g2.add_nodes([copy.deepcopy(s), Node("B", NodeType.COMPUTE, dependencies=["S"]), Node("E2", NodeType.MODEL_TRAIN, dependencies=["B"])])
        exp_g2_sig = generate_structural_signature(exp_g2)

        actual_sigs = get_signatures(subgraphs)
        self.assertIn(exp_g1_sig, actual_sigs)
        self.assertIn(exp_g2_sig, actual_sigs)

    def test_split_by_reconverging_paths_diamond(self):
        # Diamond: A -> B \
        #               -> D
        #          A -> C /
        a_ = Node("A", NodeType.DATA_LOAD)
        b_ = Node("B", NodeType.COMPUTE, dependencies=["A"])
        c_ = Node("C", NodeType.COMPUTE, dependencies=["A"])
        d_ = Node("D", NodeType.COMPUTE, dependencies=["B", "C"])
        src_graph = TaskGraph("diamond_src")
        src_graph.add_nodes([a_, b_, c_, d_])

        subgraphs = split_by_reconverging_paths(src_graph, 1)
        self.assertEqual(len(subgraphs), 2, "Diamond graph should split into two paths.")

        # Expected: A->B->D and A->C->D
        g_abd = TaskGraph("exp_abd")
        g_abd.add_nodes([copy.deepcopy(a_), Node("B", NodeType.COMPUTE, dependencies=["A"]), Node("D", NodeType.COMPUTE, dependencies=["B"])])
        g_acd = TaskGraph("exp_acd")
        g_acd.add_nodes([copy.deepcopy(a_), Node("C", NodeType.COMPUTE, dependencies=["A"]), Node("D", NodeType.COMPUTE, dependencies=["C"])])

        actual_sigs = get_signatures(subgraphs)
        self.assertIn(generate_structural_signature(g_abd), actual_sigs)
        self.assertIn(generate_structural_signature(g_acd), actual_sigs)


# Helper to extend TaskGraph for easier test setup
def add_nodes_and_build(self, nodes: List[Node]) -> TaskGraph:
    self.add_nodes(nodes)
    self.build_adjacency_lists()
    return self


TaskGraph.add_nodes_and_build = add_nodes_and_build


class TestDiscoverAndSplitParallelPaths(unittest.TestCase):
    """Comprehensive tests for the main discover_and_split_parallel_paths function."""

    def assertListOfGraphStructuresEqual(self, actual_graphs: List[TaskGraph], expected_graph_defs: List[Tuple[str, List[Node]]], msg=None):
        """
        Compares a list of actual TaskGraphs with a list of expected graph definitions.
        A graph definition is (name_for_debug, list_of_nodes_for_expected_graph).
        """
        self.assertEqual(len(actual_graphs), len(expected_graph_defs), f"{msg}: Different number of graphs. Got {len(actual_graphs)}, expected {len(expected_graph_defs)}")

        expected_signatures = []
        for name, nodes in expected_graph_defs:
            g = TaskGraph(name)
            g.add_nodes(nodes)
            expected_signatures.append(generate_structural_signature(g))

        actual_signatures = get_signatures(actual_graphs)
        self.assertListEqual(sorted(actual_signatures), sorted(expected_signatures), f"{msg}: Graph structures differ.")

    def test_empty_graph(self):
        empty_g = TaskGraph("empty")
        split_graphs = discover_and_split_parallel_paths(empty_g)
        self.assertEqual(len(split_graphs), 0, "Empty graph should result in empty list.")

    def test_linear_graph(self):
        # L1 -> L2 -> L3
        l1, l2, l3 = (Node("L1", NodeType.DATA_LOAD), Node("L2", NodeType.COMPUTE, dependencies=["L1"]), Node("L3", NodeType.MODEL_TRAIN, dependencies=["L2"]))
        linear_g = TaskGraph("linear")
        linear_g.add_nodes([l1, l2, l3])

        split_graphs = discover_and_split_parallel_paths(linear_g)

        self.assertListOfGraphStructuresEqual(
            split_graphs,
            [("linear_expected", [copy.deepcopy(n) for n in [l1, l2, l3]])],  # Deepcopy nodes for expected structure
        )

    def test_simple_reconvergence_diamond_graph(self):
        #   A
        #  / \
        # B   C
        #  \ /
        #   D
        a, b, c, d = (Node("A", NodeType.DATA_LOAD), Node("B", NodeType.COMPUTE, dependencies=["A"]), Node("C", NodeType.COMPUTE, dependencies=["A"]), Node("D", NodeType.COMPUTE, dependencies=["B", "C"]))
        diamond_g = TaskGraph("diamond")
        diamond_g.add_nodes([a, b, c, d])
        split_graphs = discover_and_split_parallel_paths(diamond_g)

        # Expected: (A->B->D) and (A->C->D)
        exp_nodes1 = [Node("A", NodeType.DATA_LOAD), Node("B", NodeType.COMPUTE, dependencies=["A"]), Node("D", NodeType.COMPUTE, dependencies=["B"])]
        exp_nodes2 = [Node("A", NodeType.DATA_LOAD), Node("C", NodeType.COMPUTE, dependencies=["A"]), Node("D", NodeType.COMPUTE, dependencies=["C"])]

        self.assertListOfGraphStructuresEqual(split_graphs, [("path_abd", exp_nodes1), ("path_acd", exp_nodes2)])

    def test_fan_out_only_graph(self):
        # A -> B (exit1)
        #   -> C (exit2)
        s_a, s_b_exit1, s_c_exit2 = Node("S_A", NodeType.DATA_LOAD), Node("S_B_exit1", NodeType.COMPUTE, dependencies=["S_A"]), Node("S_C_exit2", NodeType.COMPUTE, dependencies=["S_A"])
        fanout_g = TaskGraph("fanout_only")
        fanout_g.add_nodes([s_a, s_b_exit1, s_c_exit2])
        split_graphs = discover_and_split_parallel_paths(fanout_g)

        exp_nodes1 = [Node("S_A", NodeType.DATA_LOAD), Node("S_B_exit1", NodeType.COMPUTE, dependencies=["S_A"])]
        exp_nodes2 = [Node("S_A", NodeType.DATA_LOAD), Node("S_C_exit2", NodeType.COMPUTE, dependencies=["S_A"])]

        self.assertListOfGraphStructuresEqual(split_graphs, [("path_sa_sb", exp_nodes1), ("path_sa_sc", exp_nodes2)])

    def test_ex1_reconverge_from_prompt(self):
        # A -> B \
        #         -> C -> D_ex1 -> E_ex1
        # A1-> B1/
        node_a_orig = Node(node_id="A", node_type=NodeType.DATA_LOAD)
        node_b_orig = Node(node_id="B", node_type=NodeType.COMPUTE, dependencies=["A"])
        node_a1_orig = Node(node_id="A1", node_type=NodeType.DATA_LOAD)
        node_b1_orig = Node(node_id="B1", node_type=NodeType.COMPUTE, dependencies=["A1"])
        node_c_orig = Node(node_id="C", node_type=NodeType.COMPUTE, dependencies=["B", "B1"])
        node_d_ex1_orig = Node(node_id="D_ex1", node_type=NodeType.COMPUTE, dependencies=["C"])
        node_e_ex1_orig = Node(node_id="E_ex1", node_type=NodeType.MODEL_TRAIN, dependencies=["D_ex1"])

        original_graph_ex1 = TaskGraph(graph_id="ex1_reconverge")
        original_graph_ex1.add_nodes([node_a_orig, node_b_orig, node_a1_orig, node_b1_orig, node_c_orig, node_d_ex1_orig, node_e_ex1_orig])

        split_graphs = discover_and_split_parallel_paths(original_graph_ex1)

        # Expected path 1: A -> B -> C -> D_ex1 -> E_ex1
        exp1_nodes = [Node("A", NodeType.DATA_LOAD), Node("B", NodeType.COMPUTE, dependencies=["A"]), Node("C", NodeType.COMPUTE, dependencies=["B"]), Node("D_ex1", NodeType.COMPUTE, dependencies=["C"]), Node("E_ex1", NodeType.MODEL_TRAIN, dependencies=["D_ex1"])]
        # Expected path 2: A1 -> B1 -> C -> D_ex1 -> E_ex1
        exp2_nodes = [Node("A1", NodeType.DATA_LOAD), Node("B1", NodeType.COMPUTE, dependencies=["A1"]), Node("C", NodeType.COMPUTE, dependencies=["B1"]), Node("D_ex1", NodeType.COMPUTE, dependencies=["C"]), Node("E_ex1", NodeType.MODEL_TRAIN, dependencies=["D_ex1"])]
        self.assertListOfGraphStructuresEqual(split_graphs, [("path1_ex1", exp1_nodes), ("path2_ex1", exp2_nodes)], msg="TestEx1Reconverge")

    def test_ex2_complex_from_prompt(self):
        #      X -> P1 \
        #               -> M1 -> Z -> J1 -> K1 (exit1)
        #      Y -> P2 /        |
        #                       -> J2 -> K2 (exit2)
        #      P3 ---------------^ (P3 connects to Z)
        nx = Node("X", NodeType.DATA_LOAD)
        ny = Node("Y", NodeType.DATA_LOAD)
        np1 = Node("P1", NodeType.COMPUTE, dependencies=["X"])
        np2 = Node("P2", NodeType.COMPUTE, dependencies=["Y"])
        nm1 = Node("M1", NodeType.COMPUTE, dependencies=["P1", "P2"])
        np3 = Node("P3", NodeType.DATA_LOAD)
        nz = Node("Z", NodeType.COMPUTE, dependencies=["M1", "P3"])
        nj1 = Node("J1", NodeType.COMPUTE, dependencies=["Z"])
        nj2 = Node("J2", NodeType.COMPUTE, dependencies=["Z"])
        nk1 = Node("K1", NodeType.MODEL_TRAIN, dependencies=["J1"])  # Exit1
        nk2 = Node("K2", NodeType.MODEL_TRAIN, dependencies=["J2"])  # Exit2

        complex_graph = TaskGraph("ex2_complex")
        complex_graph.add_nodes([nx, ny, np1, np2, nm1, np3, nz, nj1, nj2, nk1, nk2])
        split_graphs = discover_and_split_parallel_paths(complex_graph)

        # Expected subgraphs (6 of them after full decomposition):
        # Path Group 1 (to K1):
        # 1. X->P1->M1->Z->J1->K1
        exp1_k1_nodes = [
            Node("X", NodeType.DATA_LOAD),
            Node("P1", NodeType.COMPUTE, dependencies=["X"]),
            Node("M1", NodeType.COMPUTE, dependencies=["P1"]),
            Node("Z", NodeType.COMPUTE, dependencies=["M1"]),
            Node("J1", NodeType.COMPUTE, dependencies=["Z"]),
            Node("K1", NodeType.MODEL_TRAIN, dependencies=["J1"]),
        ]
        # 2. Y->P2->M1->Z->J1->K1
        exp2_k1_nodes = [
            Node("Y", NodeType.DATA_LOAD),
            Node("P2", NodeType.COMPUTE, dependencies=["Y"]),
            Node("M1", NodeType.COMPUTE, dependencies=["P2"]),
            Node("Z", NodeType.COMPUTE, dependencies=["M1"]),
            Node("J1", NodeType.COMPUTE, dependencies=["Z"]),
            Node("K1", NodeType.MODEL_TRAIN, dependencies=["J1"]),
        ]
        # 3. P3->Z->J1->K1
        exp3_k1_nodes = [Node("P3", NodeType.DATA_LOAD), Node("Z", NodeType.COMPUTE, dependencies=["P3"]), Node("J1", NodeType.COMPUTE, dependencies=["Z"]), Node("K1", NodeType.MODEL_TRAIN, dependencies=["J1"])]

        # Path Group 2 (to K2):
        # 4. X->P1->M1->Z->J2->K2
        exp1_k2_nodes = [
            Node("X", NodeType.DATA_LOAD),
            Node("P1", NodeType.COMPUTE, dependencies=["X"]),
            Node("M1", NodeType.COMPUTE, dependencies=["P1"]),
            Node("Z", NodeType.COMPUTE, dependencies=["M1"]),
            Node("J2", NodeType.COMPUTE, dependencies=["Z"]),
            Node("K2", NodeType.MODEL_TRAIN, dependencies=["J2"]),
        ]
        # 5. Y->P2->M1->Z->J2->K2
        exp2_k2_nodes = [
            Node("Y", NodeType.DATA_LOAD),
            Node("P2", NodeType.COMPUTE, dependencies=["Y"]),
            Node("M1", NodeType.COMPUTE, dependencies=["P2"]),
            Node("Z", NodeType.COMPUTE, dependencies=["M1"]),
            Node("J2", NodeType.COMPUTE, dependencies=["Z"]),
            Node("K2", NodeType.MODEL_TRAIN, dependencies=["J2"]),
        ]
        # 6. P3->Z->J2->K2
        exp3_k2_nodes = [Node("P3", NodeType.DATA_LOAD), Node("Z", NodeType.COMPUTE, dependencies=["P3"]), Node("J2", NodeType.COMPUTE, dependencies=["Z"]), Node("K2", NodeType.MODEL_TRAIN, dependencies=["J2"])]

        self.assertListOfGraphStructuresEqual(split_graphs, [("xp1m1zj1k1", exp1_k1_nodes), ("yp2m1zj1k1", exp2_k1_nodes), ("p3zj1k1", exp3_k1_nodes), ("xp1m1zj2k2", exp1_k2_nodes), ("yp2m1zj2k2", exp2_k2_nodes), ("p3zj2k2", exp3_k2_nodes)], msg="TestEx2Complex")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=2)
