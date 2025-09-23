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

from siirl.workers.dag import Node, NodeRole, NodeStatus, NodeType, TaskGraph


def example_data_load_func():
    pass


def example_compute_func():
    pass


class TestTaskGraph(unittest.TestCase):
    """Unit tests for the TaskGraph class"""

    def setUp(self):
        self.graph = TaskGraph(graph_id="test_graph")
        self.node_a = Node(node_id="A", node_type=NodeType.DATA_LOAD)
        self.node_b = Node(node_id="B", node_type=NodeType.COMPUTE, node_role=NodeRole.ROLLOUT, dependencies=["A"])
        self.node_c = Node(node_id="C", node_type=NodeType.COMPUTE, node_role=NodeRole.REFERENCE, dependencies=["A"])
        self.node_d = Node(node_id="D", node_type=NodeType.BARRIER_SYNC, dependencies=["B", "C"])
        self.node_e = Node(node_id="E", node_type=NodeType.MODEL_TRAIN, node_role=NodeRole.ACTOR, dependencies=["D"])

        self.dag_module_sync_ref = f"{__name__}.example_data_load_func"
        self.dag_module_async_ref = f"{__name__}.example_compute_func"

    def test_graph_creation(self):
        self.assertEqual(self.graph.graph_id, "test_graph")
        self.assertEqual(self.graph.nodes, {})

    def test_add_node(self):
        self.graph.add_node(self.node_a)
        self.assertIn("A", self.graph.nodes)
        self.assertEqual(self.graph.nodes["A"], self.node_a)
        new_node_a = Node(node_id="A", node_type=NodeType.CUSTOM)
        self.graph.add_node(new_node_a)
        self.assertEqual(self.graph.nodes["A"].node_type, NodeType.CUSTOM)
        with self.assertRaisesRegex(ValueError, "Only Node type objects can be added to the graph"):
            self.graph.add_node("not_a_node")  # type: ignore

    def test_build_adjacency_lists(self):
        self.graph.add_node(self.node_a)
        self.graph.add_node(self.node_b)
        self.graph.add_node(self.node_c)
        self.graph.add_node(self.node_d)
        self.graph.build_adjacency_lists()
        self.assertEqual(set(self.graph.adj.get("A", [])), {"B", "C"})
        self.assertEqual(set(self.graph.adj.get("B", [])), {"D"})
        self.assertEqual(set(self.graph.adj.get("C", [])), {"D"})
        self.assertEqual(self.graph.adj.get("D", []), [])
        self.assertEqual(set(self.graph.rev_adj.get("A", [])), set())
        self.assertEqual(set(self.graph.rev_adj.get("B", [])), {"A"})
        self.assertEqual(set(self.graph.rev_adj.get("C", [])), {"A"})
        self.assertEqual(set(self.graph.rev_adj.get("D", [])), {"B", "C"})
        graph2 = TaskGraph("graph2")
        node_x = Node("X", NodeType.COMPUTE)
        node_y = Node("Y", NodeType.COMPUTE, dependencies=["X"])
        graph2.add_node(node_x)
        graph2.add_node(node_y)
        self.assertEqual(set(graph2.adj.get("X", [])), {"Y"})  # Relies on _update_adj_for_node
        self.assertEqual(set(graph2.rev_adj.get("Y", [])), {"X"})  # Relies on _update_adj_for_node

    def test_get_node(self):
        self.graph.add_node(self.node_a)
        self.assertEqual(self.graph.get_node("A"), self.node_a)
        self.assertIsNone(self.graph.get_node("X"))

    def test_get_dependencies_and_dependents(self):
        self.graph.add_node(self.node_a)
        self.graph.add_node(self.node_b)
        self.graph.add_node(self.node_c)
        self.graph.build_adjacency_lists()
        deps_b = [n.node_id for n in self.graph.get_dependencies("B")]
        self.assertEqual(deps_b, ["A"])
        dependents_a = sorted([n.node_id for n in self.graph.get_dependents("A")])
        self.assertEqual(dependents_a, ["B", "C"])
        self.assertEqual(self.graph.get_dependencies("A"), [])
        self.assertEqual(self.graph.get_dependents("C"), [])

    def test_get_entry_and_exit_nodes(self):
        self.graph.add_node(self.node_a)
        self.graph.add_node(self.node_b)
        self.graph.add_node(self.node_c)
        self.graph.add_node(self.node_d)
        self.graph.add_node(self.node_e)
        self.graph.build_adjacency_lists()
        entry_nodes = sorted([n.node_id for n in self.graph.get_entry_nodes()])
        self.assertEqual(entry_nodes, ["A"])
        exit_nodes = sorted([n.node_id for n in self.graph.get_exit_nodes()])
        self.assertEqual(exit_nodes, ["E"])

    def test_validate_graph_valid(self):
        self.graph.add_node(self.node_a)
        self.graph.add_node(self.node_b)
        self.graph.build_adjacency_lists()
        is_valid, msg = self.graph.validate_graph()
        self.assertTrue(is_valid)
        self.assertIsNone(msg)

    def test_validate_graph_missing_dependency(self):
        node_x = Node("X", NodeType.COMPUTE, dependencies=["Y_non_existent"])
        self.graph.add_node(node_x)
        self.graph.build_adjacency_lists()
        is_valid, msg = self.graph.validate_graph()
        self.assertFalse(is_valid)
        self.assertIn("The dependency 'Y_non_existent' of node 'X' does not exist in the graph", msg)

    def test_validate_graph_cyclic(self):
        node_x = Node("X_cyclic", NodeType.COMPUTE, dependencies=["Y_cyclic"])
        node_y = Node("Y_cyclic", NodeType.COMPUTE, dependencies=["X_cyclic"])
        self.graph.add_node(node_x)
        self.graph.add_node(node_y)
        self.graph.build_adjacency_lists()
        is_valid, msg = self.graph.validate_graph()
        self.assertFalse(is_valid)
        self.assertIn("There are circular dependencies in", msg)

    def test_get_topological_sort_valid(self):
        self.graph.add_node(self.node_a)
        self.graph.add_node(self.node_b)
        self.graph.add_node(self.node_c)
        self.graph.add_node(self.node_d)
        self.graph.build_adjacency_lists()
        order = self.graph.get_topological_sort()
        self.assertEqual(len(order), 4)
        self.assertEqual(set(order), {"A", "B", "C", "D"})
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("A"), order.index("C"))
        if "B" in order and "D" in order:
            self.assertLess(order.index("B"), order.index("D"))
        if "C" in order and "D" in order:
            self.assertLess(order.index("C"), order.index("D"))

    def test_get_topological_sort_empty_graph(self):
        self.assertEqual(self.graph.get_topological_sort(), [])

    def test_get_topological_sort_cyclic(self):
        node_x = Node("X_cyclic_topo", NodeType.COMPUTE, dependencies=["Y_cyclic_topo"])
        node_y = Node("Y_cyclic_topo", NodeType.COMPUTE, dependencies=["X_cyclic_topo"])
        self.graph.add_node(node_x)
        self.graph.add_node(node_y)
        self.graph.build_adjacency_lists()
        with self.assertRaisesRegex(ValueError, "There are circular dependencies"):
            self.graph.get_topological_sort()

    def test_reset_nodes_status(self):
        self.node_a.update_status(NodeStatus.COMPLETED, "Done A")
        self.node_a.output = "Output A"
        self.node_b.update_status(NodeStatus.FAILED, "Failed B")
        self.node_b.retries_done = 1
        self.graph.add_node(self.node_a)
        self.graph.add_node(self.node_b)
        self.graph.reset_nodes_status()
        self.assertEqual(self.node_a.status, NodeStatus.PENDING)
        self.assertIsNone(self.node_a.output)
        self.assertIsNone(self.node_a.error_info)
        self.assertEqual(self.node_a.retries_done, 0)
        self.assertEqual(self.node_b.status, NodeStatus.PENDING)
        self.assertIsNone(self.node_b.output)
        self.assertIsNone(self.node_b.error_info)
        self.assertEqual(self.node_b.retries_done, 0)

    def test_load_from_config_valid(self):
        graph_config = [
            {"node_id": "cfg_A", "node_type": "DATA_LOAD", "executable_ref": self.dag_module_sync_ref, "config": {"path": "dummy.csv"}},
            {"node_id": "cfg_B", "node_type": "COMPUTE", "dependencies": ["cfg_A"], "executable_ref": self.dag_module_async_ref, "config": {"operation": "sum"}, "node_role": "ACTOR", "retry_limit": 1},
            {"node_id": "cfg_C", "node_type": "BARRIER_SYNC", "dependencies": ["cfg_B"]},
        ]
        graph = TaskGraph.load_from_config("config_graph_1", graph_config)
        self.assertEqual(graph.graph_id, "config_graph_1")
        self.assertEqual(len(graph.nodes), 3)
        self.assertIn("cfg_A", graph.nodes)
        self.assertEqual(graph.nodes["cfg_A"].node_type, NodeType.DATA_LOAD)
        self.assertEqual(graph.nodes["cfg_B"].node_role, NodeRole.ACTOR)
        self.assertEqual(graph.nodes["cfg_B"].retry_limit, 1)
        self.assertTrue(callable(graph.nodes["cfg_A"].executable))
        order = graph.get_topological_sort()
        self.assertEqual(order, ["cfg_A", "cfg_B", "cfg_C"])

    def test_load_from_config_missing_field(self):
        graph_config_no_id = [{"node_type": "DATA_LOAD"}]
        with self.assertRaisesRegex(ValueError, ".*missing required field: 'node_id'.*"):
            TaskGraph.load_from_config("bad_cfg_no_id", graph_config_no_id)
        graph_config_no_type = [{"node_id": "X_no_type"}]
        with self.assertRaisesRegex(ValueError, ".*missing 'node_type'.*"):
            TaskGraph.load_from_config("bad_cfg_no_type", graph_config_no_type)

    def test_load_from_config_invalid_enum_value(self):
        graph_config_invalid_type = [{"node_id": "X_invalid_type", "node_type": "INVALID_NODE_TYPE_VALUE"}]
        with self.assertRaisesRegex(ValueError, ".*INVALID_NODE_TYPE_VALUE.*"):
            TaskGraph.load_from_config("bad_cfg_invalid_type", graph_config_invalid_type)
        graph_config_invalid_role = [{"node_id": "Y_invalid_role", "node_type": "COMPUTE", "node_role": "INVALID_ROLE_VALUE"}]
        with self.assertRaisesRegex(ValueError, ".*INVALID_ROLE_VALUE.*"):
            TaskGraph.load_from_config("bad_cfg_invalid_role", graph_config_invalid_role)

    def test_load_from_config_invalid_graph_structure(self):
        graph_config_cyclic = [
            {"node_id": "X_cfg_cyclic", "node_type": "COMPUTE", "dependencies": ["Y_cfg_cyclic"]},
            {"node_id": "Y_cfg_cyclic", "node_type": "COMPUTE", "dependencies": ["X_cfg_cyclic"]},
        ]
        with self.assertRaisesRegex(ValueError, ".*configuration is invalid:.*There are circular dependencies.*"):
            TaskGraph.load_from_config("cyclic_cfg_from_config", graph_config_cyclic)


if __name__ == "__main__":
    unittest.main(verbosity=2)
