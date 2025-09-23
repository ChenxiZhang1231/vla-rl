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
import collections
from typing import List, Optional, Set
from siirl.scheduler import ProcessGroupManager
from siirl.workers.dag import NodeType, Node, NodeRole, TaskGraph


class TestProcessGroupManager(unittest.TestCase):
    def _create_task_graph(self, graph_id: str, node_specs: List[tuple[str, NodeType, NodeRole, Optional[List[str]]]]) -> TaskGraph:
        """
        Helper to create TaskGraph with actual Node objects, including dependencies.
        node_specs: List of (node_id, node_type, node_role, list_of_dependency_ids)
        """
        tg = TaskGraph(graph_id=graph_id)  #
        nodes_to_add = []
        for node_id_spec, node_type_spec, node_role_spec, deps_spec in node_specs:
            # Node class validation for NodeRole based on NodeType
            # For simplicity, ensure roles are compatible or use NodeRole.DEFAULT for non-model types.
            current_role = node_role_spec
            if node_type_spec not in [NodeType.COMPUTE, NodeType.MODEL_TRAIN, NodeType.MODEL_INFERENCE] and current_role != NodeRole.DEFAULT:  #
                current_role = NodeRole.DEFAULT  #

            nodes_to_add.append(Node(node_id=node_id_spec, node_type=node_type_spec, node_role=current_role, dependencies=deps_spec))  #
        if nodes_to_add:
            tg.add_nodes(nodes_to_add)  #
        # PGM doesn't rely on built adjacency lists, but it's good practice for a valid TG
        tg.build_adjacency_lists()  #
        return tg

    def test_initialization_default_relevant_types(self):
        pgm = ProcessGroupManager(total_num_workers=1, ranks_taskgraph_mapping={})  #
        self.assertEqual(pgm.relevant_node_types, {NodeType.MODEL_INFERENCE, NodeType.MODEL_TRAIN})  #

    def test_initialization_custom_relevant_types(self):
        custom_types = {NodeType.COMPUTE, NodeType.DATA_LOAD}  #
        pgm = ProcessGroupManager(total_num_workers=1, ranks_taskgraph_mapping={}, relevant_node_types_param=custom_types)  #
        self.assertEqual(pgm.relevant_node_types, custom_types)  #

    def test_initialization_invalid_custom_relevant_types(self):
        with self.assertRaises(ValueError):  #
            ProcessGroupManager(total_num_workers=1, ranks_taskgraph_mapping={}, relevant_node_types_param={"not_a_set"})  # type: ignore
        with self.assertRaises(ValueError):  #
            ProcessGroupManager(total_num_workers=1, ranks_taskgraph_mapping={}, relevant_node_types_param={NodeType.COMPUTE, "not_a_node_type"})  # type: ignore

    def test_default_filtering_model_nodes_only(self):
        """Tests that by default, only MODEL_INFERENCE and MODEL_TRAIN nodes are processed."""
        # N_train depends on N_compute, but N_compute is ignored.
        # N_infer also depends on N_compute.
        tg1 = self._create_task_graph(
            "TG1",
            [
                ("N_compute", NodeType.COMPUTE, NodeRole.DEFAULT, None),  # Irrelevant by default
                ("N_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["N_compute"]),  # Relevant
                ("N_infer", NodeType.MODEL_INFERENCE, NodeRole.DEFAULT, ["N_compute"]),  # Relevant
            ],
        )
        mapping = {0: tg1, 1: tg1}  # N_train, N_infer from TG1 on ranks 0, 1
        pgm = ProcessGroupManager(total_num_workers=2, ranks_taskgraph_mapping=mapping)  #

        # N_compute is ignored, so its dependencies don't affect PGM's view of N_train/N_infer.
        self.assertEqual(pgm.node_ranks_mapping, {"N_train": [0, 1], "N_infer": [0, 1]})  #
        self.assertNotIn("N_compute", pgm.node_ranks_mapping)  #
        self.assertEqual(pgm.process_group_spec, {"process_group_1": [0, 1]})  #
        self.assertEqual(
            pgm.node_process_group_mapping,  #
            {"N_train": "process_group_1", "N_infer": "process_group_1"},
        )

        expected_node_type_pg_map = {  #
            NodeType.MODEL_TRAIN.value: {"process_group_1"},  #
            NodeType.MODEL_INFERENCE.value: {"process_group_1"},  #
        }
        self.assertEqual(pgm.node_type_process_group_mapping, expected_node_type_pg_map)  #
        self.assertNotIn(NodeType.COMPUTE.value, pgm.node_type_process_group_mapping)  #

        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG1"][NodeType.MODEL_TRAIN.value], {"process_group_1"})  #
        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG1"][NodeType.MODEL_INFERENCE.value], {"process_group_1"})  #
        self.assertNotIn(NodeType.COMPUTE.value, pgm.subgraph_node_type_pg_mapping["TG1"])  #

        # Test getters
        self.assertEqual(pgm.get_process_groups_for_node_type(NodeType.MODEL_TRAIN.value), {"process_group_1"})  #
        self.assertEqual(pgm.get_process_groups_for_node_type(NodeType.COMPUTE.value), set())  # Not relevant
        self.assertEqual(pgm.get_process_group_for_node_type_in_subgraph("TG1", NodeType.MODEL_INFERENCE.value), {"process_group_1"})  #
        self.assertEqual(pgm.get_process_group_for_node_type_in_subgraph("TG1", NodeType.COMPUTE.value), set())  #

    def test_custom_filtering_compute_and_load_only(self):
        """Tests filtering with custom relevant types (e.g., COMPUTE and DATA_LOAD)."""
        relevant_types = {NodeType.COMPUTE, NodeType.DATA_LOAD}  #
        # N_compute depends on N_load. N_train depends on N_compute but is ignored.
        tg1 = self._create_task_graph(
            "TG1",
            [
                ("N_load", NodeType.DATA_LOAD, NodeRole.DEFAULT, None),  # Relevant
                ("N_compute", NodeType.COMPUTE, NodeRole.DEFAULT, ["N_load"]),  # Relevant
                ("N_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["N_compute"]),  # Ignored
            ],
        )
        mapping = {0: tg1, 1: tg1}  # N_load, N_compute from TG1 on ranks 0, 1
        pgm = ProcessGroupManager(total_num_workers=2, ranks_taskgraph_mapping=mapping, relevant_node_types_param=relevant_types)  #

        self.assertEqual(pgm.node_ranks_mapping, {"N_load": [0, 1], "N_compute": [0, 1]})  #
        self.assertNotIn("N_train", pgm.node_ranks_mapping)  #
        self.assertEqual(pgm.process_group_spec, {"process_group_1": [0, 1]})  #
        self.assertEqual(
            pgm.node_process_group_mapping,  #
            {"N_load": "process_group_1", "N_compute": "process_group_1"},
        )

        expected_node_type_pg_map = {  #
            NodeType.DATA_LOAD.value: {"process_group_1"},  #
            NodeType.COMPUTE.value: {"process_group_1"},  #
        }
        self.assertEqual(pgm.node_type_process_group_mapping, expected_node_type_pg_map)  #
        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG1"], expected_node_type_pg_map)  #

        # Test getters
        self.assertEqual(pgm.get_process_groups_for_node_type(NodeType.COMPUTE.value), {"process_group_1"})  #
        self.assertEqual(pgm.get_process_groups_for_node_type(NodeType.MODEL_TRAIN.value), set())  # Not relevant
        self.assertEqual(pgm.get_process_group_for_node_type_in_subgraph("TG1", NodeType.DATA_LOAD.value), {"process_group_1"})  #
        self.assertEqual(pgm.get_process_group_for_node_type_in_subgraph("TG1", NodeType.MODEL_TRAIN.value), set())  #

    def test_no_relevant_nodes_in_graph_custom_filter(self):
        """Tests behavior when a graph contains no nodes of the custom configured relevant types."""
        relevant_types = {NodeType.CUSTOM}  # Only interested in CUSTOM
        tg1 = self._create_task_graph(
            "TG1",
            [
                ("N_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, None),  #
                ("N_compute", NodeType.COMPUTE, NodeRole.DEFAULT, None),  #
            ],
        )  # No CUSTOM nodes
        mapping = {0: tg1}
        pgm = ProcessGroupManager(total_num_workers=1, ranks_taskgraph_mapping=mapping, relevant_node_types_param=relevant_types)  #

        self.assertEqual(pgm.node_ranks_mapping, {})  #
        self.assertEqual(pgm.process_group_spec, {})  #
        self.assertEqual(pgm.node_process_group_mapping, {})  #
        self.assertEqual(pgm.node_type_process_group_mapping, collections.defaultdict(set))  #
        self.assertEqual(pgm.subgraph_node_type_pg_mapping, collections.defaultdict(lambda: collections.defaultdict(set)))  #

    def test_empty_relevant_types_set_no_nodes_processed(self):
        """Tests behavior with an empty set of relevant_node_types (no nodes should be processed)."""
        relevant_types: Set[NodeType] = set()  # Empty set
        tg1 = self._create_task_graph(
            "TG1",
            [
                ("N_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, None),  #
                ("N_compute", NodeType.COMPUTE, NodeRole.DEFAULT, None),  #
            ],
        )
        mapping = {0: tg1}
        pgm = ProcessGroupManager(total_num_workers=1, ranks_taskgraph_mapping=mapping, relevant_node_types_param=relevant_types)  #

        self.assertEqual(pgm.node_ranks_mapping, {})  #
        self.assertEqual(pgm.process_group_spec, {})  #
        self.assertEqual(pgm.node_type_process_group_mapping, collections.defaultdict(set))  #
        self.assertEqual(pgm.subgraph_node_type_pg_mapping, collections.defaultdict(lambda: collections.defaultdict(set)))  #

    def test_complex_scenario_with_dependencies_and_default_filtering(self):
        """A more complex setup with dependencies using default filtering."""
        # TG1: M1_train depends on C1_comp (ignored). M1_infer also depends on C1_comp (ignored).
        #      So M1_train and M1_infer are effectively roots for PGM.
        tg1 = self._create_task_graph(
            "TG1_MODELS",
            [
                ("C1_comp", NodeType.COMPUTE, NodeRole.DEFAULT, None),  # Ignored
                ("M1_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["C1_comp"]),  # Relevant
                ("M1_infer", NodeType.MODEL_INFERENCE, NodeRole.DEFAULT, ["C1_comp"]),  # Relevant
            ],
        )
        # TG2: M2_train depends on D2_load (ignored). So M2_train is effectively a root.
        tg2 = self._create_task_graph(
            "TG2_MODELS",
            [
                ("D2_load", NodeType.DATA_LOAD, NodeRole.DEFAULT, None),  # Ignored
                ("M2_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["D2_load"]),  # Relevant
            ],
        )
        # TG3: All nodes ignored. C3_comp depends on D3_load.
        tg3 = self._create_task_graph(
            "TG3_OTHER",
            [
                ("D3_load", NodeType.DATA_LOAD, NodeRole.DEFAULT, None),  # Ignored
                ("C3_comp", NodeType.COMPUTE, NodeRole.DEFAULT, ["D3_load"]),  # Ignored
            ],
        )
        # TG4: Relevant M4_train depends on irrelevant C4_comp which depends on relevant M4_infer.
        # PGM will see M4_train and M4_infer.
        tg4 = self._create_task_graph(
            "TG4_MIXED_DEP",
            [
                ("M4_infer", NodeType.MODEL_INFERENCE, NodeRole.DEFAULT, None),  # Relevant
                ("C4_comp", NodeType.COMPUTE, NodeRole.DEFAULT, ["M4_infer"]),  # Ignored
                ("M4_train", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["C4_comp"]),  # Relevant
            ],
        )

        mapping = {
            0: tg1,
            1: tg1,  # TG1: M1_train, M1_infer on ranks [0,1] -> PG1
            2: tg2,
            3: tg2,  # TG2: M2_train on ranks [2,3] -> PG2
            4: tg3,  # TG3: No relevant nodes.
            5: tg4,
            6: tg4,  # TG4: M4_infer, M4_train on ranks [5,6] -> PG3
        }
        pgm = ProcessGroupManager(total_num_workers=7, ranks_taskgraph_mapping=mapping)  #

        self.assertEqual(
            pgm.node_ranks_mapping,
            {  #
                "M1_train": [0, 1],
                "M1_infer": [0, 1],
                "M2_train": [2, 3],
                "M4_infer": [5, 6],
                "M4_train": [5, 6],
            },
        )
        # Ensure ignored nodes are not present
        for ignored_node_id in ["C1_comp", "D2_load", "D3_load", "C3_comp", "C4_comp"]:
            self.assertNotIn(ignored_node_id, pgm.node_ranks_mapping)  #

        # Rank tuples: (0,1), (2,3), (5,6). Sorted: (0,1), (2,3), (5,6).
        self.assertEqual(
            pgm.process_group_spec,
            {  #
                "process_group_1": [0, 1],
                "process_group_2": [2, 3],
                "process_group_3": [5, 6],
            },
        )

        self.assertEqual(
            pgm.node_process_group_mapping,
            {  #
                "M1_train": "process_group_1",
                "M1_infer": "process_group_1",
                "M2_train": "process_group_2",
                "M4_infer": "process_group_3",
                "M4_train": "process_group_3",
            },
        )

        expected_node_type_pg_map = {  #
            NodeType.MODEL_TRAIN.value: {"process_group_1", "process_group_2", "process_group_3"},  #
            NodeType.MODEL_INFERENCE.value: {"process_group_1", "process_group_3"},  #
        }
        self.assertEqual(pgm.node_type_process_group_mapping, expected_node_type_pg_map)  #

        # Check subgraph mappings
        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG1_MODELS"][NodeType.MODEL_TRAIN.value], {"process_group_1"})  #
        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG1_MODELS"][NodeType.MODEL_INFERENCE.value], {"process_group_1"})  #
        self.assertNotIn(NodeType.COMPUTE.value, pgm.subgraph_node_type_pg_mapping["TG1_MODELS"])  #

        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG2_MODELS"][NodeType.MODEL_TRAIN.value], {"process_group_2"})  #

        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG4_MIXED_DEP"][NodeType.MODEL_INFERENCE.value], {"process_group_3"})  #
        self.assertEqual(pgm.subgraph_node_type_pg_mapping["TG4_MIXED_DEP"][NodeType.MODEL_TRAIN.value], {"process_group_3"})  #

        self.assertNotIn("TG3_OTHER", pgm.subgraph_node_type_pg_mapping)  # No relevant nodes

    def test_highly_complex_scenario_shared_nodes_16_workers(self):
        """
        Test: A more complex scenario with 16 workers, multiple TaskGraphs,
        nodes with shared IDs across these TaskGraphs, and intricate rank overlaps.
        Uses default relevant types (MODEL_TRAIN, MODEL_INFERENCE).
        """
        # Define Node Specs: (id, type, role, dependencies)

        # --- Shared Nodes ---
        # S_Train_1 will appear in TG_Alpha and TG_Beta
        s_train_1_spec = ("S_Train_1", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, None)
        # S_Infer_1 will appear in TG_Beta and TG_Gamma
        s_infer_1_spec = ("S_Infer_1", NodeType.MODEL_INFERENCE, NodeRole.DEFAULT, None)
        # S_Train_2 will appear in TG_Gamma and TG_Delta
        s_train_2_spec = ("S_Train_2", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, None)

        # --- TaskGraph Alpha Specific Nodes ---
        tga_mt_1_spec = ("TGA_MT_1", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["S_Train_1"])
        tga_c1_spec = ("TGA_C1", NodeType.COMPUTE, NodeRole.DEFAULT, ["TGA_MT_1"])  # Irrelevant

        # --- TaskGraph Beta Specific Nodes ---
        tgb_mi_1_spec = ("TGB_MI_1", NodeType.MODEL_INFERENCE, NodeRole.DEFAULT, ["S_Train_1"])
        tgb_c1_spec = ("TGB_C1", NodeType.COMPUTE, NodeRole.DEFAULT, ["S_Infer_1"])  # Irrelevant

        # --- TaskGraph Gamma Specific Nodes ---
        tgc_mt_1_spec = ("TGC_MT_1", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, ["S_Infer_1"])
        tgc_dl1_spec = ("TGC_DL1", NodeType.DATA_LOAD, NodeRole.DEFAULT, ["S_Train_2"])  # Irrelevant

        # --- TaskGraph Delta Specific Nodes ---
        tgd_mi_1_spec = ("TGD_MI_1", NodeType.MODEL_INFERENCE, NodeRole.DEFAULT, ["S_Train_2"])
        tgd_c1_spec = ("TGD_C1", NodeType.COMPUTE, NodeRole.DEFAULT, ["TGD_MI_1"])  # Irrelevant

        # --- TaskGraph Epsilon Nodes (all irrelevant) ---
        tge_c1_spec = ("TGE_C1", NodeType.COMPUTE, NodeRole.DEFAULT, None)
        tge_dl1_spec = ("TGE_DL1", NodeType.DATA_LOAD, NodeRole.DEFAULT, ["TGE_C1"])

        # --- TaskGraph Zeta Node (isolated relevant node) ---
        tgz_mt_1_spec = ("TGZ_MT_1", NodeType.MODEL_TRAIN, NodeRole.DEFAULT, None)

        # Create TaskGraph instances
        tg_alpha = self._create_task_graph("TG_Alpha", [s_train_1_spec, tga_mt_1_spec, tga_c1_spec])
        tg_beta = self._create_task_graph("TG_Beta", [s_train_1_spec, s_infer_1_spec, tgb_mi_1_spec, tgb_c1_spec])
        tg_gamma = self._create_task_graph("TG_Gamma", [s_infer_1_spec, s_train_2_spec, tgc_mt_1_spec, tgc_dl1_spec])
        tg_delta = self._create_task_graph("TG_Delta", [s_train_2_spec, tgd_mi_1_spec, tgd_c1_spec])
        tg_epsilon = self._create_task_graph("TG_Epsilon", [tge_c1_spec, tge_dl1_spec])
        tg_zeta = self._create_task_graph("TG_Zeta", [tgz_mt_1_spec])

        # Define rank assignments for each TaskGraph for PGM input
        # PGM's `_collect_initial_topology_info` iterates `ranks_taskgraph_mapping.items()`.
        # It builds `graph_id_to_ranks`. For a node_id present in multiple GIDs,
        # `_aggregate_ranks_for_nodes` takes the union of `graph_id_to_ranks` for those GIDs.
        final_mapping_for_pgm = {}
        for r_val in range(16):
            final_mapping_for_pgm[r_val] = None  # Initialize

        # Assign ranks to TaskGraphs. If a rank is assigned multiple TGs, the last one wins for that rank.
        # PGM's graph_id_to_ranks will reflect the set of ranks ultimately pointing to each GID.
        # Ranks for TG_Alpha
        for r_idx in {0, 1}:
            final_mapping_for_pgm[r_idx] = tg_alpha
        # Ranks for TG_Beta (rank 2 will be tg_beta, rank 4 tg_gamma, rank 6 tg_delta)
        for r_idx in {2, 3, 10}:
            final_mapping_for_pgm[r_idx] = tg_beta
        # Ranks for TG_Gamma
        for r_idx in {4, 5, 11}:
            final_mapping_for_pgm[r_idx] = tg_gamma
        # Ranks for TG_Delta
        for r_idx in {6, 7, 12, 13}:
            final_mapping_for_pgm[r_idx] = tg_delta
        # Ranks for TG_Epsilon (no relevant nodes)
        for r_idx in {8, 9}:
            final_mapping_for_pgm[r_idx] = tg_epsilon
        # Ranks for TG_Zeta
        for r_idx in {14, 15}:
            final_mapping_for_pgm[r_idx] = tg_zeta

        # Update tg_alpha's mapping for rank 2 if it wasn't overwritten
        # To achieve the intended rank aggregation, the final_mapping_for_pgm should be:
        # TG_Alpha assigned to {0,1}
        # TG_Beta assigned to {2,3,10}
        # TG_Gamma assigned to {4,5,11}
        # TG_Delta assigned to {6,7,12,13}
        # TG_Epsilon assigned to {8,9}
        # TG_Zeta assigned to {14,15}
        #
        # And Node_Shared_Train is in TG_Alpha (ID: S_Train_1) and TG_Beta (ID: S_Train_1)
        # Node_Shared_Infer is in TG_Beta (ID: S_Infer_1) and TG_Gamma (ID: S_Infer_1)
        # Node_Shared_Train_2 is in TG_Gamma (ID: S_Train_2) and TG_Delta (ID: S_Train_2)
        # This setup correctly tests the union of ranks for shared node_ids.

        pgm = ProcessGroupManager(total_num_workers=16, ranks_taskgraph_mapping=final_mapping_for_pgm)

        # Expected Node Ranks based on `final_mapping_for_pgm` and node presence in TGs:
        # S_Train_1: in TG_Alpha (ranks {0,1}), in TG_Beta (ranks {2,3,10}) => {0,1,2,3,10}
        # TGA_MT_1: in TG_Alpha (ranks {0,1}) => {0,1}
        # S_Infer_1: in TG_Beta (ranks {2,3,10}), in TG_Gamma (ranks {4,5,11}) => {2,3,4,5,10,11}
        # TGB_MI_1: in TG_Beta (ranks {2,3,10}) => {2,3,10}
        # S_Train_2: in TG_Gamma (ranks {4,5,11}), in TG_Delta (ranks {6,7,12,13}) => {4,5,6,7,11,12,13}
        # TGC_MT_1: in TG_Gamma (ranks {4,5,11}) => {4,5,11}
        # TGD_MI_1: in TG_Delta (ranks {6,7,12,13}) => {6,7,12,13}
        # TGZ_MT_1: in TG_Zeta (on {14,15}) => {14,15}
        self.assertEqual(
            pgm.node_ranks_mapping,
            {"S_Train_1": sorted([0, 1, 2, 3, 10]), "TGA_MT_1": sorted([0, 1]), "S_Infer_1": sorted([2, 3, 4, 5, 10, 11]), "TGB_MI_1": sorted([2, 3, 10]), "S_Train_2": sorted([4, 5, 6, 7, 11, 12, 13]), "TGC_MT_1": sorted([4, 5, 11]), "TGD_MI_1": sorted([6, 7, 12, 13]), "TGZ_MT_1": sorted([14, 15])},
        )

        # Corrected expected process_group_spec based on lexicographical sort of unique rank tuples
        # Unique rank tuples:
        # (0,1)
        # (0,1,2,3,10)
        # (2,3,4,5,10,11)
        # (2,3,10)
        # (4,5,6,7,11,12,13)
        # (4,5,11)
        # (6,7,12,13)
        # (14,15)
        # Sorted list of these tuples gives PG names:
        # PG1: (0,1)
        # PG2: (0,1,2,3,10)
        # PG3: (2,3,4,5,10,11)  <-- Corrected from previous manual sort error
        # PG4: (2,3,10)         <-- Corrected
        # PG5: (4,5,6,7,11,12,13) <-- Corrected
        # PG6: (4,5,11)         <-- Corrected
        # PG7: (6,7,12,13)
        # PG8: (14,15)
        self.assertEqual(
            pgm.process_group_spec,
            {
                "process_group_1": sorted([0, 1]),
                "process_group_2": sorted([0, 1, 2, 3, 10]),
                "process_group_3": sorted([2, 3, 4, 5, 10, 11]),  # Corresponds to S_Infer_1
                "process_group_4": sorted([2, 3, 10]),  # Corresponds to TGB_MI_1
                "process_group_5": sorted([4, 5, 6, 7, 11, 12, 13]),  # Corresponds to S_Train_2
                "process_group_6": sorted([4, 5, 11]),  # Corresponds to TGC_MT_1
                "process_group_7": sorted([6, 7, 12, 13]),  # Corresponds to TGD_MI_1
                "process_group_8": sorted([14, 15]),  # Corresponds to TGZ_MT_1
            },
        )

        # Corrected expected node_process_group_mapping
        self.assertEqual(
            pgm.node_process_group_mapping,
            {
                "TGA_MT_1": "process_group_1",  # Ranks (0,1)
                "S_Train_1": "process_group_2",  # Ranks (0,1,2,3,10)
                "S_Infer_1": "process_group_3",  # Ranks (2,3,4,5,10,11)
                "TGB_MI_1": "process_group_4",  # Ranks (2,3,10)
                "S_Train_2": "process_group_5",  # Ranks (4,5,6,7,11,12,13)
                "TGC_MT_1": "process_group_6",  # Ranks (4,5,11)
                "TGD_MI_1": "process_group_7",  # Ranks (6,7,12,13)
                "TGZ_MT_1": "process_group_8",  # Ranks (14,15)
            },
        )

        # Corrected expected node_type_process_group_mapping
        # MODEL_TRAIN: TGA_MT_1(PG1), S_Train_1(PG2), S_Train_2(PG5), TGC_MT_1(PG6), TGZ_MT_1(PG8)
        # MODEL_INFERENCE: S_Infer_1(PG3), TGB_MI_1(PG4), TGD_MI_1(PG7)
        expected_type_map = {NodeType.MODEL_TRAIN.value: {"process_group_1", "process_group_2", "process_group_5", "process_group_6", "process_group_8"}, NodeType.MODEL_INFERENCE.value: {"process_group_3", "process_group_4", "process_group_7"}}
        self.assertEqual(pgm.node_type_process_group_mapping, expected_type_map)

        # Corrected expected subgraph_node_type_pg_mapping
        self.assertEqual(
            pgm.subgraph_node_type_pg_mapping["TG_Alpha"],
            {
                NodeType.MODEL_TRAIN.value: {"process_group_1", "process_group_2"}  # TGA_MT_1 (PG1), S_Train_1 (PG2)
            },
        )
        self.assertEqual(
            pgm.subgraph_node_type_pg_mapping["TG_Beta"],
            {
                NodeType.MODEL_TRAIN.value: {"process_group_2"},  # S_Train_1 (PG2)
                NodeType.MODEL_INFERENCE.value: {"process_group_3", "process_group_4"},  # S_Infer_1 (PG3), TGB_MI_1 (PG4)
            },
        )
        self.assertEqual(
            pgm.subgraph_node_type_pg_mapping["TG_Gamma"],
            {
                NodeType.MODEL_INFERENCE.value: {"process_group_3"},  # S_Infer_1 (PG3)
                NodeType.MODEL_TRAIN.value: {"process_group_5", "process_group_6"},  # S_Train_2 (PG5), TGC_MT_1 (PG6)
            },
        )
        self.assertEqual(
            pgm.subgraph_node_type_pg_mapping["TG_Delta"],
            {
                NodeType.MODEL_TRAIN.value: {"process_group_5"},  # S_Train_2 (PG5)
                NodeType.MODEL_INFERENCE.value: {"process_group_7"},  # TGD_MI_1 (PG7)
            },
        )
        self.assertNotIn("TG_Epsilon", pgm.subgraph_node_type_pg_mapping)
        self.assertEqual(
            pgm.subgraph_node_type_pg_mapping["TG_Zeta"],
            {
                NodeType.MODEL_TRAIN.value: {"process_group_8"}  # TGZ_MT_1 (PG8)
            },
        )

        # Verify getters for some specific cases
        self.assertEqual(pgm.get_node_assignment("S_Train_1"), {"ranks": sorted([0, 1, 2, 3, 10]), "process_group_name": "process_group_2"})
        self.assertEqual(pgm.get_process_groups_for_node_type(NodeType.MODEL_TRAIN.value), {"process_group_1", "process_group_2", "process_group_5", "process_group_6", "process_group_8"})
        self.assertEqual(pgm.get_process_group_for_node_type_in_subgraph("TG_Beta", NodeType.MODEL_INFERENCE.value), {"process_group_3", "process_group_4"})
        self.assertEqual(pgm.get_process_group_for_node_type_in_subgraph("TG_Epsilon", NodeType.COMPUTE.value), set())


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=2)
