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

# test_task_scheduler.py
import unittest
import collections
from typing import List, Dict, Optional
from siirl.workers.dag.node import Node, NodeType
from siirl.workers.dag.task_graph import TaskGraph
from siirl.scheduler.task_scheduler import TaskScheduler


# Helper function to create dummy TaskGraph objects for testing
def create_test_graph(graph_id: str, num_nodes: int, model_params: any = 0.0, dependencies_map: Optional[Dict[int, List[int]]] = None) -> TaskGraph:
    """
    Creates a TaskGraph for testing purposes.
    Args:
        graph_id (str): The ID for the graph.
        num_nodes (int): Number of nodes to create in the graph.
        model_params (any): Model parameters for the first node (if applicable). Can be float or string like "10B".
        dependencies_map (Optional[Dict[int, List[int]]]): A map where key is node index and value is a list of dependency indices.
                                                            Example: {1: [0]} means node 1 depends on node 0.
    Returns:
        TaskGraph: The created TaskGraph object.
    """
    graph = TaskGraph(graph_id=graph_id)
    nodes_to_add = []
    for i in range(num_nodes):
        node_type_val = NodeType.COMPUTE
        node_config_val = {}
        current_node_id = f"{graph_id}_n{i}"
        node_deps_ids: List[str] = []

        if dependencies_map and i in dependencies_map:
            for dep_idx in dependencies_map[i]:
                if 0 <= dep_idx < i:  # Ensure dependencies are on already defined (or to be defined earlier) nodes
                    node_deps_ids.append(f"{graph_id}_n{dep_idx}")
                else:
                    # This case should ideally be handled by graph validation, but good to be aware
                    pass  # Or raise an error for invalid dependency definition in test setup

        # Assign model_params to the first node if provided, to make it a MODEL_TRAIN type
        if model_params != 0.0 and i == 0:
            node_type_val = NodeType.MODEL_TRAIN
            node_config_val = {"model_params": model_params}

        nodes_to_add.append(Node(node_id=current_node_id, node_type=node_type_val, config=node_config_val, dependencies=node_deps_ids))

    if nodes_to_add:
        graph.add_nodes(nodes_to_add)
        graph.build_adjacency_lists()  # Crucial for graph operations and validation
        is_valid, msg = graph.validate_graph()
        if not is_valid:
            # This helps catch issues in test graph creation itself
            raise ValueError(f"Test helper created an invalid graph '{graph_id}': {msg}")
    return graph


class TestTaskScheduler(unittest.TestCase):
    """
    Unit tests for the TaskScheduler class.
    """

    def setUp(self):
        """
        Set up common resources for tests.
        This method is called before each test function.
        """
        # Default scheduler configuration for many tests
        self.num_nodes_default = 2
        self.gpus_per_node_default = 2
        self.scheduler_default = TaskScheduler(num_physical_nodes=self.num_nodes_default, gpus_per_node=self.gpus_per_node_default)
        # Total workers = 2 * 2 = 4

    def test_scheduler_initialization(self):
        """
        Test the initialization of the TaskScheduler.
        """
        scheduler = TaskScheduler(num_physical_nodes=3, gpus_per_node=4)
        self.assertEqual(scheduler.num_physical_nodes, 3)
        self.assertEqual(scheduler.gpus_per_node, 4)
        self.assertEqual(scheduler.num_workers, 12)  # 3 nodes * 4 GPUs/node
        self.assertEqual(len(scheduler.worker_to_graph_assignment), 12)
        self.assertTrue(all(assignment is None for assignment in scheduler.worker_to_graph_assignment.values()))

        # Test initialization with invalid parameters
        with self.assertRaises(ValueError):
            TaskScheduler(num_physical_nodes=0, gpus_per_node=2)
        with self.assertRaises(ValueError):
            TaskScheduler(num_physical_nodes=2, gpus_per_node=0)

    def test_reset_scheduler_state(self):
        """
        Test the _reset_scheduler_state method.
        """
        scheduler = self.scheduler_default  # Uses 2 nodes, 2 GPUs/node = 4 workers

        # Simulate some assignments
        graph1 = create_test_graph("g1", 1)
        scheduler.worker_to_graph_assignment[0] = graph1
        scheduler.node_active_worker_count[0] = 1
        scheduler.node_free_gpus[0] = [1]  # Worker 0 on node 0 is busy, worker 1 is free
        scheduler.node_free_gpus[1] = [2, 3]

        scheduler._reset_scheduler_state()  # Call the reset method

        # Check if state is reset to initial conditions
        self.assertEqual(len(scheduler.worker_to_graph_assignment), scheduler.num_workers)
        self.assertTrue(all(assignment is None for assignment in scheduler.worker_to_graph_assignment.values()))
        self.assertEqual(scheduler.node_active_worker_count, collections.defaultdict(int))

        # Verify node_free_gpus is correctly re-initialized
        expected_free_gpus = {}
        for i in range(scheduler.num_physical_nodes):
            expected_free_gpus[i] = list(range(i * scheduler.gpus_per_node, (i + 1) * scheduler.gpus_per_node))

        self.assertEqual(dict(scheduler.node_free_gpus), expected_free_gpus)

    def test_get_original_graph_id(self):
        """
        Test the _get_original_graph_id helper method.
        """
        graph_orig = TaskGraph(graph_id="original_task")
        graph_split1 = TaskGraph(graph_id="original_task_final_1")
        graph_split2 = TaskGraph(graph_id="original_task_reconv1_s0_final_2")
        graph_no_suffix = TaskGraph(graph_id="another_task")

        self.assertEqual(self.scheduler_default._get_original_graph_id(graph_orig), "original_task")
        self.assertEqual(self.scheduler_default._get_original_graph_id(graph_split1), "original_task")
        # Assuming the current logic splits by "_final_" first
        self.assertEqual(self.scheduler_default._get_original_graph_id(graph_split2), "original_task_reconv1_s0")
        self.assertEqual(self.scheduler_default._get_original_graph_id(graph_no_suffix), "another_task")

    def test_apportion_workers_to_tasks_even_strategy(self):
        """
        Test _apportion_workers_to_tasks with 'even' strategy.
        """
        scheduler = TaskScheduler(num_physical_nodes=1, gpus_per_node=10)  # 10 workers

        # Scenario 1: Fewer tasks than workers
        tasks_info1 = [(create_test_graph("g1", 1, model_params=100), 100.0), (create_test_graph("g2", 1, model_params=50), 50.0)]  # 2 tasks
        apportionment1 = scheduler._apportion_workers_to_tasks(tasks_info1, 10, "even")
        # Expected: 10 workers / 2 tasks = 5 workers per task
        self.assertEqual(apportionment1.get("g1"), 5)
        self.assertEqual(apportionment1.get("g2"), 5)
        self.assertEqual(sum(apportionment1.values()), 10)

        # Scenario 2: More workers, not perfectly divisible
        tasks_info2 = [(create_test_graph("gA", 1), 10.0), (create_test_graph("gB", 1), 20.0), (create_test_graph("gC", 1), 30.0)]  # 3 tasks
        apportionment2 = scheduler._apportion_workers_to_tasks(tasks_info2, 10, "even")
        # Expected: 10 workers / 3 tasks. Base = 3. Remainder = 1.
        # gA, gB, gC (sorted by ID)
        # gA: 3 + 1 = 4
        # gB: 3 + 0 = 3 (if gA gets remainder) -> this depends on sort order for remainder
        # gC: 3 + 0 = 3
        # Let's check the sum and individual counts based on sorted order of graph_ids.
        self.assertEqual(sum(apportionment2.values()), 10)
        counts = collections.Counter(apportionment2.values())
        self.assertEqual(counts[4], 1)  # One task gets 4
        self.assertEqual(counts[3], 2)  # Two tasks get 3

        # Scenario 3: Equal tasks and workers
        tasks_info3 = [(create_test_graph(f"t{i}", 1), 10.0) for i in range(5)]  # 5 tasks
        apportionment3 = scheduler._apportion_workers_to_tasks(tasks_info3, 5, "even")
        self.assertTrue(all(count == 1 for count in apportionment3.values()))
        self.assertEqual(sum(apportionment3.values()), 5)

        # Scenario 4: No tasks
        apportionment4 = scheduler._apportion_workers_to_tasks([], 10, "even")
        self.assertEqual(apportionment4, {})

        # Scenario 5: No workers
        apportionment5 = scheduler._apportion_workers_to_tasks(tasks_info1, 0, "even")
        self.assertTrue(all(count == 0 for count in apportionment5.values()))

    def test_apportion_workers_to_tasks_param_aware_strategy(self):
        """
        Test _apportion_workers_to_tasks with 'param_aware' strategy.
        """
        scheduler = TaskScheduler(num_physical_nodes=1, gpus_per_node=10)  # 10 workers

        # Tasks sorted by size for easier verification of param_aware logic
        task_large = create_test_graph("g_large", 1, model_params=300)
        task_medium = create_test_graph("g_medium", 1, model_params=200)
        task_small = create_test_graph("g_small", 1, model_params=100)

        tasks_info_param = [(task_large, 300.0), (task_medium, 200.0), (task_small, 100.0)]  # 3 tasks, total size 600

        # 10 workers for 3 tasks. Each gets 1 initially. 7 remaining.
        # param_aware will distribute remaining 7 workers one by one, cycling through tasks sorted by size (desc)
        # g_large (300), g_medium (200), g_small (100)
        # Initial: g_large:1, g_medium:1, g_small:1 (3 workers used)
        # Remaining 7:
        # 1. g_large (+1) -> 2
        # 2. g_medium (+1) -> 2
        # 3. g_small (+1) -> 2
        # 4. g_large (+1) -> 3
        # 5. g_medium (+1) -> 3
        # 6. g_small (+1) -> 3
        # 7. g_large (+1) -> 4
        # Final: g_large: 4, g_medium: 3, g_small: 3
        apportionment_param = scheduler._apportion_workers_to_tasks(tasks_info_param, 10, "param_aware")
        self.assertEqual(apportionment_param.get("g_large"), 4)
        self.assertEqual(apportionment_param.get("g_medium"), 3)
        self.assertEqual(apportionment_param.get("g_small"), 3)
        self.assertEqual(sum(apportionment_param.values()), 10)

        # Scenario: All tasks have zero size (should fall back to even)
        task_zero1 = create_test_graph("g_zero1", 1, model_params=0)
        task_zero2 = create_test_graph("g_zero2", 1, model_params=0)
        tasks_info_zero = [(task_zero1, 0.0), (task_zero2, 0.0)]  # 2 tasks, 10 workers
        apportionment_zero = scheduler._apportion_workers_to_tasks(tasks_info_zero, 10, "param_aware")
        # Expected: fallback to 'even', so 5 workers per task
        self.assertEqual(apportionment_zero.get("g_zero1"), 5)
        self.assertEqual(apportionment_zero.get("g_zero2"), 5)
        self.assertEqual(sum(apportionment_zero.values()), 10)

    def test_schedule_no_tasks(self):
        """
        Test scheduling when no tasks are provided.
        """
        assignments = self.scheduler_default.schedule_and_assign_tasks([])
        self.assertTrue(all(graph is None for graph in assignments.values()))
        self.assertEqual(len(assignments), self.scheduler_default.num_workers)

    def test_schedule_fewer_tasks_than_workers(self):
        """
        Test scheduling with fewer tasks than available workers.
        All tasks should be scheduled, and workers apportioned.
        """
        scheduler = TaskScheduler(num_physical_nodes=2, gpus_per_node=2)  # 4 workers
        task1 = create_test_graph("task1", 2, model_params="100M")  # Size 100M
        task2 = create_test_graph("task2", 1, model_params="50M")  # Size 50M
        original_tasks = [task1, task2]  # 2 tasks

        assignments = scheduler.schedule_and_assign_tasks(
            original_tasks,
            apportion_strategy="even",  # 4 workers / 2 tasks = 2 workers per task
            consider_node_cohesion=True,
            consider_node_load=True,
            consider_rank_preference=True,
        )

        assigned_counts = collections.Counter(g.graph_id for g in assignments.values() if g)
        # The graph IDs will be suffixed by discover_and_split_parallel_paths
        # e.g., "task1_final_1", "task2_final_1" if they are irreducible
        # For simple graphs, discover_and_split_parallel_paths returns them as is, then renames.

        # We expect two unique graph IDs in assignments, each assigned to 2 workers
        self.assertEqual(len(assigned_counts), 2)  # Two unique tasks were scheduled

        num_workers_per_task = {}
        for worker, graph_obj in assignments.items():
            if graph_obj:
                num_workers_per_task.setdefault(graph_obj.graph_id, 0)
                num_workers_per_task[graph_obj.graph_id] += 1

        self.assertTrue(all(count == 2 for count in num_workers_per_task.values()))
        self.assertEqual(sum(num_workers_per_task.values()), scheduler.num_workers)  # All workers used

    def test_schedule_more_tasks_than_workers_raises_error(self):
        """
        Test scheduling with more tasks than workers.
        This should raise a ValueError as per the new requirement.
        """
        scheduler = TaskScheduler(num_physical_nodes=1, gpus_per_node=2)  # 2 workers
        tasks = [create_test_graph("t1", 1, model_params="10M"), create_test_graph("t2", 1, model_params="20M"), create_test_graph("t3", 1, model_params="5M")]  # 3 tasks

        with self.assertRaisesRegex(ValueError, "Cannot assign all tasks"):
            scheduler.schedule_and_assign_tasks(tasks)

    def test_schedule_with_task_splitting(self):
        """
        Test scheduling with a task that gets split into multiple irreducible subgraphs.
        """
        scheduler = TaskScheduler(num_physical_nodes=1, gpus_per_node=4)  # 4 workers

        # Create a graph that will be split (e.g., two parallel paths merging)
        # A -> B --\
        #           C -> D
        # E -> F --/
        # discover_and_split_parallel_paths should create two subgraphs:
        # 1. A -> B -> C -> D
        # 2. E -> F -> C -> D
        # (Actual splitting logic is in task_loader, we test its integration)
        splittable_graph = TaskGraph(graph_id="splittable")
        splittable_graph.add_nodes(
            [
                Node(node_id="A", node_type=NodeType.DATA_LOAD),
                Node(node_id="B", node_type=NodeType.COMPUTE, dependencies=["A"]),
                Node(node_id="E", node_type=NodeType.DATA_LOAD),  # Another entry for a parallel path
                Node(node_id="F", node_type=NodeType.COMPUTE, dependencies=["E"]),
                Node(node_id="C", node_type=NodeType.COMPUTE, dependencies=["B", "F"]),  # Merge point
                Node(node_id="D", node_type=NodeType.MODEL_TRAIN, dependencies=["C"], config={"model_params": "10B"}),
            ]
        )
        splittable_graph.build_adjacency_lists()
        self.assertTrue(splittable_graph.validate_graph()[0], "Test splittable graph is invalid")

        assignments = scheduler.schedule_and_assign_tasks(
            [splittable_graph],
            apportion_strategy="even",  # 4 workers / 2 subgraphs = 2 workers per subgraph
            consider_node_cohesion=True,
        )

        # Count how many workers are assigned to each unique *scheduled* subgraph ID
        scheduled_subgraph_worker_counts = collections.defaultdict(int)
        for graph_obj in assignments.values():
            if graph_obj:
                scheduled_subgraph_worker_counts[graph_obj.graph_id] += 1

        # We expect two distinct subgraphs to be scheduled (e.g., "splittable_final_1", "splittable_final_2")
        self.assertEqual(len(scheduled_subgraph_worker_counts), 2, "Should schedule two subgraphs after splitting.")
        # Each of these two subgraphs should get 2 workers
        for subgraph_id, count in scheduled_subgraph_worker_counts.items():
            self.assertEqual(count, 2, f"Subgraph {subgraph_id} should have 2 workers.")

        self.assertEqual(sum(scheduled_subgraph_worker_counts.values()), scheduler.num_workers, "All workers should be utilized.")

    def test_placement_logic_node_cohesion(self):
        """
        Test placement logic focusing on node cohesion.
        If a task gets multiple workers, they should ideally be on the same physical node if cohesion is enabled.
        """
        # 1 physical node, 4 GPUs. So all workers for a task *must* be on node 0.
        scheduler = TaskScheduler(num_physical_nodes=1, gpus_per_node=4)
        task1 = create_test_graph("task_cohesive", 1, model_params="100M")

        # Schedule task1, which should get all 4 workers (apportion_strategy='even' or 'param_aware' with one task)
        assignments = scheduler.schedule_and_assign_tasks(
            [task1],
            apportion_strategy="even",
            consider_node_cohesion=True,
            consider_node_load=True,  # Load won't matter much with 1 node
            consider_rank_preference=True,
        )

        assigned_workers_for_task1 = []
        for worker_rank, graph_obj in assignments.items():
            if graph_obj and "task_cohesive" in graph_obj.graph_id:  # Check for the scheduled subgraph from task1
                assigned_workers_for_task1.append(worker_rank)

        self.assertEqual(len(assigned_workers_for_task1), 4, "Task1 should get 4 workers.")
        # All workers (0, 1, 2, 3) are on physical node 0 (worker_rank // gpus_per_node)
        self.assertTrue(all((rank // scheduler.gpus_per_node) == 0 for rank in assigned_workers_for_task1))

        # More complex: 2 nodes, 2 GPUs/node (4 workers total). 1 task needing 3 workers.
        scheduler_2n2g = TaskScheduler(num_physical_nodes=2, gpus_per_node=2)
        task_needs_3 = create_test_graph("task_3w", 1, model_params="200M")
        # With 4 workers and 1 task, task_3w gets 4 workers. Let's adjust.
        # We need to control apportionment. Create 2 tasks, one big, one small.
        task_big = create_test_graph("task_big", 1, model_params="300M")  # Should get more workers
        task_small = create_test_graph("task_small", 1, model_params="10M")  # Should get fewer

        # 4 workers, 2 tasks. 'param_aware'
        # task_big (300), task_small (10)
        # Initial: task_big:1, task_small:1 (2 workers used)
        # Remaining 2:
        # 1. task_big (+1) -> 2
        # 2. task_big (+1) -> 3 (if cycling favors largest first for all remainders)
        # Actually, it's 1. task_big (+1) -> 2, 2. task_small (+1) -> 2 if cycling through all tasks
        # Let's re-check _apportion_workers_to_tasks for param_aware with 2 tasks, 4 workers:
        # Initial: big:1, small:1. Remaining: 2.
        # Sorted by size: [big, small]
        # 1. big gets +1 -> big:2, small:1
        # 2. small gets +1 -> big:2, small:2  <-- This is how current param_aware works (cycles)
        # To force one task to get 3, let's use 3 workers for 1 task.
        scheduler_1n3g = TaskScheduler(num_physical_nodes=1, gpus_per_node=3)
        assignments_3w = scheduler_1n3g.schedule_and_assign_tasks(
            [task_big],  # Only one task
            apportion_strategy="even",  # Will get all 3 workers
            consider_node_cohesion=True,
        )
        assigned_workers_for_task_big = []
        for r, g in assignments_3w.items():
            if g and "task_big" in g.graph_id:
                assigned_workers_for_task_big.append(r)
        self.assertEqual(len(assigned_workers_for_task_big), 3)
        self.assertTrue(all((rank // scheduler_1n3g.gpus_per_node) == 0 for rank in assigned_workers_for_task_big))

    def test_placement_no_node_load_no_rank_preference(self):
        """
        Test placement when node load and rank preference are disabled.
        Cohesion should still work if enabled. Placement might be less deterministic for tie-breaking.
        """
        scheduler = TaskScheduler(num_physical_nodes=2, gpus_per_node=2)  # 4 workers (0,1 on node 0; 2,3 on node 1)
        task1 = create_test_graph("t1", 1, model_params="100M")
        task2 = create_test_graph("t2", 1, model_params="100M")
        # 2 tasks, 4 workers. 'even' -> 2 workers per task.

        assignments = scheduler.schedule_and_assign_tasks(
            [task1, task2],
            apportion_strategy="even",
            consider_node_cohesion=True,  # Cohesion is on
            consider_node_load=False,  # Load balancing is off
            consider_rank_preference=False,  # Rank preference is off
        )

        # Verify all workers are assigned
        self.assertEqual(len([g for g in assignments.values() if g is not None]), 4)

        # Check that each task got 2 workers
        worker_counts = collections.defaultdict(int)
        task_assignments = collections.defaultdict(list)
        for worker, graph in assignments.items():
            if graph:
                # Extract base name (e.g., "t1" from "t1_final_1")
                base_graph_id = scheduler._get_original_graph_id(graph) if "_final_" in graph.graph_id else graph.graph_id
                worker_counts[base_graph_id] += 1
                task_assignments[base_graph_id].append(worker)

        self.assertEqual(worker_counts.get("t1"), 2)
        self.assertEqual(worker_counts.get("t2"), 2)

        # With cohesion ON, the 2 workers for t1 should be on the same node.
        # And the 2 workers for t2 should be on the same node.
        # Node 0: GPUs 0, 1. Node 1: GPUs 2, 3.

        # Example: t1_final_1 assigned to workers on node 0 (0,1)
        #          t2_final_1 assigned to workers on node 1 (2,3)
        # This is one possible cohesive assignment.

        # We need to find which task is which after potential renaming by discover_and_split
        scheduled_task_ids = list(task_assignments.keys())
        if not (len(scheduled_task_ids) == 2 and all("_final_" in tid for tid in scheduled_task_ids)):
            # If splitting didn't happen as expected, this test might be flawed.
            # However, for simple graphs, they are returned as is and then renamed.
            pass

        for task_id_key in task_assignments:  # task_id_key will be like "t1_final_1"
            assigned_ranks = task_assignments[task_id_key]
            self.assertEqual(len(assigned_ranks), 2, f"Task {task_id_key} should have 2 workers")
            # Check cohesion: both workers for this task should be on the same physical node
            physical_node_for_task = assigned_ranks[0] // scheduler.gpus_per_node
            self.assertTrue(all((r // scheduler.gpus_per_node) == physical_node_for_task for r in assigned_ranks), f"Workers for task {task_id_key} are not on the same physical node: {assigned_ranks}")

        # Check that the two tasks are on different physical nodes if possible (due to even distribution of tasks)
        # This is a secondary effect of load balancing if it were on, but here, it's about filling nodes.
        node_indices_used_by_tasks = set()
        for task_id_key in task_assignments:
            node_indices_used_by_tasks.add(task_assignments[task_id_key][0] // scheduler.gpus_per_node)

        # If there are enough nodes for each task to be on a separate node, they should be.
        if scheduler.num_physical_nodes >= len(task_assignments):
            self.assertEqual(len(node_indices_used_by_tasks), len(task_assignments), "Tasks should be on different physical nodes if possible.")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=2)
