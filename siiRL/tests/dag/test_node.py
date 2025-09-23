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

import asyncio
import unittest
from typing import Any, Dict

from siirl.workers.dag import Node, NodeRole, NodeStatus, NodeType


# Helper function for testing executable_ref (defined inside the test module)
def local_sync_executable(data: Any, node_config: Dict = None) -> Dict:
    """A simple local synchronous executable function for testing."""
    # print(f"local_sync_executable called with data: {data}, config: {node_config}")
    return {"processed_data": data, "config_used": node_config}


async def local_async_executable(data: Any, node_config: Dict = None) -> Dict:
    """A simple local asynchronous executable function for testing."""
    # print(f"local_async_executable called with data: {data}, config: {node_config}")
    await asyncio.sleep(0.01)  # Simulate an IO operation
    return {"processed_async_data": data, "config_used": node_config}


"""A non-callable local variable for testing error handling."""
local_not_callable = "I am not a function, I am a result."


# Create a function that always fails for testing retries
async def always_fail_func(**kwargs):
    print(f"  [Executable] always_fail_func: Executing and about to fail...")
    await asyncio.sleep(0.05)
    raise RuntimeError("Simulated execution error")


class TestNode(unittest.TestCase):
    """Unit tests for the Node class"""

    def test_node_creation_valid(self):
        """Test the creation of a Node with valid parameters."""
        node = Node(node_id="n1", node_type=NodeType.COMPUTE, node_role=NodeRole.DEFAULT, dependencies=["dep1"], config={"key": "value"}, executable_ref=f"{__name__}.local_sync_executable", retry_limit=3)
        self.assertEqual(node.node_id, "n1")
        self.assertEqual(node.node_type, NodeType.COMPUTE)
        self.assertEqual(node.dependencies, ["dep1"])
        self.assertEqual(node.config, {"key": "value"})
        self.assertTrue(callable(node.executable))
        self.assertEqual(node.node_role, NodeRole.DEFAULT)
        self.assertEqual(node.retry_limit, 3)
        self.assertEqual(node.status, NodeStatus.PENDING)

    def test_node_creation_minimal(self):
        """Test the creation of a Node with minimal parameters."""
        node = Node(node_id="n2", node_type=NodeType.DATA_LOAD)
        self.assertEqual(node.node_id, "n2")
        self.assertEqual(node.node_type, NodeType.DATA_LOAD)
        self.assertEqual(node.node_role, NodeRole.DEFAULT)
        self.assertEqual(node.dependencies, [])
        self.assertEqual(node.config, {})
        self.assertIsNone(node.executable)
        self.assertEqual(node.retry_limit, 0)

    def test_node_creation_invalid_id(self):
        """Test the creation of a Node with an invalid node_id (empty string)."""
        with self.assertRaisesRegex(ValueError, "node_id must be a non-empty string"):
            Node(node_id="", node_type=NodeType.COMPUTE)

    def test_node_creation_invalid_type(self):
        """Test the creation of a Node with an invalid node_type."""
        with self.assertRaisesRegex(ValueError, "node_type must be a member of the NodeType enum"):
            Node(node_id="n3", node_type="INVALID_TYPE")  # type: ignore

    def test_resolve_executable_non_existent(self):
        """Test resolving a non-existent executable function reference."""
        with self.assertRaisesRegex(ImportError, "Failed to load the executable function from 'non_existent_module.non_existent_func'"):
            Node(node_id="n4", node_type=NodeType.COMPUTE, executable_ref="non_existent_module.non_existent_func")

    def test_resolve_executable_not_callable(self):
        """Test resolving a non-callable executable function reference."""
        with self.assertRaisesRegex(ImportError, f".*The object resolved from .* is not callable."):
            Node(node_id="n5", node_type=NodeType.COMPUTE, executable_ref=f"{__name__}.local_not_callable")

    def test_add_remove_dependency(self):
        """Test adding and removing dependencies."""
        node = Node(node_id="n6", node_type=NodeType.COMPUTE)
        node.add_dependency("dep1")
        self.assertIn("dep1", node.dependencies)
        node.add_dependency("dep2")
        self.assertIn("dep2", node.dependencies)
        node.add_dependency("dep1")  # Adding a duplicate dependency should have no effect
        self.assertEqual(node.dependencies.count("dep1"), 1)
        node.remove_dependency("dep1")
        self.assertNotIn("dep1", node.dependencies)
        node.remove_dependency("non_existent_dep")  # Removing a non-existent dependency should have no effect
        self.assertEqual(node.dependencies, ["dep2"])

    def test_is_ready(self):
        """Test the logic of the is_ready method."""
        node_no_deps = Node(node_id="n_no_deps", node_type=NodeType.COMPUTE)
        node_with_deps = Node(node_id="n_with_deps", node_type=NodeType.COMPUTE, dependencies=["d1", "d2"])

        self.assertTrue(node_no_deps.is_ready(set()))
        self.assertTrue(node_no_deps.is_ready({"d1"}))  # Unrelated completed nodes

        self.assertFalse(node_with_deps.is_ready(set()))
        self.assertFalse(node_with_deps.is_ready({"d1"}))
        self.assertTrue(node_with_deps.is_ready({"d1", "d2"}))
        self.assertTrue(node_with_deps.is_ready({"d1", "d2", "d3"}))

        node_with_deps.update_status(NodeStatus.RUNNING)
        self.assertFalse(node_with_deps.is_ready({"d1", "d2"}), "Nodes not in PENDING status should not be ready")

    def test_update_status(self):
        """Test updating the node status."""
        node = Node(node_id="n7", node_type=NodeType.COMPUTE)
        self.assertEqual(node.status, NodeStatus.PENDING)
        node.update_status(NodeStatus.COMPLETED)
        self.assertEqual(node.status, NodeStatus.COMPLETED)
        self.assertIsNone(node.error_info)

        node.update_status(NodeStatus.FAILED, "It failed")
        self.assertEqual(node.status, NodeStatus.FAILED)
        self.assertEqual(node.error_info, "It failed")

        # error_info should be cleared upon successful completion
        node.update_status(NodeStatus.COMPLETED)
        self.assertIsNone(node.error_info)

    def test_retry_logic(self):
        """Test retry-related logic."""
        node = Node(node_id="n8", node_type=NodeType.COMPUTE, retry_limit=2)
        self.assertFalse(node.can_retry())  # Initial state is PENDING
        node.update_status(NodeStatus.FAILED)
        self.assertTrue(node.can_retry())
        node.increment_retry_count()
        self.assertEqual(node.retries_done, 1)
        self.assertTrue(node.can_retry())
        node.increment_retry_count()
        self.assertEqual(node.retries_done, 2)
        self.assertFalse(node.can_retry())  # Retry limit reached

    def test_execute_no_executable(self):
        """Test executing a node without an executable function."""
        node = Node(node_id="n9", node_type=NodeType.BARRIER_SYNC)
        output = asyncio.run(node.run())
        self.assertIsNone(output)
        self.assertEqual(node.status, NodeStatus.RUNNING)  # run sets the status to RUNNING internally
        # If there is no executable function, it will not automatically change to COMPLETED
        # In the original code, it returns directly when there is no executable, and the status remains RUNNING
        # The caller may need to update the status later
        # For testing purposes, we check if it is None and the status is RUNNING
        # The actual scheduler should handle this situation

    def test_execute_sync_executable(self):
        """Test executing a synchronous executable function."""
        node_id = "n10_sync"
        node_config = {"multiplier": 3}
        # Note: The executable_ref here must point to a function in the test file or an imported function
        node = Node(node_id=node_id, node_type=NodeType.COMPUTE, executable_ref=f"{__name__}.local_sync_executable", config=node_config)

        # Node.run is an async def, so even if it calls a synchronous function internally, it still needs to be awaited
        # kwargs simulate the output from dependent nodes
        result_coro = node.run(data=5)  # run returns a coroutine
        result = asyncio.run(result_coro)  # Run this coroutine

        self.assertEqual(result, {"processed_data": 5, "config_used": node_config})
        self.assertEqual(node.output, {"processed_data": 5, "config_used": node_config})
        self.assertEqual(node.status, NodeStatus.COMPLETED)

    def test_execute_async_executable(self):
        """Test executing an asynchronous executable function."""
        node_id = "n11_async"
        node_config = {"adder": 10}
        node = Node(node_id=node_id, node_type=NodeType.COMPUTE, executable_ref=f"{__name__}.local_async_executable", config=node_config)

        result_coro = node.run(data=7)  # run returns a coroutine
        result = asyncio.run(result_coro)  # Run this coroutine

        self.assertEqual(result, {"processed_async_data": 7, "config_used": node_config})
        self.assertEqual(node.output, {"processed_async_data": 7, "config_used": node_config})
        self.assertEqual(node.status, NodeStatus.COMPLETED)

    def test_execute_failing_executable(self):
        """Test executing a failing executable function."""
        node = Node(node_id="n12_fail", node_type=NodeType.COMPUTE, executable_ref=f"{__name__}.always_fail_func")

        with self.assertRaisesRegex(RuntimeError, "An error occurred while executing node n12_fail: Simulated execution error"):
            asyncio.run(node.run())  # The coroutine returned by run will raise an exception when awaited

        self.assertEqual(node.status, NodeStatus.FAILED)
        self.assertIn("Simulated execution error", node.error_info)


if __name__ == "__main__":
    # Ensure that executable_ref in the test can correctly resolve functions in this file
    # If the test file is named test_my_module.py, __name__ will be 'test_my_module'
    # If you run python test_my_module.py directly, __name__ will be '__main__'
    # To ensure consistency, you can set a global variable or dynamically obtain the module name in the test
    # But usually the unittest test loader will handle the module import correctly, so __name__ should point to the test module name

    # print(f"Running tests with __name__ = {__name__}")
    # print(f"Attempting to use executable_ref like: {__name__}.local_sync_executable")

    unittest.main(verbosity=2)
