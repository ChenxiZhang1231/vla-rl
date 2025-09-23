# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
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

from loguru import logger
from itertools import product
from math_verify.errors import TimeoutException
from math_verify.grader import sympy_expr_eq

from sympy import Basic, MatrixBase
from math_verify.utils import timeout


def verify(
    gold: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    target: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
    timeout_seconds: int = 5,
) -> bool:
    """Verifies if the target expression matches the gold expression using multiple comparison strategies.

    This function implements a comprehensive comparison system for mathematical expressions,
    handling various types of mathematical objects (numbers, expressions, sets, matrices, etc.)
    with multiple fallback strategies.

    Note:
        - It's expected that both gold and pred has been parsed with math_verify.parse function.
        - Function is not symmetric, gold answer should be passed as gold and prediction as pred. The non-symmetric nature appears at assignment simplification and equation interval conversion.

    Args:
        gold: The reference/correct expression(s). Can be:
            - A single SymPy expression (Basic or MatrixBase)
            - A string
            - A list of any of the above
        target: The expression(s) to verify. Same types as gold.
        float_rounding: Number of decimal places to round floats to. Defaults to 6.
        numeric_precision: Number of decimal places to consider for numeric comparisons. Defaults to 15.
            - If you know the evaluated expressions will be small, you should increase this. See: https://docs.sympy.org/latest/modules/evalf.html
        strict: Whether to enforce strict comparison mode. Defaults to True.
            - In strict mode: Variables matter and sets are not comparable with tuples
            - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        timeout_seconds: Maximum time in seconds to spend on any single comparison operation.
            Defaults to 5 seconds.

    Returns:
        bool: True if target matches gold according to any of the comparison strategies,
              False otherwise.

    Comparison Strategy:
        1. String to String comparison
        2. Numeric expressions: Comparison within specified precision
        3. Symbolic equality through simplification
        4. Special handling for:
            - Relational expressions (equations/inequalities)
            - Sets and intervals
            - Matrices and vectors
            - Complex numbers
        5. Robust error handling with timeout protection

    Example:
        >>> verify(sympy.Rational(1, 3), 0.333333)  # Numeric comparison
        True
        >>> verify(sympy.Symbol('x') + 1, sympy.Symbol('y') + 1, strict=False)  # Variable matching
        True
        >>> verify(sympy.FiniteSet(1, 2), sympy.Tuple(1, 2), strict=False)  # Set-tuple comparison
        True
    """

    @timeout(timeout_seconds=timeout_seconds)
    def compare_single_extraction(gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str) -> bool:
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(target, (Basic, MatrixBase)):
            return sympy_expr_eq(gold, target, float_rounding, numeric_precision, strict)

        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)
        except Exception:
            #! Do not attempt to print out the g and t during handling of exception
            # Because a) it can throw an exception itself and b) it can cause it to be stuck forever during str conversion
            # logger.exception("Error during comparison")
            return False
        except TimeoutException:
            # logger.error("Timeout during comparison")
            return False

    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(compare_single_extraction_wrapper(g, t) for g, t in product(gold, target))
