from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    
    Hints:
        1. Ensure that you visit the computation graph in a post-order depth-first search.
        2. When the children nodes of the current node are visited, add the current node 
            at the front of the result order list.
    """
    # BEGIN ASSIGN2_1
    
    # WHY TOPOLOGICAL SORT?
    # During backpropagation, we need to process nodes in an order where each node
    # is processed AFTER all nodes that depend on it. This ensures gradients flow
    # correctly from outputs to inputs.
    #
    # EXAMPLE: For computation z = (x + y) * w

    # Graph structure: x, y, w (inputs) → add_node → mul_node → z (output)
    # Topological order: [z, mul_node, add_node, w, y, x]
    # This ensures we compute gradients for z first, then propagate to mul_node,
    # then to add_node, and finally to the leaf variables.
    
    result = []  # Will store nodes in post-order (leaves first, output last)
    visited = set()  # Track visited nodes to avoid processing the same node twice
    
    def dfs(var: Variable) -> None:
        """
        Perform depth-first search to visit all nodes in the computation graph.
        
        POST-ORDER TRAVERSAL: Visit children before parent
        - This means we add a node to result AFTER visiting all its dependencies
        - Example: For y = f(x), we visit x first, then add y to result
        """
        
        # SKIP CONDITIONS:
        # 1. Already visited: Prevents infinite loops in graphs with multiple paths
        #    Example: z = x + x creates two paths from x to z
        # 2. Constants: Don't need gradients (e.g., scalar multipliers like 2.0)
        #    Example: y = 2.0 * x, the 2.0 is constant and doesn't need gradients
        if var.unique_id in visited or var.is_constant():
            return
        
        # Mark as visited BEFORE processing children to handle cycles/shared nodes
        # Example: If z = x + x, marking x as visited prevents processing it twice
        visited.add(var.unique_id)
        
        # RECURSIVE STEP: Visit all parent nodes (dependencies) first
        # "Parents" are the nodes that this variable depends on in the computation
        # Example: For z = x + y, the parents of z are x and y
        # We visit them first to ensure post-order traversal
        for parent in var.parents:
            dfs(parent)
        
        # POST-ORDER: Add current node AFTER visiting all its dependencies
        # This ensures dependencies appear before dependents in the result list
        # Example: For z = x + y, after visiting x and y, we add z
        # Result so far: [x, y, z]
        result.append(var)
    
    # Start DFS from the output variable (rightmost in computation graph)
    dfs(variable)
    
    # REVERSE for backpropagation order (output to inputs)
    # After DFS, result = [inputs..., intermediate..., output]
    # After reverse: [output, intermediate..., inputs...]
    # 
    # WHY REVERSE?
    # Backpropagation flows from output to inputs (chain rule application)
    # Example: For loss = f(g(x)), we need:
    #   1. First compute d_loss/d_loss = 1.0
    #   2. Then compute d_loss/d_g using chain rule
    #   3. Finally compute d_loss/d_x using chain rule
    # Reversed order gives us exactly this sequence
    return reversed(result)
    # END ASSIGN2_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        1. Traverse nodes in topological order
        2. If the node is a leaf, the derivative should be accumulated
        3. Otherwise, the derivative should be propagated via chain rule
    """
    # BEGIN ASSIGN2_1
    
    # WHAT IS BACKPROPAGATION?
    # Backpropagation computes gradients (derivatives) of the output with respect to
    # all input variables in the computation graph using the chain rule.
    #
    # EXAMPLE: Simple computation
    # Forward:  x = 2.0, y = 3.0
    #           z = x + y = 5.0
    #           loss = z * 2 = 10.0
    #
    # Backward: We want d_loss/d_x and d_loss/d_y
    #           Starting with d_loss/d_loss = 1.0
    #           d_loss/d_z = 2.0 (since loss = z * 2)
    #           d_loss/d_x = d_loss/d_z * d_z/d_x = 2.0 * 1.0 = 2.0
    #           d_loss/d_y = d_loss/d_z * d_z/d_y = 2.0 * 1.0 = 2.0
    
    # STEP 1: Get nodes in topological order (output → inputs)
    # WHY? We need to compute gradients in the correct order so that when we
    # process a node, all gradients flowing into it have already been computed.
    topo_order = topological_sort(variable)
    
    # STEP 2: Initialize gradient storage
    # Dictionary maps variable.unique_id → accumulated gradient
    # WHY DICTIONARY? Multiple paths can lead to the same variable, requiring
    # gradient accumulation.
    #
    # EXAMPLE: z = x + x (x is used twice)
    # Computation graph: x → add_node → z
    #                    x ↗
    # When backpropagating, x receives gradients from BOTH paths:
    # d_z/d_x = d_z/d_x1 + d_z/d_x2 = 1.0 + 1.0 = 2.0
    derivs = {variable.unique_id: deriv}
    
    # STEP 3: Traverse nodes in topological order and propagate gradients
    # Process from output toward inputs, applying chain rule at each step
    for var in topo_order:
        
        # Skip nodes that don't have a gradient computed yet
        # This shouldn't happen in a correct graph, but provides safety
        if var.unique_id not in derivs:
            continue
        
        # Get the accumulated gradient for this variable
        # This is the sum of all gradients flowing into this node from its children
        d = derivs[var.unique_id]
        
        # CASE 1: LEAF NODE (input variable, e.g., model parameters or input data)
        # WHY CHECK is_leaf()? Leaf nodes are the variables we're computing gradients for
        # (e.g., weights and biases in a neural network)
        #
        # EXAMPLE: In loss = (w * x + b), w and b are leaves
        # We accumulate their gradients for the optimizer to update them
        if var.is_leaf():
            var.accumulate_derivative(d)
        
        # CASE 2: INTERMEDIATE NODE (result of an operation)
        # WHY propagate? We need to continue the chain rule to reach leaf nodes
        #
        # EXAMPLE: For z = x + y, and loss = z * 2
        # When processing z (intermediate node):
        # - We have d_loss/d_z = 2.0
        # - chain_rule() computes contributions to parents:
        #   * d_loss/d_x = d_loss/d_z * d_z/d_x = 2.0 * 1.0 = 2.0
        #   * d_loss/d_y = d_loss/d_z * d_z/d_y = 2.0 * 1.0 = 2.0
        else:
            # Apply chain rule: propagate gradient to parent nodes
            # chain_rule() returns [(parent_var, local_gradient), ...]
            # local_gradient is the derivative of this node w.r.t. the parent
            for parent, parent_deriv in var.chain_rule(d):
                
                # GRADIENT ACCUMULATION: Critical for handling multiple paths
                # WHY ACCUMULATE? A variable can be used multiple times in different
                # operations, so we must sum all gradient contributions.
                #
                # EXAMPLE 1: z = x + x
                # x receives gradient from both uses: total = grad1 + grad2
                #
                # EXAMPLE 2: Neural network with shared weights
                # loss = f(W*x1) + g(W*x2)
                # W receives gradients from both loss terms:
                # d_loss/d_W = d_f/d_W + d_g/d_W
                #
                # EXAMPLE 3: Multi-path computation
                # a = x * 2
                # b = x * 3  
                # c = a + b
                # When computing d_c/d_x:
                # Path 1: c → a → x contributes 2.0
                # Path 2: c → b → x contributes 3.0
                # Total: d_c/d_x = 2.0 + 3.0 = 5.0
                if parent.unique_id in derivs:
                    # Parent already has a gradient from another path, add to it
                    derivs[parent.unique_id] = derivs[parent.unique_id] + parent_deriv
                else:
                    # First gradient contribution to this parent, initialize it
                    # Add 0.0 to create a new tensor, avoiding aliasing issues
                    derivs[parent.unique_id] = parent_deriv + 0.0
    
    # RESULT: All leaf nodes now have their accumulated gradients stored via
    # accumulate_derivative(), ready for optimization steps (e.g., SGD, Adam)
    # END ASSIGN2_1


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
