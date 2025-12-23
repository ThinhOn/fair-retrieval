from .ilp_solver import ILPSolver
from .bfs_solver import BFSSolver
from .lagrange_solver import LagrangeSolver
from .network_flow_solver import NetworkFlowSolver


solvers = {
    "ilp": ILPSolver,
    "bfs": BFSSolver,
    "lagrange": LagrangeSolver,
    "network_flow": NetworkFlowSolver,
}


def build_solver(args):
    if solvers.get(args.solver, None) is not None:
        return solvers[args.solver]()
    else:
        raise NameError(f"Undefined indexing method for {args.solver}!")