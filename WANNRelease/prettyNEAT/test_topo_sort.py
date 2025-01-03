import numpy as np
import networkx as nx
import time
import statistics

def original_topo_sort(connMat):
    """Original topological sort implementation"""
    edge_in = np.sum(connMat, axis=0)
    Q = np.where(edge_in==0)[0]
    
    for i in range(len(connMat)):
        if (len(Q) == 0) or (i >= len(Q)):
            return False  # Cycle found
        
        edge_out = connMat[Q[i],:]
        edge_in = edge_in - edge_out
        nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
        Q = np.hstack((Q, nextNodes))
        
        if sum(edge_in) == 0:
            break
    
    return Q

def nx_topo_sort(connMat):
    """NetworkX-based topological sort implementation"""
    G = nx.DiGraph()
    
    # Add edges where connMat is 1
    rows, cols = np.where(connMat == 1)
    edges = zip(rows, cols)
    G.add_edges_from(edges)
    
    try:
        # Convert to list and then numpy array to match original output format
        return np.array(list(nx.topological_sort(G)))
    except nx.NetworkXUnfeasible:
        return False

# Test cases
def test_topo_sorts():
    # Test Case 1: Simple DAG
    connMat1 = np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    
    # Test Case 2: With cycle
    connMat2 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    
    # Run tests
    print("Test Case 1 - Simple DAG:")
    print("Original sort:", original_topo_sort(connMat1))
    print("NetworkX sort:", nx_topo_sort(connMat1))
    print("\nTest Case 2 - With Cycle:")
    print("Original sort:", original_topo_sort(connMat2))
    print("NetworkX sort:", nx_topo_sort(connMat2))
    
    # Add timing tests
    # Create a larger test case for meaningful timing
    n = 1000
    sparsity = 0.01  # 1% of edges present
    large_mat = np.random.random((n, n)) < sparsity
    np.fill_diagonal(large_mat, 0)  # Remove self-loops
    
    # Run multiple trials
    n_trials = 100
    original_times = []
    nx_times = []
    
    for _ in range(n_trials):
        # Time original implementation
        start = time.perf_counter()
        original_topo_sort(large_mat)
        original_times.append(time.perf_counter() - start)
        
        # Time NetworkX implementation
        start = time.perf_counter()
        nx_topo_sort(large_mat)
        nx_times.append(time.perf_counter() - start)
    
    print("\nTiming Results (average over", n_trials, "trials):")
    print(f"Original implementation: {statistics.mean(original_times):.6f} seconds")
    print(f"NetworkX implementation: {statistics.mean(nx_times):.6f} seconds")
    print(f"NetworkX is {statistics.mean(original_times)/statistics.mean(nx_times):.2f}x faster")

if __name__ == "__main__":
    test_topo_sorts() 