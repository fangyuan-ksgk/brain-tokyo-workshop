import numpy as np
import torch
import torch.nn as nn


# -- ANN Ordering -------------------------------------------------------- -- #

def getNodeOrder(nodeG,connG):
  """Builds connection matrix from genome through topological sorting.

  Args:
    nodeG - (np_array) - node genes
            [3 X nUniqueGenes]
            [0,:] == Node Id
            [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
            [2,:] == Activation function (as int)

    connG - (np_array) - connection genes
            [5 X nUniqueGenes] 
            [0,:] == Innovation Number (unique Id)
            [1,:] == Source Node Id
            [2,:] == Destination Node Id
            [3,:] == Weight Value
            [4,:] == Enabled?  

  Returns:
    Q    - [int]      - sorted node order as indices
    wMat - (np_array) - ordered weight matrix
           [N X N]

    OR

    False, False      - if cycle is found

  Todo:
    * setdiff1d is slow, as all numbers are positive ints is there a
      better way to do with indexing tricks (as in quickINTersect)?
  """
  conn = np.copy(connG)
  node = np.copy(nodeG)
  nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
  nOuts = len(node[0,node[1,:] == 2])

  # Create connection and initial weight matrices
  conn[3,conn[4,:]==0] = np.nan # disabled but still connected
  src  = conn[1,:].astype(int)
  dest = conn[2,:].astype(int)

  # Reordering node : input, bias, output, hidden 
  reordered_index = np.r_[node[0, node[1, :] ==1], node[0,node[1,:] == 4], node[0,node[1,:] == 2], node[0,node[1,:] == 3]]

  # Get Edge on reordered nodes 
  src_mask = (src.reshape(-1, 1) == reordered_index.reshape(1, -1)) # (n_conn, n_node)
  dest_mask = (dest.reshape(-1, 1) == reordered_index.reshape(1, -1))
  src = (src_mask @ np.arange(len(reordered_index)).reshape(-1, 1)).flatten()  # Convert to 1D
  dest = (dest_mask @ np.arange(len(reordered_index)).reshape(-1, 1)).flatten()  # Convert to 1D

  # Create weight matrix according to reordered nodes
  wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
  wMat[src,dest] = conn[3,:] # assign weight to the connection

  # Get connection matrix (connection between hidden nodes)
  connMat = wMat[nIns+nOuts:,nIns+nOuts:]
  connMat[connMat!=0] = 1

  # Topological Sort of Hidden Nodes (according to connection matrix)
  # Q : sorted "local index" of hidden nodes (smallest index 0)
  edge_in = np.sum(connMat,axis=0) # sum of edges ending with each node
  Q = np.where(edge_in==0)[0]  # Start with nodes with no incoming connections
  for i in range(len(connMat)):
      if (len(Q) == 0) or (i >= len(Q)):
          Q = []
          # return False, False # Cycle found, can't sort
      edge_out = connMat[Q[i],:]
      edge_in  = edge_in - edge_out # Remove nodes' conns from total
      nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
      Q = np.hstack((Q,nextNodes))

      if sum(edge_in) == 0:
          break

  # Add In and outs back and reorder wMat according to sort
  Q += nIns+nOuts # Shifted local index due to reordering (input, bias, output, hidden) 

  Q = np.r_[np.arange(nIns),              
          Q,                              
          np.arange(nIns,nIns+nOuts)]     
  
  return Q, wMat


def getLayer(wMat):
  """Get layer of each node in weight matrix
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1. Input and output nodes are ignored and assigned layer
  0 and max(X)+1 at the end.

  Args:
    wMat  - (np_array) - ordered weight matrix
           [N X N]

  Returns:
    layer - [int]      - layer # of each node

  Todo:
    * With very large networks this might be a performance sink -- especially, 
    given that this happen in the serial part of the algorithm. There is
    probably a more clever way to do this given the adjacency matrix.
  """
  wMat[np.isnan(wMat)] = 0  
  wMat[wMat!=0]=1
  nNode = np.shape(wMat)[0]
  layer = np.zeros((nNode))
  while (True): # Loop until sorting is stable
    prevOrder = np.copy(layer)
    for curr in range(nNode):
      srcLayer=np.zeros((nNode))
      for src in range(nNode):
        srcLayer[src] = layer[src]*wMat[src,curr]   
      layer[curr] = np.max(srcLayer)+1    
    if all(prevOrder==layer):
      break
  return layer-1


# -- ANN Activation ------------------------------------------------------ -- #

def act(weights, aVec, nInput, nOutput, inPattern):
  """Returns FFANN output given a single input pattern
  If the variable weights is a vector it is turned into a square weight matrix.
  
  Allows the network to return the result of several samples at once if given a matrix instead of a vector of inputs:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)

  Args:
    weights   - (np_array) - ordered weight matrix or vector
                [N X N] or [N**2]
    aVec      - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
    nInput    - (int)      - number of input nodes
    nOutput   - (int)      - number of output nodes
    inPattern - (np_array) - input activation
                [1 X nInput] or [nSamples X nInput]

  Returns:
    output    - (np_array) - output activation
                [1 X nOutput] or [nSamples X nOutput]
  """
  # Turn weight vector into weight matrix
  if np.ndim(weights) < 2:
      nNodes = int(np.sqrt(np.shape(weights)[0]))
      wMat = np.reshape(weights, (nNodes, nNodes))
  else:
      nNodes = np.shape(weights)[0]
      wMat = weights
  wMat[np.isnan(wMat)]=0

  # Vectorize input
  if np.ndim(inPattern) > 1:
      nSamples = np.shape(inPattern)[0]
  else:
      nSamples = 1

  # Run input pattern through ANN    
  nodeAct  = np.zeros((nSamples,nNodes)) # Store activation of each node
  nodeAct[:,0] = 1 # Bias activation
  nodeAct[:,1:nInput+1] = inPattern # Prepare input node activation

  # Propagate signal through hidden to output nodes
  for iNode in range(nInput+1,nNodes):
      rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
      nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct) # Looping sparse dot-product to compute each node's activation
  output = nodeAct[:,-nOutput:]   
  return output


def applyAct(actId, x):
  """Returns value after an activation function is applied
  Lookup table to allow activations to be stored in numpy arrays

  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent [tanh] (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine
  case 11 -- Squared

  Args:
    actId   - (int)   - key to look up table
    x       - (???)   - value to be input into activation
              [? X ?] - any type or dimensionality

  Returns:
    output  - (float) - value after activation is applied
              [? X ?] - same dimensionality as input
  """
  if actId == 1:   # Linear
    value = x

  if actId == 2:   # Unsigned Step Function
    value = 1.0*(x>0.0)
    #value = (np.tanh(50*x/2.0) + 1.0)/2.0

  elif actId == 3: # Sin
    value = np.sin(np.pi*x) 

  elif actId == 4: # Gaussian with mean 0 and sigma 1
    value = np.exp(-np.multiply(x, x) / 2.0)

  elif actId == 5: # Hyperbolic Tangent (signed)
    value = np.tanh(x)     

  elif actId == 6: # Sigmoid (unsigned)
    value = (np.tanh(x/2.0) + 1.0)/2.0

  elif actId == 7: # Inverse
    value = -x

  elif actId == 8: # Absolute Value
    value = abs(x)   
    
  elif actId == 9: # Relu
    value = np.maximum(0, x)   

  elif actId == 10: # Cosine
    value = np.cos(np.pi*x)

  elif actId == 11: # Squared
    value = x**2
    
  else:
    value = x

  return value


# -- Action Selection ---------------------------------------------------- -- #

def selectAct(action, actSelect):  
  """Selects action based on vector of actions

    Single Action:
    - Hard: a single action is chosen based on the highest index
    - Prob: a single action is chosen probablistically with higher values
            more likely to be chosen

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned 

  Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

  Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
  """  
  if actSelect == 'softmax':
    action = softmax(action)
  elif actSelect == 'prob':
    action = weightedRandom(np.sum(action,axis=0))
  else:
    action = action.flatten()
  return action

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    Assumes: [samples x dims]

    Args:
      x - (np_array) - unnormalized values
          [samples x dims]

    Returns:
      softmax - (np_array) - softmax normalized in dim 1
    
    Todo: Untangle all the transposes...    
    """    
    if x.ndim == 1:
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=0)
    else:
      e_x = np.exp(x.T - np.max(x,axis=1))
      return (e_x / e_x.sum(axis=0)).T

def weightedRandom(weights):
  """Returns random index, with each choices chance weighted
  Args:
    weights   - (np_array) - weighting of each choice
                [N X 1]

  Returns:
    i         - (int)      - chosen index
  """
  minVal = np.min(weights)
  weights = weights - minVal # handle negative vals
  cumVal = np.cumsum(weights)
  pick = np.random.uniform(0, cumVal[-1])
  for i in range(len(weights)):
    if cumVal[i] >= pick:
      return i
        

# -- File I/O ------------------------------------------------------------ -- #

def exportNet(filename,wMat, aVec):
  indMat = np.c_[wMat,aVec]
  np.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')

def importNet(fileName):
  ind = np.loadtxt(fileName, delimiter=',')
  wMat = ind[:,:-1]     # Weight Matrix
  aVec = ind[:,-1]      # Activation functions

  # Create weight key
  wVec = wMat.flatten()
  wVec[np.isnan(wVec)]=0
  wKey = np.where(wVec!=0)[0] 

  return wVec, aVec, wKey


class LayeredWANN(nn.Module): # ToBeTested
  
    def __init__(self, weights: torch.Tensor, aVec: torch.Tensor, nInput: int, nOutput: int):
        super().__init__()
        
        # Convert numpy arrays to torch if needed
        if not torch.is_tensor(weights):
            weights = torch.from_numpy(weights).float()
        if not torch.is_tensor(aVec):
            aVec = torch.from_numpy(aVec).long()
            
        # Get layer information
        self.node_layers = torch.from_numpy(getLayer(weights.numpy())).long()
        n_layers = self.node_layers.max() + 1
        
        # Create sparse layers
        self.layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            # Get nodes in current and next layer
            curr_mask = self.node_layers == layer_idx
            next_mask = self.node_layers == (layer_idx + 1)
            
            # Extract relevant weights
            layer_weights = weights[curr_mask][:, next_mask]
            
            # Create sparse linear layer
            sparse_layer = SparseLinear(
                weight_matrix=layer_weights
            )
            self.layers.append(sparse_layer)
            
        # Store activation functions per node
        self.aVec = aVec
        self.nInput = nInput
        self.nOutput = nOutput
        
    def forward(self, x):
        # Initial node activations
        batch_size = x.shape[0]
        all_activations = torch.zeros(
            batch_size, 
            len(self.node_layers), 
            device=x.device
        )
        all_activations[:, 0] = 1  # Bias
        all_activations[:, 1:self.nInput+1] = x
        
        # Process each layer
        for layer_idx, layer in enumerate(self.layers):
            curr_mask = self.node_layers == layer_idx
            next_mask = self.node_layers == (layer_idx + 1)
            
            # Forward through sparse layer
            curr_activations = all_activations[:, curr_mask]
            next_raw = layer(curr_activations)
            
            # Apply activation functions
            next_nodes = torch.where(next_mask)[0]
            for i, node_idx in enumerate(next_nodes):
                act_fn = self.aVec[node_idx]
                next_raw[:, i] = self.apply_act(act_fn, next_raw[:, i])
            
            all_activations[:, next_mask] = next_raw
            
        return all_activations[:, -self.nOutput:]
    
    @staticmethod
    def apply_act(act_id, x):
        # Same as before
        return applyAct(act_id, x)

class SparseLinear(nn.Module):
    def __init__(self, weight_matrix):
        super().__init__()
        
        # Create mask for non-zero weights
        self.mask = (weight_matrix != 0)
        
        # Initialize weights using the non-zero values
        weights = torch.zeros_like(weight_matrix)
        weights[self.mask] = weight_matrix[self.mask]
        self.weight = nn.Parameter(weights)
        
    def forward(self, x):
        # Apply mask during forward pass
        masked_weight = self.weight * self.mask
        return torch.matmul(x, masked_weight)