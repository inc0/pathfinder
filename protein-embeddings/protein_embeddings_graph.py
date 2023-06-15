# %%

import glob
import torch
import os
import json
import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from torch_geometric.data import Data, Batch
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.nn import GAE
import torch_geometric.transforms as T
from torch.nn import Linear, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, HingeEmbeddingLoss
from p_tqdm import p_map
from torch_geometric.nn import GAE
from torch_geometric.utils import to_dense_adj
from torch.utils.data import random_split
import time 


all_proteins = glob.glob("/datasets/bigbind/BigBindV1/**/*_graph.json")

# %%

def process_pdb(pdb_file, processed_dir):
    with open(pdb_file, 'r') as f:
        graph = json.load(f)

    # Process node features
    node_features = [[node['atom_number'], node['valence'], node['electronegativity'], node['charge']] for node in graph['nodes']]
    x = torch.tensor(node_features, dtype=torch.float)

    # Process edges and edge weights
    edge_index = torch.tensor([(g[0], g[1]) for g in  graph['edges']], dtype=torch.long).T
    edge_attr = torch.tensor([[g[2]] for g in  graph['edges']], dtype=torch.float16)

    try:
        if max(edge_index[0]) != x.shape[0] - 1:
            return ValueError(f"Incorrect edges {pdb_file}")
    except Exception as e:
        return ValueError(f"Other error {e}")

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    fname = os.path.basename(pdb_file).replace("json", "pt")

    torch.save(data, os.path.join(processed_dir, f"{fname}"))
    return True


class ProteinGraphs(Dataset):
    processed_dir = "/datasets/bigbind/processed"
    
    def __init__(self, root="/datasets/bigbind/", transform=None, pre_transform=None):
        super(ProteinGraphs, self).__init__(root, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        return all_proteins

    @property
    def processed_file_names(self):
        return [f for f in os.listdir(self.processed_dir) if "pre_filter" not in f and "pre_transform" not in f]

    def process(self):
        results = []
        for p in tqdm.tqdm(self.raw_paths):
            results.append(process_pdb(p, self.processed_dir))
        print(f"Failed {len([r for r in results if r is not True])} files")
    
    def len(self):
        return len(self.processed_paths)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data


class GCN_GAT_Autoencoder(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, embedding_dim):
        super(GCN_GAT_Autoencoder, self).__init__()

        # Encoder: Define the GCN and GAT layers
        self.gcn = GCNConv(num_features, hidden_channels)
        self.gat = GATv2Conv(hidden_channels, hidden_channels, edge_dim=1)
        self.gcn2 = GCNConv(hidden_channels, embedding_dim)
        self.enc_linear1 = Linear(hidden_channels, hidden_channels)
        self.enc_linear2 = Linear(hidden_channels, embedding_dim)

        # Decoder: Define the decoder layers
        self.node_linear1 = Linear(embedding_dim, hidden_channels)
        self.node_linear2 = Linear(hidden_channels, hidden_channels)
        self.node_linear3 = Linear(hidden_channels, hidden_channels)
        self.node_linear4 = Linear(hidden_channels, num_features)


    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Encoder: Apply GCN and GAT
        x = self.gcn(x, edge_index)
        x = torch.relu(x)
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        
        # self.embeddings = global_mean_pool(x, batch)

        # # Decoder: Apply decoder layers to get node embeddings
        node_x = self.node_linear1(x)
        node_x = torch.relu(node_x)
        # from ipdb import set_trace; set_trace()

        # # Compute similarity measure (dot product) between all pairs of node embeddings
        node_logits = torch.mm(node_x, node_x.t())

        return node_logits

def train():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model.train()
        epoch_loss = 0
        batch = 0
        for data in tqdm.tqdm(train_loader):
            # Move data to device
            data = data.to(device)
            optimizer.zero_grad()
            # Forward pass
            out = model(data)
            # decoded = model.decode(out, edge_index=data.edge_index)
            adjacency_matrix = to_dense_adj(data.edge_index) #, edge_attr=data.edge_attr)
            # Compute the loss
            loss = loss_fn(out, adjacency_matrix)
            # loss = model.recon_loss(out, pos_edge_index=data.edge_index)
            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
            batch += 1
            optimizer.step()
            if batch % 10 == 0:
                print(f"Loss: {loss}")
        return loss



# %%
if __name__ == "__main__":
    hidden_channels = 2048  # Number of hidden channels
    embedding_dim = 1024  # Dimension of the graph-level embedding
    batch_size = 32

    train_ratio = 0.9  # or any other ratio

    graph_dataset = ProteinGraphs(root='/datasets/bigbind/')

    train_size = int(len(graph_dataset) * train_ratio)
    test_size = len(graph_dataset) - train_size
    print(f"Data split: {train_size} / {test_size}")
    train_dataset, test_dataset = random_split(graph_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # %%

    # Initialize the model and the optimizer
    model = GCN_GAT_Autoencoder(4, hidden_channels, embedding_dim)
    optimizer = Adam(model.parameters(), lr=0.11)
    loss_fn = MSELoss()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    model = model.to(device)

    train_loss = []

    # Training loop
    start_time = time.time()

    print("Training starts")
    for epoch in range(3):
        loss = train()
        train_loss.append(loss)
        print(f"Epoch: {epoch+1}, Loss: {loss} Elapsed {time.time() - start_time}")

    print("Saving model")
    torch.save(model, "./protein_embedding_model.pth")

# %%
