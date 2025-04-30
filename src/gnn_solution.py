import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data, DataLoader
import chess
import chess.pgn


def board_to_graph(board):
    node_feats = []
    idx_of = {}
    for idx, (sq, piece) in enumerate(board.piece_map().items()):
        # one-hot type + color bit
        type_oh   = [1 if piece.piece_type == t else 0 for t in range(1,7)]
        color_bit = [1] if piece.color else [0]
        node_feats.append(type_oh + color_bit)
        idx_of[sq] = idx

    x = torch.tensor(node_feats, dtype=torch.float)

    us, vs, labels = [], [], []
    for u_sq, u_idx in idx_of.items():
        attacks = board.attacks(u_sq)
        for v_sq, v_idx in idx_of.items():
            if attacks & chess.BB_SQUARES[v_sq]:
                u_col = board.piece_at(u_sq).color
                v_col = board.piece_at(v_sq).color
                lab   = 0 if u_col == v_col else 1
                us.append(u_idx); vs.append(v_idx); labels.append(lab)

    edge_index = torch.tensor([us, vs], dtype=torch.long)
    edge_attr  = torch.tensor(labels,  dtype=torch.long)
    return x, edge_index, edge_attr

def pgn_to_data_list(pgn_path):
    """Read first game in PGN, return list of Data objects (one per move)."""
    data_list = []
    board_list = []

    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)
    board = game.board()
    board_list.append(board.copy())  # initial position

    for move in game.mainline_moves():
        board.push(move)
        x, ei, ea = board_to_graph(board)
        data_list.append(Data(x=x, edge_index=ei, edge_attr=ea))
        board_list.append(board.copy())

    return data_list, board_list


class GNNAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_dim)
        self.conv2   = GCNConv(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # defend, attack, none
        )

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return h

    def decode(self, h, edge_index):
        h_u = h[edge_index[0]]
        h_v = h[edge_index[1]]
        return self.decoder(torch.cat([h_u, h_v], dim=1))

    def forward(self, x, edge_index, batch):
        h = self.encode(x, edge_index)
        z = global_add_pool(h, batch)
        return h, z

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        h, _ = model(data.x, data.edge_index, data.batch)

        pos_edge = data.edge_index
        neg_edge = negative_sampling(
            edge_index=pos_edge,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge.size(1)
        )

        logits_pos = model.decode(h, pos_edge)
        logits_neg = model.decode(h, neg_edge)

        labels_pos = data.edge_attr.long()
        labels_neg = torch.full((neg_edge.size(1),),
                                2, dtype=torch.long, device=device)

        logits = torch.cat([logits_pos, logits_neg], dim=0)
        labels = torch.cat([labels_pos, labels_neg],   dim=0)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def tipping_point(model, data_seq, device):
    model.eval()
    zs = []
    for data in data_seq:
        data = data.to(device)
        _, z = model(data.x, data.edge_index, data.batch)
        zs.append(z)
    diffs = [torch.norm(zs[i+1] - zs[i]).item()
             for i in range(len(zs)-1)]
    return max(range(len(diffs)), key=lambda i: diffs[i]) + 1

if __name__ == "__main__":
    pgn_file = "./data/lichess_db_standard_rated_2013-01.pgn"
    data_list, board_list = pgn_to_data_list(pgn_file)

    loader = DataLoader(data_list, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = data_list[0].num_node_features
    model  = GNNAutoEncoder(in_dim, hidden_dim=64).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        loss = train_epoch(model, loader, opt, device)
        print(f"Epoch {epoch:02d}  Loss: {loss:.4f}")

    tip_idx = tipping_point(model, data_list, device)
    print(f"Tipping point occurs at move #{tip_idx}")

    print("\n=== TIPPING POINT DETECTED ===")
    before_idx = tip_idx - 1
    after_idx  = tip_idx
    print(f" Move #{before_idx}:")
    print(board_list[before_idx])
    print(f"\n Move #{after_idx}:")
    print(board_list[after_idx])