from torch_geometric.datasets import Planetoid
import os

def get_data() -> Planetoid:
    if not os.path.exists("data"):
        os.makedirs("data")
    return Planetoid("data", "CiteSeer")

if __name__ == '__main__':
    dataset = get_data()
    dataset.download()