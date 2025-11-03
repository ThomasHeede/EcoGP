import os.path
import warnings

import torch

from torch.utils.data import DataLoader, Dataset


class DataSampler(Dataset):
    def __init__(self, data):
        self.__dict__.update(data.__dict__)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return idx

    def get_batch_data(self, idx):
        batch = {
            # Metadata
            "device": self.device,
            "n_samples_total": self.n_samples,
            "n_samples_batch": len(idx),
            "n_species": self.n_species,
            "n_env": self.n_env,
            # Data
            "X": self.get_X(idx).to(self.device),
            "Y": self.get_Y(idx).to(self.device),
        }
        # if self.using_coordinates:
        #     unique_locs_idx, reverse = self.get_dist_idx_reverse(idx)
        #     # Data
        #     batch.update({
        #         "coords": self.coords[unique_locs_idx].to(self.device)
        #     })
        #     # Metadata
        #     batch.update({
        #         "n_locs_batch": batch.get("coords").shape[0],
        #         "unique_batch_locs": unique_locs_idx,
        #         "batch_inverse": reverse,
        #     })
        if self.using_coordinates:
            # Data
            batch.update({
                "coords": self.coords[idx].to(self.device)
            })

        if self.using_traits:
            batch.update({"traits": self.traits.to(self.device)})
            batch.update({
                "n_traits": batch.get("traits").shape[1],
            })

        return batch

    def get_dist_batch(self, idx):
        return self.coords_inverse_indicies[idx]

    def get_dist_idx_reverse(self, idx):
        batch = self.get_dist_batch(idx)
        unique, reverse = torch.unique(batch, dim=0, return_inverse=True)
        return unique, reverse

    def get_X(self, idx=None):
        if idx is None:
            return self.X
        else:
            return self.X[idx].to(self.device)

    def get_Y(self, idx=None):
        if idx is None:
            return self.Y
        else:
            return self.Y[idx].to(self.device)


if __name__ == "__main__":
    from DataLoad import DataLoad

    Y_path = "../data/clean/Y.csv"
    X_path = "../data/clean/X.csv"
    coords_path = "../data/clean/XY.csv"
    total_counts_path = "../data/clean/total_counts.csv"
    device = torch.device("cpu")
    normalize_X = True
    batch_size = 50

    data = DataLoad(Y_path, X_path, coords_path, device, normalize_X, verbose=True, total_counts_path=total_counts_path)
    datasampler = DataSampler(data)
