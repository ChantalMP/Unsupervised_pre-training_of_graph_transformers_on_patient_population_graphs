import torch


class FeatAccuracy:

    def eval(self, input_dict, margin=0):
        assert 'y_pred' in input_dict and 'y_true' in input_dict

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.from_numpy(y_true)

        assert (y_true.numel() == y_pred.numel())
        # Flatten to one dimension
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        return {'acc': int((abs(y_true - y_pred) <= margin).sum()) / y_true.numel()}


class RegMAE:

    def eval(self, input_dict):
        assert ('y_pred' in input_dict)
        assert ('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor))
        assert (y_true.shape == y_pred.shape)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
