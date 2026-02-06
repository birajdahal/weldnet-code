# Other models: TimeInput, HighDimProp, LDNet, LDON

import time
import glob
import itertools
import datetime
import copy
import os
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from copy import deepcopy
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler

from .utils import num_params
from .base import (
    FFNet, DeepONet, get_activation, BaseHelper,
    compute_relative_errors, load_model_by_metadata,
    plot_errorparams_shared, train_pca_shared,
    FFAutoencoder, PCAAutoencoder,
)


class TimeInputModel():
    def __init__(self, dataset, ticlass, tiinfo, activation, useparams=False, td=None, seed=0, device=0, residual=0, pcadim=0):
        self.dataset = dataset
        self.device = device
        self.td = td
        self.useparams = useparams
        self.residual = residual

        if self.td is None:
            self.prefix = f"{self.dataset.name}{str(ticlass.__name__)}"
        else:
            self.prefix = self.td

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.timetaken = 0

        datacopy = self.dataset.data.copy()
        self.numtrain = int(datacopy.shape[0] * 0.9)

        self.T = self.dataset.data.shape[1]
        self.trainarr = datacopy[:self.numtrain]
        self.testarr = datacopy[self.numtrain:]

        if self.useparams:
            T = self.trainarr.shape[1]

            paramtrain = np.repeat(self.dataset.params[:self.numtrain, None, :], T, axis=1)
            paramtest = np.repeat(self.dataset.params[self.numtrain:, None, :], T, axis=1)

            self.trainarr = np.concatenate([self.trainarr, paramtrain], axis=-1)
            self.testarr = np.concatenate([self.testarr, paramtest], axis=-1)

        self.ticlass = ticlass
        self.optparams = None

        self.datadim = len(self.dataset.data.shape) - 2
        if len(tiinfo) == 1:
            if self.useparams:
                tiinfo[0][0] += dataset.params.shape[1]

            self.model = self.ticlass(tiinfo[0], activation).to(self.device)
        elif len(tiinfo) == 2:
            assert(not self.useparams)
            self.model = self.ticlass(tiinfo[0], tiinfo[1], activation).to(self.device)
        else:
            raise ValueError(f"tiinfo must have 1 or 2 elements, got {len(tiinfo)}")

        self.pca = False
        self.pcadim = pcadim
        if self.pcadim == 0:
            tiinfo[0][-1] = self.dataset.data.shape[-1]
            self.pcadim = 0
        else:
            self.pcadim = pcadim
            tiinfo[0][-1] = pcadim

        self.metadata = {
            "model_class": ticlass.__name__,
            "tiinfo": tiinfo,
            "activation": activation.__name__ if hasattr(activation, '__name__') else str(activation),
            "dataset_name": dataset.name,
            "data_shape": list(dataset.data.shape),
            "data_checksum": float(np.sum(dataset.data)),
            "seed": seed,
            "useparams": self.useparams
        }

        self.epochs = []

    def train_pca(self, data):
        """Train PCA on data using shared utility."""
        self.pca, errors = train_pca_shared(data, self.pcadim, self.datadim)
        return errors

    def get_ti_errors(self, testarr, ords=(2,), times=None, aggregate=True):
        assert(aggregate or len(ords) == 1)

        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)

        if times is None:
            times = range(self.T - 1)

        allts = torch.linspace(0, 1, self.T)[1:]
        ts = allts[[t - 1 for t in times]]

        out = self.forward(testarr[:, 0], ts)

        n = testarr.shape[0]
        orig = testarr[:, 1:].cpu().detach().numpy()
        out = out.cpu().detach().numpy()

        if self.useparams:
            orig = orig[..., :-self.dataset.params.shape[1]]

        if aggregate:
            orig = orig.reshape([n, -1])
            out = out.reshape([n, -1])
            testerrs = []
            for o in ords:
                testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

            return tuple(testerrs)

        else:
            o = ords[0]
            testerrs = []

            if len(times) == 1:
                t = times[0]
                origslice = orig[:, t-1].reshape([n, -1])
                outslice = out.reshape([n, -1])
                return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
            else:
                for t in range(orig.shape[1]):
                    origslice = orig[:, t].reshape([n, -1])
                    outslice = out[:, t].reshape([n, -1])
                    testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

                return testerrs

    def forward(self, x, ts):
        if isinstance(self.model, FFNet):
            if self.pca:
                x = torch.matmul(x - self.pca["center"], self.pca["tensor"].T)

            origshape = list(x.shape[-self.datadim:])
            x = x.reshape(list(x.shape[:-self.datadim]) + [-1])

            T = np.asarray(ts.cpu()).shape[0]
            x_exp = x.unsqueeze(-2)
            t_exp = ts.reshape(*([1] * (x.dim() - 1)), -1, 1)

            x_brd = x_exp.expand(*x.shape[:-1], T, x.shape[-1])
            t_brd = t_exp.expand(*x.shape[:-1], T, 1)

            xts = torch.cat((x_brd, t_brd), dim=-1)

            out = self.model(xts)

            if self.useparams:
                origshape[-1] -= self.dataset.params.shape[1]

            output = out.reshape(list(out.shape)[:-1] + origshape)

            if self.residual == 2:
                pred = xts[..., :-1] + t_brd * output
            elif self.residual == 1:
                pred = xts[..., :-1] + output
            else:
                pred = output

            if self.pca:
                pred = torch.matmul(out, self.pca["tensor"]) + self.pca["center"]

            return pred

        elif isinstance(self.model, DeepONet):
            B, S = x.shape
            device = x.device

            ts_tensor = torch.as_tensor(ts, device=device)
            spaces = torch.linspace(0, 1, S, device=device)

            s_grid, t_grid = torch.meshgrid(spaces, ts_tensor, indexing='ij')

            inputs = torch.stack((s_grid, t_grid), dim=-1)

            out = self.model(x, inputs)

            if self.pca:
                out = torch.matmul(out - self.pca["center"], self.pca["tensor"].T)

            return out

        else:
            raise NotImplementedError(f"Unsupported model type: {type(self.model)}")

    def train_model(self, epochs, save=False, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False, numts=1):
        def train_epoch(dataloader, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []

            def closure(batch):
                optimizer.zero_grad()
                alltimes = torch.linspace(0, 1, self.T, dtype=batch.dtype).to(self.device)
                ts = random.sample(range(self.T), numts)
                res = 0
                for t in ts:
                    out = self.forward(batch[:, t], alltimes[t+1:])
                    res += loss(batch[:, t+1:], out)
                res /= len(ts)
                res.backward()
                return res

            for batch in dataloader:
                self.trainstep += 1
                error = optimizer.step(lambda: closure(batch))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None and ep > epochs // 2:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_ti_errors(testarr, ords=(1, 2, np.inf))
                print(f"{ep+1}: Train Loss {error:.3e}, Relative TI Error (1, 2, inf): {testerr1:.3f}, {testerr2:.3f}, {testerrinf:.3f}")

            return losses

        loss = nn.MSELoss() if loss is None else loss()
        losses = []
        self.trainstep = 0

        train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
        test = self.testarr

        opt = optim(self.model.parameters(), lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
        dataloader = DataLoader(train, shuffle=False, batch_size=batch)

        if self.optparams is not None:
            opt.load_state_dict(self.optparams)

        print(f"Number of NN trainable parameters: {num_params(self.model)}")
        print(f"Starting training TI model at {time.asctime()}...")

        start = time.time()
        bestdict = {"loss": float(np.inf), "ep": 0}
        for ep in range(epochs):
            lossesN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
            losses += lossesN

            if best and ep > epochs // 2:
                avgloss = np.mean(lossesN)
                if avgloss < bestdict["loss"]:
                    bestdict["model"] = self.model.state_dict()
                    bestdict["opt"] = opt.state_dict()
                    bestdict["loss"] = avgloss
                    bestdict["ep"] = ep

        self.timetaken += time.time() - start
        print(f"Finished training TI model at {time.asctime()}...")

        if best and "model" in bestdict:
            self.model.load_state_dict(bestdict["model"])
            opt.load_state_dict(bestdict["opt"])

        self.optparams = opt.state_dict()
        self.epochs.append(epochs)

        return {"losses": losses}


class TimeInputHelper(BaseHelper):
    """Helper class for TimeInputModel."""

    def create_timeinput(self, dataset, config=None, **args):
        if config is None:
            config = self.config

        assert(len(dataset.data.shape) < 5)
        if len(dataset.data.shape) == 3:
            din = dataset.data.shape[2]
        else:
            din = dataset.data.shape[2] * dataset.data.shape[3]

        td = args.get("td", None)
        seed = args.get("seed", 0)
        device = args.get("device", 0)
        k = args.get("k", None)

        useparams = args.get("useparams", False)
        residual = args.get("residual", 0)

        ticlass = args.get("ticlass", config.ticlass)
        if ticlass == "DeepONet":
            ticlass = DeepONet
            tibranch = deepcopy(args.get("branchseq", config.branchseq))
            titrunk = deepcopy(args.get("trunkseq", config.trunkseq))

            tibranch[0] = din
            titrunk[0] = len(dataset.data.shape) - 2 + 1

            if k is not None:
                tibranch[-1] = k
                titrunk[-1] = k

            tiinfo = (tibranch, titrunk)

        else:
            assert(ticlass == "FFNet")
            ticlass = FFNet
            ffseq = deepcopy(args.get("ffseq", config.ffseq))

            ffseq[0] = din + 1
            ffseq[-1] = din

            tiinfo = (ffseq,)

        activation = get_activation(args.get("activation", config.activation))

        pcadim = 0
        if hasattr(config, "pcadim"):
            pcadim = config.pcadim
        if "pcadim" in args:
            pcadim = args["pcadim"]

        if pcadim > 0:
            tiinfo[0][0] = pcadim + 1
            tiinfo[0][-1] = pcadim

        return TimeInputModel(dataset, ticlass, tiinfo, activation, td=td, seed=seed, device=device, useparams=useparams, residual=residual, pcadim=pcadim)

    @staticmethod
    def get_operrs(ti, times=None, testonly=True):
        if isinstance(ti, TimeInputModel):
            if testonly:
                data = ti.testarr
            else:
                data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

            errors = ti.get_ti_errors(data, times=times, aggregate=False)

        return errors

    @staticmethod
    def plot_op_predicts(ti, testonly=True, xs=None, cmap="viridis"):
        if testonly:
            data = ti.testarr
        else:
            data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

        if xs is None:
            if ti.useparams:
                xs = np.linspace(0, 1, data.shape[2] - ti.dataset.params.shape[1])
            else:
                xs = np.linspace(0, 1, data.shape[2])

        import torch
        data = torch.tensor(np.float32(data)).to(ti.device)

        times = torch.arange(1, ti.T)
        tt = times.to(ti.device) / ti.T
        predicts = ti.forward(data[:, 0], tt)

        predicts = predicts.cpu().detach()
        data = data.cpu().detach()

        if ti.useparams:
            data = data[..., :-ti.dataset.params.shape[1]]

        errors = []
        n = predicts.shape[0]
        for s in times:
            currpredict = predicts[:, s-1].reshape((n, -1))
            currreference = data[:, s].reshape((n, -1))
            errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))

        print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

        if len(data.shape) == 3:
            fig, ax = plt.subplots(figsize=(4, 3))
        elif len(data.shape) == 4:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            fig.subplots_adjust(right=0.90)
            sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

        @widgets.interact(i=(0, n-1), s=(1, ti.T-1))
        def plot_interact(i=0, s=1):
            print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

            if len(data.shape) == 3:
                ax.clear()
                ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s]):.4f}")
                ax.plot(xs, data[i, 0], label="Input", linewidth=1)
                ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
                ax.plot(xs, data[i, s], label="Exact", linewidth=1)
                ax.legend()
            elif len(data.shape) == 4:
                for axx in axes:
                    axx.clear()
                axes[0].imshow(data[i, 0], cmap=cmap)
                axes[0].set_title("Initial")
                axes[1].imshow(data[i, s], cmap=cmap)
                axes[1].set_title("Exact")
                axes[2].imshow(predicts[i, s-1], cmap=cmap)
                axes[2].set_title("Predicted")
                cb = axes[3].imshow(np.abs(predicts[i, s-1] - data[i, s]), cmap=cmap)
                axes[3].set_title("|Difference|")
                fig.colorbar(cb, cax=sub_ax)

    @staticmethod
    def plot_errorparams(ti, param=-1):
        from .base import plot_errorparams_shared
        plot_errorparams_shared(ti, TimeInputHelper.get_operrs, param)


class HighDimProp():
    def __init__(self, dataset, propclass, propseq, activation, autonomous=True, useparams=False, td=None, seed=0, residual=True, device=0):
        self.dataset = dataset
        self.device = device
        self.td = td
        self.useparams = useparams
        self.residual = residual
        self.autonomous = autonomous

        if self.td is None:
            self.prefix = f"HighDimProp{self.dataset.name}{str(propclass.__name__)}"
        else:
            self.prefix = self.td

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.timetaken = 0

        datacopy = self.dataset.data.copy()
        self.numtrain = int(datacopy.shape[0] * 0.9)

        self.T = self.dataset.data.shape[1]
        self.trainarr = datacopy[:self.numtrain]
        self.testarr = datacopy[self.numtrain:]

        if self.useparams:
            T = self.trainarr.shape[1]

            paramtrain = np.repeat(self.dataset.params[:self.numtrain, None, :], T, axis=1)
            paramtest = np.repeat(self.dataset.params[self.numtrain:, None, :], T, axis=1)

            self.trainarr = np.concatenate([self.trainarr, paramtrain], axis=-1)
            self.testarr = np.concatenate([self.testarr, paramtest], axis=-1)

        self.propclass = propclass
        self.propseq = propseq
        self.optparams = None

        self.datadim = len(self.dataset.data.shape) - 2
        if self.useparams:
            propseq[0] += dataset.params.shape[1]

        self.prop = self.propclass(propseq, activation).to(self.device)
        self.propinfo = [propseq, activation.__name__ if hasattr(activation, '__name__') else str(activation), self.residual, self.useparams, self.autonomous]

        self.metadata = {
            "model_class": propclass.__name__,
            "propinfo": self.propinfo,
            "dataset_name": dataset.name,
            "data_shape": list(dataset.data.shape),
            "data_checksum": float(np.sum(dataset.data)),
            "seed": seed,
            "useparams": self.useparams
        }

        self.epochs = []

    def get_errors(self, testarr, ords=(2,), t=None, aggregate=True):
        assert(aggregate or len(ords) == 1)

        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)

        if t is None:
            t = self.T - 1

        out = torch.stack(self.propagate(testarr[:, 0], t), axis=1)

        n = testarr.shape[0]
        orig = testarr[:, 1:].cpu().detach().numpy()
        out = out.cpu().detach().numpy()

        if self.useparams:
            orig = orig[..., :-self.dataset.params.shape[1]]

        if aggregate:
            orig = orig.reshape([n, -1])
            out = out.reshape([n, -1])
            testerrs = []
            for o in ords:
                testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

            return tuple(testerrs)

        else:
            o = ords[0]
            testerrs = []

            if t == 1:
                origslice = orig[:, t].reshape([n, -1])
                outslice = out.reshape([n, -1])
                return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
            else:
                for tt in range(out.shape[1]):
                    origslice = orig[:, tt].reshape([n, -1])
                    outslice = out[:, tt].reshape([n, -1])
                    testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

                return testerrs

    def propagate(self, arr, steps, t=0):
        assert(t + steps < self.T)

        codes = torch.tensor(arr).to(self.device, dtype=torch.float32)

        codeslist = []
        for step in range(steps):
            tcurr = t + 1 + step

            if self.autonomous:
                codeinput = codes
            else:
                ttensor = torch.tensor(np.repeat((tcurr - 1), codes.shape[0])).unsqueeze(1).to(self.device).float()
                codeinput = torch.cat((codes, ttensor), dim=1)

            codes = self.prop_forward(self.prop, codeinput)

            codeslist.append(codes)

        return codeslist

    def prop_forward(self, prop, batch):
        batchbase = batch
        if self.useparams:
            params = batch[..., -self.dataset.params.shape[1]:]
            batchbase = batch[..., :-self.dataset.params.shape[1]]

        out = prop.forward(batch)

        if self.residual:
            out = out + batchbase

        if self.useparams:
            out = torch.cat([out, params], dim=-1)

        return out

    def load_model(self, filename_prefix, verbose=False, min_epochs=0):
        search_path = f"savedmodels/highdimprop/{filename_prefix}*.pickle"
        matching_files = glob.glob(search_path)

        print("Searching for model files matching prefix:", filename_prefix)
        if not hasattr(self, "metadata"):
            raise ValueError("Missing self.metadata. Cannot match models without metadata.")

        for addr in matching_files:
            try:
                with open(addr, "rb") as handle:
                    dic = pickle.load(handle)
            except Exception as e:
                if verbose:
                    print(f"Skipping {addr} due to read error: {e}")
                continue

            meta = dic.get("metadata", {})
            is_match = all(
                meta.get(k) == self.metadata.get(k)
                for k in meta.keys()
            )

            model_epochs = dic["epochs"]
            if model_epochs is None:
                if verbose:
                    print(f"Skipping {addr} due to missing epoch metadata.")
                continue
            elif isinstance(model_epochs, list):
                if sum(model_epochs) < min_epochs:
                    if verbose:
                        print(f"Skipping {addr} due to insufficient epochs")
                    continue
            elif model_epochs < min_epochs:
                if verbose:
                    print(f"Skipping {addr} due to insufficient epochs")
                continue

            if is_match:
                print("Model match found. Loading from:", addr)
                self.prop.load_state_dict(dic["model"])
                self.epochs = model_epochs
                self.timetaken = dic["timetaken"]
                if "opt" in dic:
                    self.optparams = dic["opt"]

                return True
            elif verbose:
                print("Metadata mismatch in file:", addr)

        print("Load failed. No matching models found.")
        return False

    def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, accumulateprop=False, best=True, verbose=False):
        def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []

            def closure(batch):
                optimizer.zero_grad()

                if accumulateprop:
                    outprop = self.propagate(batch[:, 0], self.T-1)
                    propped = torch.stack(outprop, axis=1)
                else:
                    propped = self.prop(batch[:, :-1])

                if self.useparams:
                    batch = batch[..., :-self.dataset.params.shape[1]]

                res = loss(propped, batch[:, 1:])
                res.backward()

                if writer is not None and self.trainstep % 5 == 0:
                    writer.add_scalar("main/loss", res, global_step=self.trainstep)

                return res

            for batch in dataloader:
                self.trainstep += 1
                error = optimizer.step(lambda: closure(batch))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None and ep > epochs // 2:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_errors(testarr, ords=(1, 2, np.inf))
                if scheduler is not None:
                    print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative HDP Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                else:
                    print(f"{ep+1}: Train Loss {error:.3e}, Relative HDP Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

            return losses, [], [], []

        loss = nn.MSELoss() if loss is None else loss()

        losses = []
        self.trainstep = 0

        train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
        test = self.testarr

        opt = optim(self.prop.parameters(), lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
        dataloader = DataLoader(train, shuffle=False, batch_size=batch)

        if self.optparams is not None:
            opt.load_state_dict(self.optparams)

        writer = None
        if self.td is not None:
            name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
            writer = torch.utils.tensorboard.SummaryWriter(name)
            print("Tensorboard writer location is " + name)

        print("Number of NN trainable parameters", num_params(self.prop))
        print(f"Starting training HighDimProp model {self.metadata['model_class']} at {time.asctime()}...")
        print("train", train.shape, "test", test.shape)

        start = time.time()
        bestdict = {"loss": float(np.inf), "ep": 0}
        for ep in range(epochs):
            lossesN, _, _, _ = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
            losses += lossesN

            if best and ep > epochs // 2:
                avgloss = np.mean(lossesN)
                if avgloss < bestdict["loss"]:
                    bestdict["model"] = self.prop.state_dict()
                    bestdict["opt"] = opt.state_dict()
                    bestdict["loss"] = avgloss
                    bestdict["ep"] = ep

        end = time.time()
        self.timetaken += end - start
        print(f"Finished training HighDimProp model at {time.asctime()}...")

        if best:
            self.prop.load_state_dict(bestdict["model"])
            opt.load_state_dict(bestdict["opt"])

        self.optparams = opt.state_dict()
        self.epochs.append(epochs)

        if save:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            readable_shape = "x".join(map(str, self.propseq))

            total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

            filename = (
                f"{self.dataset.name}_"
                f"{self.propclass.__name__}_"
                f"{self.metadata['propinfo'][1]}_"
                f"{readable_shape}_"
                f"{self.seed}_"
                f"{total_epochs}ep_"
                f"{now}.pickle"
            )

            dire = "savedmodels/highdimprop"
            addr = os.path.join(dire, filename)

            if not os.path.exists(dire):
                os.makedirs(dire)

            with open(addr, "wb") as handle:
                pickle.dump({
                    "model": self.prop.state_dict(),
                    "metadata": self.metadata,
                    "opt": self.optparams,
                    "epochs": self.epochs,
                    "timetaken": self.timetaken
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Model saved at", addr)

        return {"losses": losses}


class HighDimPropHelper(BaseHelper):
    """Helper class for HighDimProp."""

    def create_propnet(self, dataset, config=None, **args):
        if config is None:
            config = self.config

        assert(len(dataset.data.shape) < 4)
        if len(dataset.data.shape) == 3:
            din = dataset.data.shape[2]
        else:
            din = dataset.data.shape[2] * dataset.data.shape[3]

        td = args.get("td", None)
        seed = args.get("seed", 0)
        device = args.get("device", 0)

        useparams = args.get("useparams", False)
        residual = args.get("residual", False)
        autonomous = args.get("autonomous", True)

        propclass = args.get("propclass", config.propclass)
        assert(propclass == "FFNet")
        propclass = FFNet

        propseq = deepcopy(args.get("propseq", config.propseq))
        propseq[0] = din
        propseq[-1] = din

        activation = get_activation(args.get("activation", config.activation))

        return HighDimProp(dataset, propclass, propseq, activation, autonomous=autonomous, useparams=useparams, td=td, seed=seed, residual=residual, device=device)

    @staticmethod
    def get_operrs(ti, times=None, testonly=True):
        if testonly:
            data = ti.testarr
        else:
            data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

        if times is not None and len(times) == 1:
            testarr = torch.tensor(data, dtype=torch.float32)
            out = torch.stack(ti.propagate(testarr[:, 0], ti.T - 1), axis=1)
            n = testarr.shape[0]
            orig = testarr[:, 1:].cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            if ti.useparams:
                orig = orig[..., :-ti.dataset.params.shape[1]]
            idx = times[0] - 1
            origslice = orig[:, idx].reshape([n, -1])
            outslice = out[:, idx].reshape([n, -1])
            return np.linalg.norm(origslice - outslice, axis=1) / np.linalg.norm(origslice, axis=1)

        errors = ti.get_errors(data, aggregate=False)
        return errors

    @staticmethod
    def plot_op_predicts(propnet, testonly=False, xs=None, cmap="viridis"):
        if testonly:
            data = propnet.dataset.data[propnet.numtrain:,]
        else:
            data = propnet.dataset.data

        if xs is None:
            xs = np.linspace(0, 1, len(data[0, 0]))

        import torch
        datas = torch.tensor(np.float32(data)).to(propnet.device)
        predicts = torch.stack(propnet.propagate(datas[:, 0], propnet.T-1), axis=1).cpu().detach()

        errors = []
        n = predicts.shape[0]
        for s in range(predicts.shape[1]):
            currpredict = predicts[:, s].reshape((n, -1))
            currreference = data[:, s+1].reshape((n, -1))
            errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))

        print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

        if len(data.shape) == 3:
            fig, ax = plt.subplots(figsize=(4, 3))

        @widgets.interact(i=(0, n-1), s=(1, propnet.T-1))
        def plot_interact(i=0, s=1):
            print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

            if len(data.shape) == 3:
                ax.clear()
                ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s]):.4f}")
                ax.plot(xs, data[i, 0], label="Input", linewidth=1)
                ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
                ax.plot(xs, data[i, s], label="Exact", linewidth=1)
                ax.legend()

    @staticmethod
    def plot_errorparams(propnet, param=-1):
        from .base import plot_errorparams_shared
        plot_errorparams_shared(propnet, HighDimPropHelper.get_operrs, param)


# Latent dynamics model: LDNet (Grid-based Latent Dynamics Network)


class LDHelper(BaseHelper):
    """Helper class for LDNet."""

    def create_ldnet(self, dataset, k, config=None, **args):
        if config is None:
            config = self.config

        assert(len(dataset.data.shape) < 4 or args.get("pcadim", 0) > 0)

        din = dataset.params.shape[-1]
        dout = dataset.data.shape[-1]

        td = args.get("td", None)
        seed = args.get("seed", 0)
        device = args.get("device", 0)

        dynclass = globals()[args.get("dynclass", config.dynclass)]
        dynparams = copy.deepcopy(dict(args.get("dynparams", config.dynparams)))
        decclass = globals()[args.get("decclass", config.decclass)]
        recparams = copy.deepcopy(dict(args.get("recparams", config.recparams)))

        dynparams["seq"][0] = k + din
        dynparams["seq"][-1] = k
        recparams["seq"][0] = k + din
        recparams["seq"][-1] = dout

        pcadim = 0
        if hasattr(config.recparams, "pcadim"):
            pcadim = config.recparams.pcadim
        if "pcadim" in args:
            pcadim = args["pcadim"]

        if pcadim > 0:
            recparams["seq"][-1] = pcadim

        return LDNet(dataset, k, dynclass, dynparams, decclass, recparams, td=td, seed=seed, device=device, pcadim=pcadim)

    @staticmethod
    def get_operrs(ldnet, times=None, testonly=False):
        if testonly:
            data = ldnet.testarr
            params = ldnet.testparams
        else:
            data = np.concatenate((ldnet.trainarr, ldnet.testarr), axis=0)
            params = np.concatenate((ldnet.trainparams, ldnet.testparams), axis=0)

        errors = ldnet.get_errors(data, params, times=times, aggregate=False)

        return errors

    @staticmethod
    def plot_op_predicts(ldnet, testonly=False, xs=None, cmap="viridis"):
        if testonly:
            data = ldnet.dataset.data[ldnet.numtrain:,]
            params = ldnet.dataset.params[ldnet.numtrain:,]
        else:
            data = ldnet.dataset.data
            params = ldnet.dataset.params

        if xs == None:
            xs = np.linspace(0, 1, len(data[0, 0]))

        params = torch.tensor(np.float32(params)).to(ldnet.device)

        predicts = ldnet.propagate(params).cpu().detach()

        errors = []
        n = predicts.shape[0]
        for s in range(data.shape[1]):
            currpredict = predicts[:, s-1].reshape((n, -1))
            currreference = data[:, s].reshape((n, -1))
            errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))

        print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

        if len(data.shape) == 3:
            fig, ax = plt.subplots(figsize=(4, 3))

        @widgets.interact(i=(0, n-1), s=(1, ldnet.T-1))
        def plot_interact(i=0, s=1):
            print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

            if len(data.shape) == 3:
                ax.clear()
                ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
                ax.plot(xs, data[i, 0], label="Input", linewidth=1)
                ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
                ax.plot(xs, data[i, s], label="Exact", linewidth=1)
                ax.legend()

    @staticmethod
    def plot_errorparams(ldnet, param=-1):
        plot_errorparams_shared(ldnet, LDHelper.get_operrs, param)


class LDNet():
    def __init__(self, dataset, k, dynclass, dynparams, decclass, recparams, td, seed, device, dt=1e-2, pcadim=0):
        self.dataset = dataset
        self.device = device
        self.td = td
        self.k = k
        self.f = self.dataset.params.shape[1]

        if self.td is None:
            self.prefix = f"{self.dataset.name}{str(dynclass.__name__)}LDNet-{dynparams['seq'][-1]}"
        else:
            self.prefix = self.td

        self.dt = dt

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.timetaken = 0

        datacopy = self.dataset.data.copy()
        self.numtrain = int(datacopy.shape[0] * 0.9)

        self.T = self.dataset.data.shape[1]
        self.trainarr = datacopy[:self.numtrain]
        self.testarr = datacopy[self.numtrain:]
        self.trainparams = self.dataset.params[:self.numtrain]
        self.testparams = self.dataset.params[self.numtrain:]
        self.optparams = None

        self.datadim = 1  # len(self.dataset.data.shape) - 2

        self.dynclass = dynclass
        self.dynparams = copy.deepcopy(dynparams)
        self.decclass = decclass
        self.recparams = copy.deepcopy(recparams)

        dynparams["seq"][0] = self.k + self.f
        dynparams["seq"][-1] = self.k
        recparams["seq"][0] = self.k + self.f

        self.pca = False
        self.pcadim = pcadim
        if self.pcadim == 0:
            recparams["seq"][-1] = self.dataset.data.shape[-1]
            self.pcadim = 0
        else:
            self.pcadim = pcadim
            recparams["seq"][-1] = pcadim

        self.dynnet = dynclass(**dynparams).float().to(device)
        self.recnet = decclass(**recparams).float().to(device)

        self.metadata = {
            "dynclass": dynclass.__name__,
            "dynparams": dynparams,
            "decclass": decclass.__name__,
            "recparams": recparams,
            "dataset_name": dataset.name,
            "data_shape": list(dataset.data.shape),
            "data_checksum": float(np.sum(dataset.data)),
            "seed": seed,
        }

        self.epochs = []

    def train_pca(self, data):
        """Train PCA on data using shared utility."""
        self.pca, errors = train_pca_shared(data, self.pcadim, self.datadim)
        return errors

    def propagate(self, code, start=0, end=-1, returncodes=False):
        if end == -1:
            end = self.T - 1

        z = code

        # get first decode
        if z.shape[-1] != self.f + self.k:
            if z.shape[-1] == self.f:
                z_fixed = z
                z_dynamic = torch.zeros(list(z_fixed.shape[:-1]) + [self.k], device=z_fixed.device)
                z = torch.cat([z_fixed, z_dynamic], dim=-1)
            else:
                raise ValueError(f"Unexpected code shape: {z.shape}, expected last dim {self.f} or {self.f + self.k}")

        zpreds = [z]
        for t in range(start, end):
            z = self.forward(z)
            zpreds.append(z)

        zpreds = torch.stack(zpreds, dim=1)
        upreds = self.recnet(zpreds)

        if self.pca:
            upreds = torch.matmul(upreds, self.pca["tensor"]) + self.pca["center"]

        if returncodes:
            return upreds, zpreds
        else:
            return upreds

    def get_errors(self, testarr, testparams, ords=(2,), times=None, aggregate=True):
        """Compute relative errors using shared utility."""
        assert(aggregate or len(ords) == 1)

        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)
        if isinstance(testparams, np.ndarray):
            testparams = torch.tensor(testparams, dtype=torch.float32)

        if times is None:
            times = range(self.T-1)

        out = self.propagate(testparams)
        orig = testarr.cpu().detach().numpy()
        out = out.cpu().detach().numpy()

        return compute_relative_errors(orig, out, ords=ords, aggregate=aggregate, times=times)

    def forward(self, z_full, decode=False):
        if z_full.shape[-1] == self.f + self.k:
            z_fixed = z_full[..., :self.f]
            z_dynamic = z_full[..., self.f:]

        else:
            if z_full.shape[-1] == self.f:
                z_fixed = z_full
                z_dynamic = torch.zeros(list(z_fixed.shape[:-1]) + [self.k], device=z_fixed.device)
                z_full = torch.cat([z_fixed, z_dynamic], dim=-1)
            else:
                raise ValueError(f"Unexpected z_full shape: {z_full.shape}, expected last dim {self.f} or {self.f + self.k}")

        deltaz = self.dynnet(z_full)
        z_next_dynamic = z_dynamic + self.dt * deltaz
        z_next_full = torch.cat([z_fixed, z_next_dynamic], dim=-1)

        if decode:
            decoded = self.recnet(z_next_full)

            if self.pca:
                decoded = torch.matmul(decoded, self.pca["tensor"]) + self.pca["center"]

            return decoded, z_next_full

        else:
            return z_next_full

    def load_model(self, filename_prefix, verbose=False, min_epochs=0):
        """Load model using shared utility."""
        if not hasattr(self, "metadata"):
            raise ValueError("Missing self.metadata. Cannot match models without metadata.")

        search_path = f"savedmodels/ldnet/{filename_prefix}*.pickle"
        dic, addr = load_model_by_metadata(search_path, self.metadata, min_epochs, verbose)

        if dic is not None:
            self.dynnet.load_state_dict(dic["dynnet"])
            self.recnet.load_state_dict(dic["recnet"])
            self.pca = dic.get("pca")
            self.epochs = dic["epochs"]
            self.timetaken = dic["timetaken"]
            if "opt" in dic:
                self.optparams = dic["opt"]
            return True
        return False

    def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
        def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None, testparams=None):
            losses = []
            testerrors1 = []
            testerrors2 = []
            testerrorsinf = []

            def closure(values, params):
                optimizer.zero_grad()

                out = self.propagate(params)

                target = values

                res = loss(out, target)
                res.backward()

                if writer is not None and self.trainstep % 5 == 0:
                    writer.add_scalar("main/loss", res, global_step=self.trainstep)

                return res

            for values, params in dataloader:
                self.trainstep += 1
                error = optimizer.step(lambda: closure(values.reshape([values.shape[0], values.shape[1], -1]), params))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None and ep > epochs // 3:
                scheduler.step(np.mean(losses))

            # print test
            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_errors(testarr, testparams, ords=(1, 2, np.inf))
                if scheduler is not None:
                    print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative LDNet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                else:
                    print(f"{ep+1}: Train Loss {error:.3e}, Relative LDNet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

                if writer is not None:
                    writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
                    writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
                    writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

            return losses, testerrors1, testerrors2, testerrorsinf

        loss = nn.MSELoss() if loss is None else loss()

        losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
        self.trainstep = 0

        train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
        params = torch.tensor(self.trainparams, dtype=torch.float32).to(self.device)
        test = self.testarr

        opt = optim(itertools.chain(self.dynnet.parameters(), self.recnet.parameters()), lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
        dataloader = DataLoader(torch.utils.data.TensorDataset(train, params), shuffle=False, batch_size=batch)

        if self.optparams is not None:
            opt.load_state_dict(self.optparams)

        writer = None
        if self.td is not None:
            name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
            writer = torch.utils.tensorboard.SummaryWriter(name)
            print("Tensorboard writer location is " + name)

        if self.pcadim > 0:
            print(f"Training PCA-{self.pcadim} first.")
            testerr1, testerr2, testerrinf = self.train_pca(train.reshape([train.shape[0] * train.shape[1], -1]))
            print(f"Relative PCA Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        print("Number of NN trainable parameters", num_params(self.dynnet), "+", num_params(self.recnet))
        print(f"Starting training LDNet model at {time.asctime()}...")
        print("train", train.shape, "test", test.shape)

        start = time.time()
        bestdict = {"loss": float(np.inf), "ep": 0}
        for ep in range(epochs):
            lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test, testparams=self.testparams)
            losses += lossesN
            testerrors1 += testerrors1N
            testerrors2 += testerrors2N
            testerrorsinf += testerrorsinfN

            if best and ep > epochs // 2:
                avgloss = np.mean(lossesN)
                if avgloss < bestdict["loss"]:
                    bestdict["dynnet"] = self.dynnet.state_dict()
                    bestdict["recnet"] = self.recnet.state_dict()
                    bestdict["opt"] = opt.state_dict()
                    bestdict["loss"] = avgloss
                    bestdict["ep"] = ep
                elif verbose:
                    print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

        end = time.time()
        self.timetaken += end - start
        print(f"Finished training LDNet model at {time.asctime()}...")

        if best:
            self.dynnet.load_state_dict(bestdict["dynnet"])
            self.recnet.load_state_dict(bestdict["recnet"])
            opt.load_state_dict(bestdict["opt"])

        self.optparams = opt.state_dict()
        self.epochs.append(epochs)

        if save:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Compute total training epochs
            total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

            filename = (
                f"{self.dataset.name}_"
                f"{self.dynclass.__name__}_"
                f"{self.dynparams['seq']}_"
                f"{self.decclass.__name__}_"
                f"{self.recparams['seq']}_"
                f"{self.seed}_"
                f"{total_epochs}ep_"
                f"{now}.pickle"
            )

            dire = "savedmodels/ldnet"
            addr = os.path.join(dire, filename)

            if not os.path.exists(dire):
                os.makedirs(dire)

            with open(addr, "wb") as handle:
                pickle.dump({
                    "dynnet": self.dynnet.state_dict(),
                    "recnet": self.recnet.state_dict(),
                    "pca": self.pca,
                    "metadata": self.metadata,
                    "opt": self.optparams,
                    "epochs": self.epochs,
                    "timetaken": self.timetaken
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Model saved at", addr)

        return {"losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf}


# Latent DeepONet (LDON) - Combines autoencoder with DeepONet for operator learning.


class LDONHelper(BaseHelper):
    """Helper class for LDON."""

    def create_ldon(self, dataset, k, p=10, config=None, **args):
        if config is None:
            config = self.config

        assert len(dataset.data.shape) < 4

        td = args.get("td", None)
        seed = args.get("seed", 0)
        device = args.get("device", 0)
        pcadim = args.get("pcadim", 0)

        donparams = copy.deepcopy(dict(args.get("donparams", config.donparams)))
        aeclass_name = args.get("aeclass", config.aeclass)
        ae_classes = {"FFAutoencoder": FFAutoencoder, "PCAAutoencoder": PCAAutoencoder}
        aeclass = ae_classes[aeclass_name] if isinstance(aeclass_name, str) else aeclass_name
        aeparams = copy.deepcopy(dict(args.get("aeparams", config.aeparams)))

        donparams["bseq"][0] = k
        donparams["bseq"][-1] = k * p
        donparams["tseq"][0] = 1
        donparams["tseq"][-1] = p

        return LDON(dataset, k, p, aeclass, aeparams, donparams, td=td, seed=seed, device=device, pcadim=pcadim)

    @staticmethod
    def get_operrs(ldon, times=None, testonly=False):
        if testonly:
            data = ldon.dataset.data[ldon.numtrain:]
        else:
            data = ldon.dataset.data

        inputs = torch.tensor(data[:, 0], dtype=torch.float32).to(ldon.device)
        encode = ldon.aenet.encode(inputs)
        errors = ldon.get_don_errors(encode, times=times, aggregate=False)

        return errors


class LDON:
    """
    Latent DeepONet - Autoencoder + DeepONet for operator learning.

    The model encodes the initial condition to a latent space,
    then uses a DeepONet (branch + trunk networks) to predict
    the latent trajectory, which is decoded back to the full space.
    """

    def __init__(self, dataset, k, p, aeclass, aeparams, donparams, td, seed, device, pcadim=0):
        self.dataset = dataset
        self.device = device
        self.td = td
        self.k = k
        self.p = p

        self.timetaken = 0

        if self.td is None:
            self.prefix = f"{self.dataset.name}{str(aeclass.__name__)}LDON"
        else:
            self.prefix = self.td

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        datacopy = self.dataset.data.copy()
        self.numtrain = int(datacopy.shape[0] * 0.9)

        self.T = self.dataset.data.shape[1]
        self.trainarr = datacopy[:self.numtrain]
        self.testarr = datacopy[self.numtrain:]
        self.optparams = None

        self.datadim = len(self.dataset.data.shape) - 2
        self.aestep = 0
        self.donstep = 0

        aeparams = copy.deepcopy(aeparams)
        donparams = copy.deepcopy(donparams)

        aeparams["encodeSeq"][0] = self.dataset.data.shape[-1]
        aeparams["encodeSeq"][-1] = self.k
        aeparams["decodeSeq"][0] = self.k
        aeparams["decodeSeq"][-1] = self.dataset.data.shape[-1]

        donparams["bseq"][0] = self.k
        donparams["bseq"][-1] = self.k * self.p
        donparams["tseq"][0] = 1
        donparams["tseq"][-1] = p

        aeparams["pcadim"] = pcadim
        self.aeclass = aeclass
        self.aeparams = aeparams
        self.donparams = donparams

        self.aenet = aeclass(**aeparams).float().to(device)

        branch = FFNet(donparams["bseq"], donparams["bactivation"]).float().to(device)
        trunk = FFNet(donparams["tseq"], donparams["tactivation"]).float().to(device)
        bias = nn.Parameter(torch.zeros(k).to(device))

        self.don = {"branch": branch, "trunk": trunk, "bias": bias}

        self.metadata = {
            "aeclass": aeclass.__name__,
            "aeparams": aeparams,
            "donparams": donparams,
            "dataset_name": dataset.name,
            "data_shape": list(dataset.data.shape),
            "data_checksum": float(np.sum(dataset.data)),
            "seed": seed,
        }

        self.epochs = []

    def reconstruct(self, z, ts):
        """Reconstruct latent trajectory using DeepONet."""
        B = z.shape[0]

        branch_out = self.don["branch"](z)
        branch_out = branch_out.reshape([B, self.k, -1])

        ts = ts.reshape([len(ts), 1])
        trunk_out = self.don["trunk"](ts)

        z_trajectory = torch.einsum('bkp,tp->btk', branch_out, trunk_out) + self.don["bias"]

        return z_trajectory

    def propagate(self, code, start=1, end=-1):
        """Propagate latent code through time."""
        fullts = torch.linspace(0, 1, self.T).float().to(self.device)

        if end > 0:
            ts = fullts[start:end + 1]
        else:
            ts = fullts[start:]

        out = self.reconstruct(code, ts)
        return out

    def get_op_errors(self, testarr, ords=(2,), times=None, aggregate=False):
        """Get operator errors (full pipeline: encode -> propagate -> decode)."""
        assert aggregate or len(ords) == 1

        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)

        if times is None:
            times = range(self.T - 1)

        codes = self.aenet.encode(testarr)
        init = codes[:, 0].detach()

        out = self.aenet.decode(self.propagate(init))

        n = init.shape[0]
        orig = testarr[:, 1:].cpu().detach().numpy()
        out = out.cpu().detach().numpy()

        if aggregate:
            orig = orig.reshape([n, -1])
            out = out.reshape([n, -1])
            testerrs = []
            for o in ords:
                testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))
            return tuple(testerrs)
        else:
            o = ords[0]
            testerrs = []

            if len(times) == 1:
                t = times[0]
                origslice = orig[:, t - 1].reshape([n, -1])
                outslice = out[:, t - 1].reshape([n, -1])
                return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
            else:
                for t in range(orig.shape[1]):
                    origslice = orig[:, t].reshape([n, -1])
                    outslice = out[:, t].reshape([n, -1])
                    testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))
                return testerrs

    def get_don_errors(self, testarr, ords=(2,), times=None, aggregate=True):
        """Get DeepONet errors (latent space only)."""
        assert aggregate or len(ords) == 1

        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)

        testinit = testarr[:, 0]
        testrest = testarr[:, 1:]
        if times is None:
            times = range(self.T - 1)

        out = self.propagate(testinit)

        n = testinit.shape[0]
        orig = testrest.cpu().detach().numpy()
        out = out.cpu().detach().numpy()

        if aggregate:
            orig = orig.reshape([n, -1])
            out = out.reshape([n, -1])
            testerrs = []
            for o in ords:
                testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))
            return tuple(testerrs)
        else:
            o = ords[0]
            testerrs = []

            if len(times) == 1:
                t = times[0]
                origslice = orig[:, t - 1].reshape([n, -1])
                outslice = out[:, t - 1].reshape([n, -1])
                return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
            else:
                for t in range(orig.shape[1]):
                    origslice = orig[:, t].reshape([n, -1])
                    outslice = out[:, t].reshape([n, -1])
                    testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))
                return testerrs

    def get_ae_errors(self, testarr, ords=(2,)):
        """Get autoencoder reconstruction errors."""
        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)

        N = testarr.shape[0]
        T = testarr.shape[1]
        out = self.aenet(testarr).cpu().detach().numpy().reshape([N * T, -1])
        orig = testarr.cpu().detach().numpy().reshape([N * T, -1])

        testerrs = []
        for o in ords:
            testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

        return tuple(testerrs)

    def train_aenet(self, epochs, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
        """Train the autoencoder component."""
        def aenet_epoch(dataloader, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []

            def closure(codes):
                optimizer.zero_grad()
                out = self.aenet(codes)
                target = codes
                res = loss(out, target)
                res.backward()
                return res

            for codes in dataloader:
                self.aestep += 1
                error = optimizer.step(lambda: closure(codes))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None and ep > epochs // 2:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_ae_errors(testarr, ords=(1, 2, np.inf))
                print(f"{ep + 1}: Train Loss {error:.3e}, Relative AE Error (1, 2, inf): {testerr1:.4f}, {testerr2:.4f}, {testerrinf:.4f}")

            return losses

        loss = nn.MSELoss() if loss is None else loss()
        losses = []

        train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
        test = torch.tensor(self.testarr, dtype=torch.float32).to(self.device)

        if hasattr(self.aenet, 'pcadim') and self.aenet.pcadim > 0:
            print("Training PCA preprocessing...")
            testerr1, testerr2, testerrinf = self.aenet.train_pca(train.reshape([train.shape[0] * train.shape[1], -1]))
            print(f"PCA relative error (1, 2, inf): {testerr1:.4f}, {testerr2:.4f}, {testerrinf:.4f}")

        opt = optim(self.aenet.parameters(), lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.3)
        dataloader = DataLoader(train, shuffle=False, batch_size=batch)

        if self.optparams is not None:
            opt.load_state_dict(self.optparams)

        print("Number of AE trainable parameters:", num_params(self.aenet))
        print(f"Starting training LDON AE at {time.asctime()}...")
        start = time.time()

        bestdict = {"loss": float(np.inf), "ep": 0}
        for ep in range(epochs):
            lossesN = aenet_epoch(dataloader, optimizer=opt, scheduler=scheduler, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
            losses += lossesN

            if best and ep > epochs // 2:
                avgloss = np.mean(lossesN)
                if avgloss < bestdict["loss"]:
                    bestdict["aenet"] = self.aenet.state_dict()
                    bestdict["opt"] = opt.state_dict()
                    bestdict["loss"] = avgloss
                    bestdict["ep"] = ep

        print(f"Finished training LDON AE at {time.asctime()}...")
        self.timetaken += time.time() - start

        if best and "aenet" in bestdict:
            self.aenet.load_state_dict(bestdict["aenet"])
            opt.load_state_dict(bestdict["opt"])

        return {"losses": losses}

    def train_don(self, epochs, save=False, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, verbose=False):
        """Train the DeepONet component."""
        def don_epoch(dataloader, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []

            def closure(codes):
                optimizer.zero_grad()
                initial = codes[:, 0]
                out = self.propagate(initial)
                target = codes[:, 1:]
                res = loss(out, target)
                res.backward()
                return res

            for (codes,) in dataloader:
                self.donstep += 1
                error = optimizer.step(lambda: closure(codes))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None and ep > epochs // 2:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_don_errors(testarr, ords=(1, 2, np.inf))
                print(f"{ep + 1}: Train Loss {error:.3e}, Relative DON Error (1, 2, inf): {testerr1:.4f}, {testerr2:.4f}, {testerrinf:.4f}")

            return losses

        assert self.aestep > 0, "Must train autoencoder first (call train_aenet)"

        loss = nn.MSELoss() if loss is None else loss()
        losses = []

        trains = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
        encoded = self.aenet.encode(trains).detach()

        tests = torch.tensor(self.testarr, dtype=torch.float32).to(self.device)
        encodedtest = self.aenet.encode(tests).detach()

        branch = self.don["branch"]
        trunk = self.don["trunk"]
        bias = self.don["bias"]
        opt = optim(list(branch.parameters()) + list(trunk.parameters()) + [bias], lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
        dataloader = DataLoader(TensorDataset(encoded), shuffle=False, batch_size=batch)

        print("Number of DON trainable parameters:", num_params(branch) + num_params(trunk))
        print(f"Starting training LDON DON at {time.asctime()}...")
        start = time.time()

        for ep in range(epochs):
            lossesN = don_epoch(dataloader, optimizer=opt, scheduler=scheduler, ep=ep, printinterval=printinterval, loss=loss, testarr=encodedtest)
            losses += lossesN

        print(f"Finished training LDON DON at {time.asctime()}...")
        self.timetaken += time.time() - start

        self.optparams = opt.state_dict()
        self.epochs.append(epochs)

        return {"losses": losses}
