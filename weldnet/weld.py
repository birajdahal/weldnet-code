# WeldNet and related classes

import time
import glob
import datetime
import copy
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import matplotlib.cm as cm
import ruptures as rpt
from concurrent.futures import ThreadPoolExecutor

from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from omegaconf import OmegaConf

from .utils import num_params
from .base import (
    FFNet, get_activation, determine_param,
    JC_Modules, PCAAutoencoder, FFAutoencoder, FFVAE, Other_Modules
)


class WindowTrajectory():
    def find_window(self, t, left=True):
        foundw = []
        for (w, wvs) in enumerate(self.windowvals):
            if t in wvs:
                foundw.append(w)

        if len(foundw) > 0:
            return min(foundw) if left else max(foundw)
        else:
            return False

    def decode_window(self, w, tensor):
        if not isinstance(self.aes[w], PCAAutoencoder):
            tensor = tensor.to(next(self.aes[w].parameters()).device)
        return self.aes[w].decode(tensor)

    def project_window(self, w, tensor):
        return self.decode_window(w, self.encode_window(w, tensor))

    def encode_window(self, w, tensor):
        if not isinstance(self.aes[w], PCAAutoencoder):
            tensor = tensor.to(next(self.aes[w].parameters()).device)
        return self.encode_model(self.aes[w], tensor)


class WeldNet(WindowTrajectory):
    def determine_windows(self, alg="uniform"):
        if alg == "uniform":
            total = self.T + self.W - 1
            M = total // self.W
            remainder = total - self.W * M

            left = [list(range(k*(M-1), (k+1)*(M-1) + 1)) for k in range(self.W - remainder)]
            start = left[-1][-1]
            right = [list(range(start + k*(M), start + (k+1)*(M) + 1)) for k in range(remainder)]
            return left + right

        elif alg == "change-l2":
            wvals = np.asarray([rpt.Dynp(model="l2").fit(x).predict(n_bkps=self.W-1)[:-1] for x in self.alltrain])
            bkpts = [0] + np.median(wvals, axis=0).astype(int).tolist() + [self.T-1]

            windowvals = [list(range(bkpts[i], bkpts[i+1]+1)) for i in range(len(bkpts)-1)]
            return windowvals

    def __init__(self, dataset, windows, aeclass, aeparams, propclass, propparams, transclass, transparams, dynamicwindow=False, td=None, seed=0, device=0, decodedprop=False, accumulateprop=False, tiprop=False, autonomous=True):
        self.dataset = dataset
        self.device = device
        self.td = td

        if self.td is None:
            self.prefix = f'{self.dataset.name}{str(aeclass.__name__)}-{propparams["seq"][-1]}-{"auton" if autonomous else "nonauton"}'
        else:
            self.prefix = self.td

        self.autonomous = autonomous
        self.residualprop = True
        self.tiprop = tiprop

        self.accumulateprop = accumulateprop
        assert(autonomous)

        self.decodedprop = decodedprop

        if not self.autonomous:
            propparams["seq"] = list(propparams["seq"])
            propparams["seq"][0] = propparams["seq"][-1] + 1

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.seed = seed

        datacopy = self.dataset.data.copy()
        self.numtrain = int(datacopy.shape[0] * 0.9)

        self.T = self.dataset.data.shape[1]
        self.W = windows

        self.aes = []
        self.props = []
        self.trains = []
        self.tests = []

        self.timetaken = 0

        self.alltrain = datacopy[:self.numtrain]
        self.alltest = datacopy[self.numtrain:]

        self.paramtrain = self.dataset.params[:self.numtrain]
        self.paramtest = self.dataset.params[self.numtrain:]

        self.transstep = 0
        self.propstep = 0

        aeparams["datadim"] = len(self.dataset.data.shape) - 2

        if "pcadim" in aeparams and aeparams["pcadim"] == 0:
            del aeparams["pcadim"]

        self.aedata = [aeclass.__name__, aeparams, windows]
        self.propdata = [propclass.__name__, propparams, windows, self.autonomous, self.residualprop, self.accumulateprop, self.decodedprop]

        self.aeepochs = []
        self.propepochs = []
        self.transepochs = []

        if transclass is not None and windows > 1:
            self.transcoderdata = [transclass.__name__, transparams, windows, self.residualprop]
        else:
            self.transcoderdata = None

        if dynamicwindow:
            self.windowvals = self.determine_windows("change-l2")
        else:
            self.windowvals = self.determine_windows("uniform")

        print("Windows:", [[x[0], x[-1]] for x in self.windowvals])

        self.transcoders = []

        self.aeclass = aeclass
        self.aeparams = aeparams
        self.propclass = propclass
        self.propparams = propparams
        self.transclass = transclass
        self.transparams = transparams

        self.metadata = {
            "aeinfo": self.aedata,
            "propinfo": self.propdata,
            "transinfo": self.transcoderdata,
            "trainedtogether": False,
            "dataset_name": dataset.name,
            "data_shape": list(dataset.data.shape),
            "data_checksum": float(np.sum(dataset.data)),
            "seed": seed,
        }
        self.epochs = []

    def transcode(self, t, codes):
        w = self.find_window(t)
        assert(self.windowvals[w][-1] == t)

        if len(self.transcoders) == 0:
            decoded = self.decode_window(w, codes)
            codes = self.encode_window(w+1, decoded)
            print(f"Default transcoding {w} to {w+1} at time {t}")
        else:
            if isinstance(self.transcoders[w], nn.Module):
                outs = self.transcoders[w](codes)

            else:
                if torch.is_tensor(codes):
                    codes_np = codes.detach().cpu().numpy()
                    outs = torch.tensor(self.transcoders[w].predict(codes_np), dtype=torch.float32, device=codes.device)

                else:
                    outs = self.transcoders[w].predict(codes)

            if self.residualprop:
                codes = codes + outs
            else:
                codes = outs

        return torch.tensor(codes, dtype=torch.float32)

    def propagate(self, arr, t, steps, arrencoded=False, fixedw=None, return_codes=True):
        assert(t + steps < self.T)

        if fixedw is not None:
            w = fixedw
        else:
            w = self.find_window(t)

        inputt = torch.tensor(arr).to(self.device, dtype=torch.float32)

        if arrencoded:
            codes = inputt
        else:
            codes = self.encode_window(w, inputt)

        codeslist = []
        wprev = w

        if self.tiprop:
            times = []
            startt = t
            for step in range(steps):
                tcurr = t + 1 + step

                if fixedw is not None:
                    wcurr = w
                else:
                    wcurr = self.find_window(tcurr)

                if wcurr != wprev and tcurr-1-startt > 0:
                    times.append((tcurr-1-startt, wcurr))

                wprev = wcurr

            if len(times) == 0 or (len(times) > 0 and times[-1] != (tcurr-startt, wcurr)):
                times.append((tcurr-startt, wcurr))

            for i, (amount, ww) in enumerate(times):
                if ww > 1 and i < len(times) - 1:
                    codes = self.transcode(self.windowvals[ww-1][-1], codes)

                out = self.prop_forward(self.props[ww], codes, ts=torch.arange(1, amount+1)/self.T)

                out = list(torch.unbind(out, dim=1))
                codeslist = codeslist + out
                codes = out[-1]

        else:
            for step in range(steps):
                tcurr = t + 1 + step

                if fixedw is not None:
                    wcurr = w
                else:
                    wcurr = self.find_window(tcurr)

                if wcurr != wprev:
                    codes = self.transcode(tcurr-1, codes)
                    inputt = codes

                if self.autonomous:
                    codeinput = codes
                else:
                    ttensor = torch.tensor(np.repeat((tcurr*0 - 1), codes.shape[0])).unsqueeze(1).to(self.device).float()
                    codeinput = torch.cat((codes, ttensor), dim=1)

                codes = self.prop_forward(self.props[wcurr], codeinput)

                wprev = wcurr
                codeslist.append(codes)

        return codeslist

    def get_proj_errors(self, model, testarr, ords=(2,)):
        if isinstance(testarr, np.ndarray):
            testarr = torch.tensor(testarr, dtype=torch.float32)

        if not isinstance(model, PCAAutoencoder):
            testarr = testarr.to(next(model.parameters()).device)

        proj = model.decode(model.encode(testarr))

        if len(testarr.shape) > 3:
            assert(len(testarr.shape) == 4)
            testarr = testarr.reshape(list(testarr.shape[:-2]) + [-1])
            proj = proj.reshape(list(proj.shape[:-2]) + [-1])

        n = testarr.shape[0]
        testarr = testarr.cpu().detach().numpy().reshape([n, -1])
        proj = proj.cpu().detach().numpy().reshape([n, -1])

        testerrs = []

        for o in ords:
            testerro = np.mean(np.linalg.norm(testarr - proj, axis=1, ord=o) / np.linalg.norm(testarr, axis=1, ord=o))
            testerrs.append(testerro)

        return tuple(testerrs)

    def encode_model(self, model, batch):
        return model.encode(batch)

    def prop_forward(self, prop, batch, ts=None):
        batchbase = batch

        if ts is not None and self.tiprop:
            z_shape = batchbase.shape
            *leading_dims, N = z_shape
            T = ts.shape[0]

            z_expanded = batchbase.unsqueeze(-2).expand(*leading_dims, T, N)

            t_shape = [1] * len(leading_dims) + [T, 1]
            t_expanded = ts.view(*t_shape).expand(*leading_dims, T, 1)

            batch = torch.cat([z_expanded, t_expanded], dim=-1)
            batchbase = z_expanded

        out = prop.forward(batch)

        if self.residualprop:
            out = out + batchbase

        return out

    def trans_forward(self, trans, batch):
        return trans.forward(batch)

    def train_aes(self, epochs_first, warmstart_epochs=0, onlydecoder=False, roll=False, optim=torch.optim.AdamW, lr=1e-4, plottb=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):
        def ae_epoch(model, dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []
            testerrors1 = []
            testerrors2 = []
            testerrorsinf = []

            device = self.device

            def closure(batch):
                optimizer.zero_grad()

                if roll:
                    rot = np.random.randint(0, batch.shape[-1])
                    batch = torch.roll(batch, shifts=rot, dims=-1)

                if isinstance(model, FFVAE):
                    recon, mu, logvar = model(batch, variance=True)
                    self.reconerr, self.kld = model.loss_function(recon, batch, mu, logvar)

                    res = self.reconerr + self.kld
                    res.backward()

                else:
                    enc = self.encode_model(model, batch)

                    proj = model.decode(enc)

                    res = loss(batch, proj)
                    totalloss = res

                    totalloss.backward()

                if writer is not None and self.aestep % 5:
                    writer.add_scalar("main/loss", float(res.cpu().detach()), global_step=self.aestep)

                return res

            for batch in dataloader:
                self.aestep += 1
                error = optimizer.step(lambda: closure(batch))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None and ep > epochs_first // 2:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_proj_errors(model, testarr, ords=(1, 2, np.inf))

                if isinstance(model, FFVAE):
                    prefix = f"{ep+1}: Train Loss {self.reconerr:.3e} + {self.kld:.3e}"
                else:
                    prefix = f"{ep+1}: Train Loss {error:.3e}"

                if scheduler is not None:
                    print(f"{prefix}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                else:
                    print(f"{prefix}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

                if writer is not None:
                    writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
                    writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
                    writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

            return losses, testerrors1, testerrors2, testerrorsinf

        loss = nn.MSELoss() if loss is None else loss()
        encoding_param = determine_param(self.dataset, encoding_param)

        losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

        start = time.time()
        print(f"Training {self.W} WeldNet AEs")
        self.trains = []
        self.tests = []
        for w in range(self.W):
            if len(self.aes) <= w:
                self.aes.append(self.aeclass(**self.aeparams) if self.aeclass not in Other_Modules else self.aeclass(self.aeparams.copy()))

            ae = self.aes[w]

            losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
            bestdict = {"loss": float(np.inf), "ep": 0}

            self.aestep = 0
            epochs = epochs_first
            train = torch.tensor(self.alltrain[:, self.windowvals[w]], dtype=torch.float32)
            test = self.alltest[:, self.windowvals[w]]

            if isinstance(ae, PCAAutoencoder):
                ae.train_pca(train.cpu().numpy())
                testerr1, testerr2, testerrinf = self.get_proj_errors(ae, test, ords=(1, 2, np.inf))
                print(f"Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

                continue

            if ae.pcadim > 0:
                print("Going to reshape", train.shape)
                testerr1, testerr2, testerrinf = ae.train_pca(train.reshape([train.shape[0] * train.shape[1], -1]))
                print(f"PCA for preprocessing has relative error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

            self.trains.append(train)
            self.tests.append(test)

            if onlydecoder:
                trainparams = ae.decoder.parameters()
            else:
                trainparams = ae.parameters()

            opt = optim(trainparams, lr=lr, weight_decay=ridge)
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
            dataloader = DataLoader(train, shuffle=False, batch_size=batch)

            writer = None
            if self.td is not None:
                name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-weld{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
                writer = torch.utils.tensorboard.SummaryWriter(name)
                print("Tensorboard writer location is " + name)

            print("Number of NN trainable parameters", num_params(ae))
            print(f"Starting training WeldNet AE {w+1}/{self.W} ({self.windowvals[w][0]}->{self.windowvals[w][-1]}) at {time.asctime()}...")

            print("train", train.shape, "test", test.shape)

            if warmstart_epochs > 0 and w > 0:
                epochs = warmstart_epochs
                state = self.aes[w-1].state_dict()
                ae.load_state_dict(state)
                print(f"Warm started model {w} from model {w-1}")

            self.aestep = 0
            for ep in range(epochs):
                lossesN, testerrors1N, testerrors2N, testerrorsinfN = ae_epoch(ae, dataloader, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
                losses += lossesN
                testerrors1 += testerrors1N
                testerrors2 += testerrors2N
                testerrorsinf += testerrorsinfN

                if best and ep > epochs // 2:
                    avgloss = np.mean(lossesN)
                    if avgloss < bestdict["loss"]:
                        bestdict["model"] = ae.state_dict()
                        bestdict["opt"] = opt.state_dict()
                        bestdict["loss"] = avgloss
                        bestdict["ep"] = ep
                    elif verbose:
                        print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

                if ep % 5 == 0 and plottb:
                    WeldHelper.plot_encoding_window(self, w, encoding_param, step=self.aestep, writer=writer, tensorboard=True)

            print(f"Finish training AE {w} at {time.asctime()}.")
            losses_all.append(losses)
            testerrors1_all.append(testerrors1)
            testerrors2_all.append(testerrors2)
            testerrorsinf_all.append(testerrorsinf)

            if best:
                ae.load_state_dict(bestdict["model"])
                opt.load_state_dict(bestdict["opt"])

        self.aeepochs.append(epochs_first)
        if epochs != epochs_first:
            self.aeepochs.append(epochs)

        end = time.time()
        self.timetaken += end - start
        print("Finished training all timewindows")
        return {"losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf}

    def train_aes_plus_props(self, epochs, lamb=0.1, save=True, roll=False, optim=torch.optim.AdamW, lr=1e-4, plottb=False, forceparallel=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):
        def both_epoch(model, modelprop, dataloader, writer=None, w=0, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []
            testerrors1 = []
            testerrors2 = []
            testerrorsinf = []

            def closure(batch):
                optimizer.zero_grad()

                if roll:
                    rot = np.random.randint(0, batch.shape[-1])
                    batch = torch.roll(batch, shifts=rot, dims=-1)

                enc = self.encode_model(model, batch)

                proj = model.decode(enc)

                res = loss(batch, proj)

                starts = enc[:, :-1]
                exacts = enc[:, 1:]

                if self.tiprop:
                    predicted = self.propagate(starts[:, 0], self.windowvals[w][0], exacts.shape[1], arrencoded=True)
                    predicted = torch.stack(predicted, dim=1)

                else:
                    predicted = self.prop_forward(modelprop, starts)

                error = loss(predicted, exacts)

                totalloss = res + lamb * error

                totalloss.backward()

                num = totalloss.cpu().detach()
                if writer is not None and self.aestep % 5 == 0:
                    writer.add_scalar("main/loss", float(num), global_step=self.aestep)

                return totalloss

            for batch in dataloader:
                batch = batch.to(next(model.parameters()).device)
                self.aestep += 1
                lossout = optimizer.step(lambda: closure(batch))
                losses.append(float(lossout.cpu().detach()))

            if scheduler is not None and ep > epochs // 2:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testerr1, testerr2, testerrinf = self.get_proj_errors(model, testarr, ords=(1, 2, np.inf))

                if scheduler is not None:
                    print(f"W{w+1} / {ep+1}: Train Loss {lossout:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                else:
                    print(f"W{w+1} / {ep+1}: Train Loss {lossout:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

                if writer is not None:
                    writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
                    writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
                    writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

            return losses, testerrors1, testerrors2, testerrorsinf

        def train_window(w, parallel=False):
            ae = self.aes[w]
            prop = self.props[w]

            losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
            bestdict = {"loss": float(np.inf), "ep": 0}

            self.aestep = 0
            train = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float32)
            test = self.alltest[:, self.windowvals[w], :]

            if isinstance(ae, PCAAutoencoder):
                ae.train_pca(train.cpu().numpy())
                testerr1, testerr2, testerrinf = self.get_proj_errors(ae, test, ords=(1, 2, np.inf))
                print(f"Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                return

            if ae.pcadim > 0:
                testerr1, testerr2, testerrinf = ae.train_pca(train.reshape([train.shape[0] * train.shape[1], -1]))
                print(f"PCA for preprocessing has relative error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

            if parallel:
                ae.to(f"cuda:{w % torch.cuda.device_count()}")
                prop.to(f"cuda:{w % torch.cuda.device_count()}")
            else:
                ae.to(next(self.aes[0].parameters()).device)
                prop.to(next(self.props[0].parameters()).device)

            self.trains.append(train)
            self.tests.append(test)

            opt = optim(list(ae.parameters()) + list(prop.parameters()), lr=lr, weight_decay=ridge)
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
            dataloader = DataLoader(train, shuffle=False, batch_size=batch)

            writer = None
            if self.td is not None:
                name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-weld{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
                writer = torch.utils.tensorboard.SummaryWriter(name)
                print("Tensorboard writer location is " + name)

            print("Number of NN trainable parameters", num_params(ae))
            print(f"Starting training WeldNet AE + Prop {w+1}/{self.W} ({self.windowvals[w][0]}->{self.windowvals[w][-1]}) at {time.asctime()}...")

            print("train", train.shape, "test", test.shape)

            self.aestep = 0
            for ep in range(epochs):
                lossesN, testerrors1N, testerrors2N, testerrorsinfN = both_epoch(ae, prop, dataloader, w=w, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
                losses += lossesN
                testerrors1 += testerrors1N
                testerrors2 += testerrors2N
                testerrorsinf += testerrorsinfN

                if best and ep > epochs // 2:
                    avgloss = np.mean(lossesN)
                    if avgloss < bestdict["loss"]:
                        bestdict["model"] = ae.state_dict()
                        bestdict["opt"] = opt.state_dict()
                        bestdict["loss"] = avgloss
                        bestdict["ep"] = ep
                    elif verbose:
                        print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

                if ep % 5 == 0 and plottb:
                    WeldHelper.plot_encoding_window(self, w, encoding_param, step=self.aestep, writer=writer, tensorboard=True)

            print(f"Finish training AE and Prop {w} at {time.asctime()}.")

        loss = nn.MSELoss() if loss is None else loss()
        encoding_param = determine_param(self.dataset, encoding_param)

        losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

        start = time.time()
        self.metadata["trainedtogether"] = True
        print(f"Training {self.W} WeldNet AEs and props together")
        self.trains = []
        self.tests = []

        for w in range(self.W):
            if len(self.aes) <= w:
                self.aes.append(self.aeclass(**self.aeparams).to(self.device) if self.aeclass not in Other_Modules else self.aeclass(self.aeparams.copy()))
            if len(self.props) <= w:
                self.props.append(self.propclass(**self.propparams).to(self.device) if self.propclass not in Other_Modules else self.propclass(self.propparams.copy()))

        if forceparallel or (self.W > 1 and torch.cuda.device_count() >= self.W):
            print("Spawning threads for each window")
            with ThreadPoolExecutor(max_workers=self.W) as ex:
                futures = [ex.submit(train_window, rank, True) for rank in range(self.W)]
                for f in futures:
                    f.result()
        else:
            for ww in range(self.W):
                train_window(ww)

        self.aeepochs.append(epochs)

        end = time.time()
        self.timetaken += end - start
        print("Finished training all timewindows")
        return {}

    def train_transcoders(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, verbose=False, propagated_trans=True, forceparallel=False, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True):
        def transcoder_epoch(model, dataloader, writer=None, scheduler=None, optimizer=None, ep=0, printinterval=10, loss=None, testarr=None):
            losses = []
            testerrors1 = []
            testerrors2 = []
            testerrorsinf = []

            def closure(batch):
                optimizer.zero_grad()

                x = batch[:, :, 0]
                y = batch[:, :, 1]

                predict = self.trans_forward(model, x)

                if self.residualprop:
                    predict = x + predict

                res = loss(predict, y)
                res.backward()

                if writer is not None and self.transstep % 5:
                    writer.add_scalar("main/loss", float(res.cpu().detach()), global_step=self.transstep)

                return res

            for batch in dataloader:
                self.transstep += 1
                error = optimizer.step(lambda: closure(batch))
                losses.append(float(error.cpu().detach()))

            if scheduler is not None:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testdom = torch.tensor(testarr[:, :, 0], dtype=torch.float32)
                testran = testarr[:, :, 1]
                predict = model(testdom)

                if self.residualprop:
                    predict = testdom + predict

                predict = predict.cpu().detach().numpy()

                testerr1 = np.mean(np.linalg.norm(testran - predict, axis=1, ord=1) / np.linalg.norm(testran, axis=1, ord=1))
                testerr2 = np.mean(np.linalg.norm(testran - predict, axis=1, ord=2) / np.linalg.norm(testran, axis=1, ord=2))
                testerrinf = np.mean(np.linalg.norm(testran - predict, axis=1, ord=np.inf) / np.linalg.norm(testran, axis=1, ord=np.inf))

                testerrors1.append(testerr1)
                testerrors2.append(testerr2)
                testerrorsinf.append(testerrinf)

                if scheduler is not None:
                    print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative Transcoding Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                else:
                    print(f"{ep+1}: Train Loss {error:.3e}, Relative Transcoding Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

                if writer is not None:
                    writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
                    writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
                    writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

            return losses, testerrors1, testerrors2, testerrorsinf

        assert(self.transclass is not None)

        loss = nn.MSELoss() if loss is None else loss()
        encoding_param = determine_param(self.dataset, encoding_param)

        losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

        start = time.time()
        print(f"Training {self.W-1} WeldNet Transcoders")

        def train_window(w, parallel=False):
            losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
            bestdict = {"loss": float(np.inf), "ep": 0}

            self.transstep = 0

            trans_t = self.windowvals[w][-1]
            t0 = self.windowvals[w][0]
            trans_idx = trans_t - t0

            data = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float, device=self.device)
            datatest = torch.tensor(self.alltest[:, self.windowvals[w], :], dtype=torch.float, device=self.device)

            encodedran = self.encode_window(w+1, data[:, trans_idx]).detach()
            encodedtestran = self.encode_window(w+1, datatest[:, trans_idx]).detach()

            if propagated_trans:
                encodeddom = self.encode_window(w, data[:, 0]).detach()
                encodedtestdom = self.encode_window(w, datatest[:, 0]).detach()

                steps_to_trans = trans_t - t0
                if steps_to_trans > 0:
                    encodedinputs = torch.tensor(self.propagate(encodeddom, t0, steps_to_trans, arrencoded=True, fixedw=w, return_codes=True)[-1].detach())
                    encodedtestinputs = torch.tensor(self.propagate(encodedtestdom, t0, steps_to_trans, arrencoded=True, fixedw=w, return_codes=True)[-1].detach())
                else:
                    encodedinputs = encodeddom
                    encodedtestinputs = encodedtestdom
            else:
                encodedinputs = self.encode_window(w, data[:, trans_idx]).detach()
                encodedtestinputs = self.encode_window(w, datatest[:, trans_idx]).detach()

            train = torch.stack((encodedinputs, encodedran), dim=2)
            test = torch.stack((encodedtestinputs, encodedtestran), dim=2).detach().cpu().numpy()

            trans = self.transcoders[w]

            if parallel:
                trans.to(f"cuda:{w % torch.cuda.device_count()}")
            else:
                trans.to(next(self.transcoders[0].parameters()).device)

            opt = optim(trans.parameters(), lr=lr, weight_decay=ridge)
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)

            dataloader = DataLoader(train, shuffle=False, batch_size=batch)

            writer = None
            if self.td is not None:
                name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-trans{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
                writer = torch.utils.tensorboard.SummaryWriter(name)
                print("Tensorboard writer location is " + name)

            print("Number of NN trainable parameters", num_params(trans))
            print(f"Starting training Weldnet transcoder {w+1}/{self.W-1} (t={trans_t}) at {time.asctime()}...")
            print("train", train.shape, "test", test.shape)

            self.transstep = 0
            for ep in range(epochs):
                lossesN, testerrors1N, testerrors2N, testerrorsinfN = transcoder_epoch(trans, dataloader, optimizer=opt, writer=writer, scheduler=scheduler, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
                losses += lossesN
                testerrors1 += testerrors1N
                testerrors2 += testerrors2N
                testerrorsinf += testerrorsinfN

                if best and ep > epochs // 2:
                    avgloss = np.mean(lossesN)
                    if avgloss < bestdict["loss"]:
                        bestdict["model"] = trans.state_dict()
                        bestdict["opt"] = opt.state_dict()
                        bestdict["loss"] = avgloss
                        bestdict["ep"] = ep
                    elif verbose:
                        print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

            print(f"Finish training transcoder {w} at {time.asctime()}.")
            losses_all.append(losses)
            testerrors1_all.append(testerrors1)
            testerrors2_all.append(testerrors2)
            testerrorsinf_all.append(testerrorsinf)

            if best and "model" in bestdict:
                trans.load_state_dict(bestdict["model"])
                opt.load_state_dict(bestdict["opt"])

        for w in range(self.W - 1):
            if len(self.transcoders) <= w:
                self.transcoders.append(self.transclass(**self.transparams).to(self.device))

        if forceparallel or (self.W > 2 and torch.cuda.device_count() >= self.W - 1):
            print("Spawning threads for each transcoder")
            with ThreadPoolExecutor(max_workers=self.W - 1) as ex:
                futures = [ex.submit(train_window, rank, True) for rank in range(self.W - 1)]
                for f in futures:
                    f.result()
        else:
            for ww in range(self.W - 1):
                train_window(ww)

        self.transepochs.append(epochs)
        end = time.time()
        self.timetaken += end - start
        print("Finished training all timewindow transcoders")
        return {"losses": losses_all, "testerrors1": testerrors1_all, "testerrors2": testerrors2_all, "testerrorsinf": testerrorsinf_all}

    def train_propagators(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, forceparallel=False, loss=None, encoding_param=-1, best=True, verbose=False):
        def prop_epoch(w, dataloader, writer=None, scheduler=None, optimizer=None, ep=0, printinterval=10, loss=None, testtensor=None):
            model = self.props[w]

            losses = []
            testerrors1 = []
            testerrors2 = []
            testerrorsinf = []

            def closure(x, y, t):
                optimizer.zero_grad()

                if t is not None:
                    xt = torch.cat((x, t), dim=2)
                else:
                    xt = x

                if self.tiprop:
                    predict = self.propagate(x[:, 0], self.windowvals[w][0], xt.shape[1], arrencoded=True, return_codes=True)
                    predict = torch.stack(predict, dim=1)

                elif self.accumulateprop:
                    x0 = xt[:, :1]

                    xlist = []
                    for _ in range(xt.shape[1]):
                        x0 = self.prop_forward(model, x0)
                        xlist.append(x0)

                    predict = torch.cat(xlist, dim=1)

                else:
                    predict = self.prop_forward(model, x)

                if self.decodedprop:
                    decoder = self.aes[w]
                    predict_decoded = decoder.decode(predict)
                    target_decoded = decoder.decode(y).detach()
                    res = loss(predict_decoded, target_decoded)
                else:
                    res = loss(predict, y)

                lossval = res
                lossval.backward(retain_graph=False)

                if writer is not None and self.propstep % 10 == 0:
                    writer.add_scalar("main/loss", float(res.cpu().detach()), global_step=self.propstep)

                return res

            for batch in dataloader:
                self.propstep += 1
                batch = batch.to(self.device)

                x = batch[:, :-1]
                y = batch[:, 1:]

                if not self.autonomous and not self.accumulateprop:
                    t = -1 + 0*torch.tensor(np.repeat(np.expand_dims(self.windowvals[w][:-1], 0), x.shape[0], axis=0)).unsqueeze(2).to(self.device).float()
                else:
                    t = None

                error = optimizer.step(lambda: closure(x, y, t))
                losses.append(float(error.cpu().detach()))

                if writer is not None:
                    if self.propstep % 5 == 0:
                        writer.add_scalar("propagator/loss", float(error), global_step=self.propstep)

            if scheduler is not None:
                scheduler.step(np.mean(losses))

            if printinterval > 0 and (ep % printinterval == 0):
                testinputs = testtensor[:, 0:1, :]
                testoutputs = testtensor[:, -1, :].cpu().detach().numpy()

                steps = len(self.windowvals[w]) - 1
                codes = testinputs
                for _ in range(steps):
                    codes = self.prop_forward(self.props[w], codes)
                predict = codes.squeeze(1).cpu().detach().numpy()

                testerr1 = np.mean(np.linalg.norm(predict - testoutputs, axis=1, ord=1) / np.linalg.norm(testoutputs, axis=1, ord=1))
                testerr2 = np.mean(np.linalg.norm(predict - testoutputs, axis=1, ord=2) / np.linalg.norm(testoutputs, axis=1, ord=2))
                testerrinf = np.mean(np.linalg.norm(predict - testoutputs, axis=1, ord=np.inf) / np.linalg.norm(testoutputs, axis=1, ord=np.inf))

                testerrors1.append(testerr1)
                testerrors2.append(testerr2)
                testerrorsinf.append(testerrinf)

                if scheduler is not None:
                    print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative Propagator Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
                else:
                    print(f"{ep+1}: Train Loss {error:.3e}, Relative Propagator Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

                if writer is not None:
                    writer.add_scalar("misc/relativeL1prop", testerr1, global_step=ep)
                    writer.add_scalar("main/relativeL2prop", testerr2, global_step=ep)
                    writer.add_scalar("misc/relativeLInfprop", testerrinf, global_step=ep)

            return losses, testerrors1, testerrors2, testerrorsinf

        loss = nn.MSELoss() if loss is None else loss()
        encoding_param = determine_param(self.dataset, encoding_param)

        losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

        start = time.time()
        print(f"Training {self.W} WeldNet Propagators")

        def train_window(w, parallel=False):
            losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
            bestdict = {"loss": float(np.inf), "ep": 0}

            self.propstep = 0
            train = torch.tensor(self.alltrain[:, self.windowvals[w]], dtype=torch.float32)
            test = self.alltest[:, self.windowvals[w]]

            encoded = self.encode_window(w, train).detach()
            encodedtest = self.encode_window(w, torch.tensor(test, dtype=torch.float32).to(self.device)).detach()

            prop = self.props[w]

            if parallel:
                prop.to(f"cuda:{w % torch.cuda.device_count()}")
            else:
                prop.to(next(self.props[0].parameters()).device)

            opt = optim(prop.parameters(), lr=lr, weight_decay=ridge)
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)

            dataloader = DataLoader(encoded, shuffle=False, batch_size=batch)

            writer = None
            if self.td is not None:
                name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-prop{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
                writer = torch.utils.tensorboard.SummaryWriter(name)
                print("Tensorboard writer location is " + name)

            print("Number of NN trainable parameters", num_params(prop))
            print(f"Starting training WeldNet propagator {w+1}/{self.W} ({self.windowvals[w][0]}->{self.windowvals[w][-1]}) at {time.asctime()}...")
            print("train", encoded.shape, "test", encodedtest.shape)

            self.propstep = 0
            for ep in range(epochs):
                lossesN, testerrors1N, testerrors2N, testerrorsinfN = prop_epoch(w, dataloader, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testtensor=encodedtest)
                losses += lossesN
                testerrors1 += testerrors1N
                testerrors2 += testerrors2N
                testerrorsinf += testerrorsinfN

                if best and ep > epochs // 2:
                    avgloss = np.mean(lossesN)
                    if avgloss < bestdict["loss"]:
                        bestdict["model"] = prop.state_dict()
                        bestdict["opt"] = opt.state_dict()
                        bestdict["loss"] = avgloss
                        bestdict["ep"] = ep
                    elif verbose:
                        print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

            print(f"Finish training propagator {w} at {time.asctime()}.")
            losses_all.append(losses)
            testerrors1_all.append(testerrors1)
            testerrors2_all.append(testerrors2)
            testerrorsinf_all.append(testerrorsinf)

            if best and "model" in bestdict:
                prop.load_state_dict(bestdict["model"])
                opt.load_state_dict(bestdict["opt"])

        for w in range(self.W):
            if len(self.props) <= w:
                self.props.append(self.propclass(**self.propparams).to(self.device))

        if forceparallel or (self.W > 1 and torch.cuda.device_count() >= self.W):
            print("Spawning threads for each propagator")
            with ThreadPoolExecutor(max_workers=self.W) as ex:
                futures = [ex.submit(train_window, rank, True) for rank in range(self.W)]
                for f in futures:
                    f.result()
        else:
            for ww in range(self.W):
                train_window(ww)

        self.propepochs.append(epochs)
        end = time.time()
        self.timetaken += end - start
        print("Finished training all timewindow propagators")
        return {"losses": losses_all, "testerrors1": testerrors1_all, "testerrors2": testerrors2_all, "testerrorsinf": testerrorsinf_all}

    def load_models(self, filename_prefix, verbose=False, min_epochs=0, together=True):
        search_path = f"savedmodels/{'weldtogether' if together else 'weld'}/{filename_prefix}*.pickle"

        matching_files = glob.glob(search_path)

        if together:
            self.metadata["trainedtogether"] = True

        print("Searching for model files matching prefix:", filename_prefix)
        if not hasattr(self, "metadata"):
            raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

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
                str(meta.get(k)) == str(self.metadata.get(k))
                for k in meta.keys()
            )

            model_epochs = dic.get("aeepochs")
            if model_epochs is None:
                if verbose:
                    print(f"Skipping {addr} due to missing epoch metadata.")
                continue
            elif isinstance(model_epochs, list):
                if sum(model_epochs) < min_epochs:
                    if verbose:
                        print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
                    continue
            elif model_epochs < min_epochs:
                if verbose:
                    print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
                continue

            if is_match:
                print("Model match found. Loading from:", addr)

                self.aes = dic["aes"]
                self.props = dic["props"]
                self.transcoders = dic["trans"]
                self.timetaken = dic["timetaken"]

                self.epochs = model_epochs

                return True
            elif verbose:
                print("Metadata mismatch in file:", addr)
                for k in self.metadata:
                    if meta.get(k) != self.metadata.get(k):
                        print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

        print("Load failed. No matching models found.")
        print("Searched:", matching_files)
        return False

    def save_models(self):
        assert(len(self.aes) == self.W)
        assert(len(self.props) == self.W)

        if self.transcoders is None:
            assert(self.W == 1)
            num_paramstrans = 0
        else:
            assert(len(self.transcoders) == self.W - 1)
            num_paramstrans = sum([num_params(x) for x in self.transcoders])

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        num_paramsAE = sum([num_params(x) for x in self.aes])
        num_paramsprops = sum([num_params(x) for x in self.props])

        total_epochs = sum(self.aeepochs)

        filename = (
            f"{self.dataset.name}_"
            f"{self.aeclass.__name__}_"
            f"{num_paramsAE}_"
            f"{num_paramsprops}_"
            f"{num_paramstrans}_"
            f"{self.seed}_"
            f"{total_epochs}ep_"
            f"{now}.pickle"
        )

        if self.metadata.get("trainedtogether", False):
            dire = "savedmodels/weldtogether"
        else:
            dire = "savedmodels/weld"

        addr = os.path.join(dire, filename)

        if not os.path.exists(dire):
            os.makedirs(dire)

        with open(addr, "wb") as handle:
            pickle.dump({
                "aes": self.aes,
                "props": self.props,
                "trans": self.transcoders,
                "aeepochs": self.aeepochs,
                "propepochs": self.propepochs,
                "transepochs": self.transepochs,
                "metadata": self.metadata,
                "timetaken": self.timetaken
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Models saved at", addr)


class WeldHelper():
    def __init__(self, config):
        self.update_config(config)

    def update_config(self, config):
        self.config = config

    def create_weld(self, dataset, config=None, **args):
        if config is None:
            config = self.config

        assert(len(dataset.data.shape) < 5)

        windows = args.get("windows", config.windows)
        aeclass = globals()[args.get("aeclass", config.aeclass)]
        aeparams = dict(args.get("aeparams", config.aeparams))
        seqclass = globals()[args.get("seqclass", config.seqclass)]
        seqparams = dict(args.get("seqparams", config.seqparams))

        if len(dataset.data.shape) == 3:
            din = dataset.data.shape[2]
        else:
            din = dataset.data.shape[2] * dataset.data.shape[3]

        if aeclass in Other_Modules:
            aeparams = OmegaConf.create(aeparams)
            aeparams.sample.spatial_resolution = din
            if "k" in args:
                aeparams.sample.latents_dims = args["k"]
                seqparams["seq"][0] = args["k"]
                seqparams["seq"][-1] = args["k"]
            seqparams["activation"] = get_activation(seqparams["activation"])

        else:
            if aeclass == PCAAutoencoder:
                aeparams["inputdim"] = din
                aeparams["inputdim"] = din
                seqparams["activation"] = get_activation(seqparams["activation"])
                if "k" in args:
                    aeparams["reduced"] = args["k"]
                    seqparams["seq"][0] = args["k"]
                    seqparams["seq"][-1] = args["k"]

            else:
                aeparams["encodeSeq"][0] = din

                aeparams["decodeSeq"][-1] = din

                aeparams["activation"] = get_activation(aeparams["activation"])
                seqparams["activation"] = get_activation(seqparams["activation"])

                if "k" in args:
                    aeparams["encodeSeq"][-1] = args["k"]
                    aeparams["decodeSeq"][0] = args["k"]
                    seqparams["seq"][0] = args["k"]
                    seqparams["seq"][-1] = args["k"]

                pcadim = 0
                if hasattr(config.aeparams, "pcadim"):
                    pcadim = config.aeparams.pcadim
                if "pcadim" in args:
                    pcadim = args["pcadim"]

                aeparams["pcadim"] = pcadim

                if pcadim > 0:
                    aeparams["encodeSeq"][0] = pcadim

                    aeparams["decodeSeq"][-1] = pcadim

        td = args.get("td", None)
        seed = args.get("seed", 0)
        device = args.get("device", 0)
        autonomous = args.get("autonomous", True)
        accumulateprop = args.get("accumulateprop", True)
        decodedprop = args.get("decodedprop", False)
        dynamicwindow = args.get("dynamicwindow", False)

        propparams = copy.deepcopy(seqparams)
        tiprop = args.get("tiprop", False)
        if tiprop:
            propparams["seq"][0] = propparams["seq"][-1] + 1

        return WeldNet(dataset, windows, aeclass, aeparams.copy(), seqclass, propparams, seqclass, copy.deepcopy(seqparams), dynamicwindow=dynamicwindow, accumulateprop=accumulateprop, decodedprop=decodedprop, td=td, tiprop=tiprop, seed=seed, device=device, autonomous=autonomous)

    @staticmethod
    def plot_encoding_window(weld, ws=-1, p=-1, writer=None, step=None, tensorboard=False, maxscatter=10000, testonly=False, threedim=False):
        if p == -1:
            p = weld.dataset.determine_params()

        if type(p) == type(0):
            p = [p]

        if ws == -1:
            ws = range(len(weld.aes))

        if type(ws) == type(0):
            ws = [ws]

        for w in ws:
            dim = weld.aes[w].reduced

            arr = weld.tests[w] if testonly else np.concatenate([weld.tests[w], weld.trains[w].cpu().numpy()])
            params = weld.dataset.params
            arr = torch.tensor(arr).to(weld.device, dtype=torch.float32)
            enc = weld.encode_window(w, arr)

            points = enc.cpu().detach().numpy()

            plt.rcParams.update({'font.size': 12})
            for pp in p:
                fig = plt.figure(figsize=(8, 3))

                if dim == 2 or tensorboard or not threedim:
                    ax0 = fig.add_subplot(121)
                    ax1 = fig.add_subplot(122)

                    sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], c=params[:maxscatter, pp], s=2)
                    plt.colorbar(sc0, ax=ax0, location="right", pad=0)
                    ax0.set_xlabel("$z_1$")
                    ax0.set_ylabel("$z_2$")
                    ax0.set_title(f"Parameter {pp}")

                    sc1 = ax1.scatter(points[:maxscatter, 0], points[:maxscatter, 1], c=params[:maxscatter, -1], s=2)
                    plt.colorbar(sc1, ax=ax1, location="right", pad=0)
                    ax1.set_xlabel("$z_1$")
                    ax1.set_ylabel("$z_2$")
                    ax1.set_title(f"Time")
                elif dim >= 3:
                    ax0 = fig.add_subplot(121, projection="3d")
                    ax1 = fig.add_subplot(122, projection="3d")

                    sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], points[:maxscatter, 2], c=params[:maxscatter, pp], s=2)
                    plt.colorbar(sc0, ax=ax0, location="right", pad=0)
                    ax0.set_xlabel("$z_1$")
                    ax0.set_ylabel("$z_2$")
                    ax0.set_zlabel("$z_3$")
                    ax0.set_title(f"Parameter {pp}")

                    sc1 = ax1.scatter(points[:maxscatter, 0], points[:maxscatter, 1], points[:maxscatter, 2], c=params[:maxscatter, -1], s=2)
                    plt.colorbar(sc1, ax=ax1, location="right", pad=0)
                    ax1.set_xlabel("$z_1$")
                    ax1.set_ylabel("$z_2$")
                    ax1.set_zlabel("$z_3$")
                    ax1.set_title(f"Time")
                else:
                    raise ValueError(f"Latent dim must be >= 2 for plotting, got {dim}")

                fig.tight_layout()

                if tensorboard:
                    assert(step is not None)
                    fig.suptitle(step)
                    writer.add_figure(f'main/latent-p{pp}', fig, global_step=step)
                    writer.flush()
                    torch.cuda.empty_cache()
                    plt.close(fig)

    @staticmethod
    def get_projerr(weld, times=None, testonly=False, relative=True, eps=1e-8):
        if times is None:
            times = range(int(weld.T))

        errors = []
        for t in times:
            if testonly:
                data = weld.alltest
            else:
                data = np.concatenate([weld.alltest, weld.alltrain])

            data_t = torch.tensor(data[:, t:t+1, :]).to(weld.device, dtype=torch.float32)

            w = weld.find_window(t)
            proj_t = weld.project_window(w, data_t)

            proj = proj_t.cpu().detach().numpy()
            data_np = data_t.cpu().detach().numpy()

            N = data_np.shape[0]
            data_flat = data_np.reshape([N, -1])
            proj_flat = proj.reshape([N, -1])

            diff = proj_flat - data_flat
            num = np.linalg.norm(diff, axis=1)

            if relative:
                denom = np.linalg.norm(data_flat, axis=1)
                rel = num / np.maximum(denom, eps)
                errors.append(rel.mean())
            else:
                errors.append(num.mean())

        return errors

    @staticmethod
    def get_operrs(weld, steps=-1, t=0, testonly=False, fullerror=False, times=None):
        if times is not None:
            fullerror = True

        if steps == -1:
            steps = weld.T - t - 1

        assert(t + steps < weld.T)

        if testonly:
            data = weld.alltest
        else:
            data = np.concatenate([weld.alltest, weld.alltrain])

        inputt = torch.tensor(data[:, t:t+1, :]).to(weld.device, dtype=torch.float32)

        references = [data[:, t+s, :] for s in range(1, steps+1)]

        predicteds = weld.propagate(inputt, t, steps)

        predictedvals = []
        for i, x in enumerate(predicteds):
            win = weld.find_window(t+i+1)
            predictedvals.append(weld.decode_window(win, x).cpu().detach().numpy())

        predicteds = [x.cpu().detach().numpy() for x in predicteds]

        inputt = inputt.cpu().detach().numpy()

        N = predictedvals[0].shape[0]
        predictedvals = [x.reshape([N, -1]) for x in predictedvals]
        references = [x.reshape([N, -1]) for x in references]

        errors = []

        for s in range(1, steps+1):
            if fullerror:
                errors.append(np.linalg.norm(predictedvals[s-1] - references[s-1], axis=1) / np.linalg.norm(references[s-1], axis=1))
            else:
                errors.append(np.mean(np.linalg.norm(predictedvals[s-1] - references[s-1], axis=1) / np.linalg.norm(references[s-1], axis=1)))

        if times is not None and len(times) == 1:
            return errors[times[0] - t - 1]

        return errors

    @staticmethod
    def plot_encoding_time(weld, ts=-1, p=-1, writer=None, step=None, tensorboard=False, maxscatter=10000, testonly=False, threedim=False):
        if p == -1:
            p = weld.dataset.determine_params()

        if type(p) == type(0):
            p = [p]

        if ts == -1:
            ts = range(weld.T)

        if type(ts) == type(0):
            ts = [ts]

        allpoints = []
        allparams = []

        if testonly:
            dataset = weld.alltest
        else:
            dataset = np.concatenate([weld.alltest, weld.alltrain])

        for t in ts:
            w = weld.find_window(t)
            dim = weld.aes[w].reduced

            arr = dataset[:, t:t+1, :]
            params = weld.dataset.params
            arr = torch.tensor(arr).to(weld.device, dtype=torch.float32)
            enc = weld.encode_window(w, arr)

            points = enc.cpu().detach().numpy()
            points = points[:(maxscatter // len(ts))+1]
            params = params[:(maxscatter // len(ts))+1]

            allpoints.append(points)
            allparams.append(params)

        points = np.concatenate(allpoints, axis=0)
        params = np.concatenate(allparams, axis=0)

        plt.rcParams.update({'font.size': 12})
        for pp in p:
            fig = plt.figure(figsize=(8, 3))

            if dim == 2 or tensorboard or not threedim:
                ax0 = fig.add_subplot()

                sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], c=params[:maxscatter, pp], s=2)
                plt.colorbar(sc0, ax=ax0, location="right", pad=0)
                ax0.set_xlabel("$z_1$")
                ax0.set_ylabel("$z_2$")
                ax0.set_title(f"Parameter {pp}")
            elif dim >= 3:
                ax0 = fig.add_subplot(projection="3d")

                sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], points[:maxscatter, 2], c=params[:maxscatter, pp], s=2)
                plt.colorbar(sc0, ax=ax0, location="right", pad=0)
                ax0.set_xlabel("$z_1$")
                ax0.set_ylabel("$z_2$")
                ax0.set_zlabel("$z_3$")
                ax0.set_title(f"Parameter {pp}")

            else:
                raise ValueError(f"Latent dim must be >= 2 for plotting, got {dim}")

            fig.tight_layout()

            if tensorboard:
                assert(step is not None)
                fig.suptitle(step)
                writer.add_figure(f'main/latent-p{pp}', fig, global_step=step)
                writer.flush()
                torch.cuda.empty_cache()
                plt.close(fig)

        return fig

    @staticmethod
    def compare_pcaproj(weld, k=10, testonly=False, windowaverage=False):
        aeerrs = []
        pcaerrs = []

        for w in range(weld.W):
            data = weld.tests[w] if testonly else weld.dataset.data[:, weld.windowvals[w], :]
            data = torch.tensor(data).to(weld.device, dtype=torch.float32)
            proj = weld.project_window(w, data)

            data = data.cpu().detach().numpy()
            proj = proj.cpu().detach().numpy()

            pca = PCA(n_components=k)
            pca = pca.fit(data.reshape(-1, data.shape[-1]))

            if windowaverage:
                aeerrs.append(np.mean(np.linalg.norm(proj - data) / np.linalg.norm(data)))
            else:
                for t in range(data.shape[1]):
                    projslice = proj[:, t]
                    dataslice = data[:, t]
                    aeerrs.append(np.mean(np.linalg.norm(projslice - dataslice) / np.linalg.norm(dataslice)))

                    components = pca.transform(dataslice)
                    rdata = pca.inverse_transform(components)
                    pcaerrs.append(np.mean(np.linalg.norm(rdata - dataslice) / np.linalg.norm(dataslice)))

        fig, ax = plt.subplots()

        if windowaverage:
            times = np.arange(weld.W)
            ax.set_xlabel("Window")
        else:
            times = np.arange(weld.T)
            ax.set_xlabel("Time")

        ax.plot(times, aeerrs, marker='o', label="AE")
        ax.plot(times, pcaerrs, marker='o', label=f"PCA{k}")

        ax.set_ylabel("RelL2 Reconstruction Error")
        ax.legend()

        fig.tight_layout()

    @staticmethod
    def get_onestepprop(weld, testonly=False, relative=True):
        errors = []

        for i in range(weld.W):
            if testonly:
                data = weld.dataset.data[weld.numtrain:, :, :]
            else:
                data = weld.dataset.data

            if i < weld.W - 1:
                data = data[:, weld.windowvals[i][:-1]]
            else:
                data = data[:, weld.windowvals[i]]

            datatensor = torch.tensor(data).to(dtype=torch.float32)

            latents = weld.encode_window(i, datatensor).detach()
            predicteds = weld.prop_forward(weld.props[i], latents[:, :-1]).detach().cpu()
            references = latents[:, 1:].detach().cpu()

            for s in range(predicteds.shape[1]):
                if relative:
                    errors.append(np.mean(np.linalg.norm(predicteds[:, s] - references[:, s], axis=1) / np.linalg.norm(references[:, s], axis=1)))
                else:
                    errors.append(np.mean(np.linalg.norm(predicteds[:, s] - references[:, s], axis=1)))

        return errors

    @staticmethod
    def get_properrs(weld, steps=-1, t=0, testonly=False, relative=True):
        if steps == -1:
            steps = weld.T - t - 1

        assert(t + steps < weld.T)

        if testonly:
            data = weld.dataset.data[weld.numtrain:, :, :]
        else:
            data = weld.dataset.data

        datatensor = torch.tensor(data).to(weld.device, dtype=torch.float32)
        inputt = datatensor[:, t, :]

        references = [weld.encode_window(weld.find_window(t+s+1), datatensor[:, t+s+1, :]).cpu().detach().numpy() for s in range(steps)]

        predicteds = weld.propagate(inputt, t, steps)
        predicteds = [x.cpu().detach().numpy() for x in predicteds]

        errors = []
        for s in range(steps):
            if relative:
                errors.append(np.mean(np.linalg.norm(predicteds[s] - references[s], axis=1) / np.linalg.norm(references[s], axis=1)))
            else:
                errors.append(np.mean(np.linalg.norm(predicteds[s] - references[s], axis=1)))

        return errors

    @staticmethod
    def plot_op_predicts(weld, t=0, steps=-1, xs=None, testonly=False, yscalefixed=False, cmap="viridis"):
        if steps == -1:
            steps = weld.T - t - 1

        assert(t + steps < weld.T)

        data = weld.dataset.data

        if xs is None:
            xs = np.linspace(0, 1, data.shape[2])

        if testonly:
            data = data[weld.numtrain:,]

        inputt = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)
        references = [data[:, t+s+1, :] for s in range(steps)]

        predicteds = weld.propagate(inputt, t, steps)
        predictedvals = [weld.decode_window(weld.find_window(t+i+1), x).cpu().detach().numpy() for i, x in enumerate(predicteds)]

        predicteds = [x.cpu().detach().numpy() for x in predicteds]
        inputt = inputt.cpu().detach().numpy()

        errors = []
        n = predictedvals[0].shape[0]
        for s in range(1, steps+1):
            predict = predictedvals[s-1].reshape((n, -1))
            reference = references[s-1].reshape((n, -1))
            errors.append(np.mean(np.linalg.norm(predict - reference, axis=1) / np.linalg.norm(reference, axis=1)))

        print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

        if len(data.shape) == 3:
            fig, ax = plt.subplots(figsize=(4, 3))
        elif len(data.shape) == 4:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            fig.subplots_adjust(right=0.90)
            sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

        n = references[0].shape[0]

        @widgets.interact(i=(0, n-1), s=(1, steps))
        def plot_interact(i=0, s=1):
            predict = predictedvals[s-1].reshape((n, -1))
            reference = references[s-1].reshape((n, -1))
            error = np.mean(np.linalg.norm(predict - reference, axis=1) / np.linalg.norm(reference, axis=1))
            print(f"Avg Relative L2 Error for t{t} to t{t+s}: {error:.4f}")

            if len(data.shape) == 3:
                ax.clear()
                ax.set_title(f"RelL2 {np.linalg.norm(predictedvals[s-1][i, :] - references[s-1][i, :]) / np.linalg.norm(references[s-1][i, :])}")
                ax.plot(xs, inputt[i], label="Input", linewidth=1)
                ax.plot(xs, predictedvals[s-1][i], label="Predicted", linewidth=1)
                ax.plot(xs, references[s-1][i], label="Exact", linewidth=1)
                ax.legend()

                if yscalefixed:
                    ax.set_ylim(min(np.min(predictedvals), np.min(references)), max(np.max(predictedvals), np.max(references)))
            elif len(data.shape) == 4:
                for axx in axes:
                    axx.clear()

                axes[0].imshow(inputt[i], cmap=cmap)
                axes[0].set_title("Initial")
                axes[1].imshow(references[s-1][i], cmap=cmap)
                axes[1].set_title("Exact")
                axes[2].imshow(predictedvals[s-1][i], cmap=cmap)
                axes[2].set_title("Predicted")

                cb = axes[3].imshow(np.abs(predictedvals[s-1][i] - references[s-1][i]), cmap=cmap)
                axes[3].set_title("|Difference|")
                fig.colorbar(cb, cax=sub_ax)

    @staticmethod
    def plot_ae_projection(weld, xs=None, testonly=False, cmap="viridis", yscalefixed=False):
        if testonly:
            data = weld.dataset.data[weld.numtrain:,]
        else:
            data = weld.dataset.data

        if xs is None:
            xs = np.linspace(0, 1, data.shape[2])

        exacts = [torch.tensor(data[:, i, :]).to(weld.device, dtype=torch.float32) for i in range(data.shape[1])]
        projecteds = [weld.project_window(weld.find_window(i), exacts[i]).cpu().detach().numpy() for i in range(len(exacts))]
        exacts = [x.cpu().detach().numpy() for x in exacts]

        errors = []
        for s in range(weld.T):
            exact = exacts[s]
            project = projecteds[s]
            exact = exact.reshape(list(exact.shape[:1]) + [-1])
            project = project.reshape(list(project.shape[:1]) + [-1])
            errors.append(np.mean(np.linalg.norm(exact - project, axis=1) / np.linalg.norm(exact, axis=1)))

        print(f"Average Relative L2 AE Error over all times: {np.mean(errors):.4f}")

        if len(data.shape) == 3:
            fig, ax = plt.subplots(figsize=(4, 3))
        elif len(data.shape) == 4:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            fig.subplots_adjust(right=0.90)
            sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

        @widgets.interact(i=(0, exacts[0].shape[0]-1), s=(0, weld.T-1))
        def plot_interact(i=0, s=1):
            err = np.linalg.norm(exacts[s][i] - projecteds[s][i]) / np.linalg.norm(exacts[s][i])
            print(f"Relative L2 AE Error: {err:.4f}")

            if len(data.shape) == 3:
                ax.clear()
                ax.plot(xs, projecteds[s][i], label="Predicted", linewidth=1)
                ax.plot(xs, exacts[s][i], label="Exact", linewidth=1)
                ax.legend()

                if yscalefixed:
                    ax.set_ylim(min(np.min(projecteds), np.min(exacts)), max(np.max(projecteds), np.max(exacts)))
            elif len(data.shape) == 4:
                axes[0].clear()
                axes[0].imshow(exacts[s][i], cmap=cmap)
                axes[0].set_title("Exact")
                axes[1].clear()
                axes[1].imshow(projecteds[s][i], cmap=cmap)
                axes[1].set_title("Predicted")

                axes[2].clear()
                cb = axes[2].imshow(np.abs(projecteds[s][i] - exacts[s][i]), cmap=cmap)
                axes[2].set_title("Difference")
                fig.colorbar(cb, cax=sub_ax)

    @staticmethod
    def plot_prop_scatter(weld, t=0, steps=-1, p=0, testonly=False):
        if steps == -1:
            steps = weld.T - t - 1

        assert(t + steps < weld.T)

        if testonly:
            data = weld.dataset.data[weld.numtrain:, :, :]
            params = weld.dataset.params[weld.numtrain:]
        else:
            data = weld.dataset.data
            params = weld.dataset.params

        reference = torch.tensor(data[:, t+steps, :]).to(weld.device, dtype=torch.float32)

        window = weld.find_window(t)
        windowtarget = weld.find_window(t + steps)

        correct = weld.encode_window(windowtarget, reference)

        arr = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)
        predicted = weld.propagate(arr, t, steps)[-1]
        predictedvals = weld.decode_window(windowtarget, predicted).cpu().detach().numpy()

        predicted = predicted.cpu().detach().numpy()
        correct = correct.cpu().detach().numpy()
        reference = reference.cpu().detach().numpy()

        error = np.mean(np.linalg.norm(predictedvals - reference, axis=1) / np.linalg.norm(reference, axis=1))
        print(f"Relative L2 Error for t{t} to t{t+steps}", error)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        sc0 = axes[0].scatter(predicted[:, 0], predicted[:, 1], c=params[:, p], s=2)
        sc1 = axes[1].scatter(correct[:, 0], correct[:, 1], c=params[:, p], s=2)
        plt.colorbar(sc0, ax=axes[0])
        plt.colorbar(sc1, ax=axes[1])

        axes[0].set_xlabel("Encoded Param 1")
        axes[0].set_ylabel("Encoded Param 2")
        axes[0].set_title(f"Predicted t{t+steps} from t{t}, {windowtarget - window + 1} windows")

        axes[1].set_xlabel("Encoded Param 1")
        axes[1].set_ylabel("Encoded Param 2")
        axes[1].set_title(f"Exact t{t+steps}")
        fig.tight_layout()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(predicted[:, 0], predicted[:, 1], c=params[:, p], s=10, marker="+", cmap="flag", label="Predicted")
        sc = ax.scatter(correct[:, 0], correct[:, 1], c=params[:, p], s=10, marker=".", cmap="flag", label="Exact")
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel("Encoded Param 1")
        ax.set_ylabel("Encoded Param 2")
        ax.set_title(f"Comparison t{t} to t{t+steps}, {windowtarget - window + 1} windows")
        ax.legend()

    @staticmethod
    def plot_projops(weld, title="", save=False, testonly=True):
        fig, ax = plt.subplots(figsize=(8, 5))

        projerrs = WeldHelper.get_projerr(weld, testonly=testonly)
        operrs = WeldHelper.get_operrs(weld, testonly=testonly)

        ax.plot(range(len(projerrs)), projerrs, label="Reconstruction Error")
        ax.plot(range(1, len(operrs)+1), operrs, label="Operator Error")

        for vals in weld.windowvals[:-1]:
            v = vals[-1]
            ax.axvline(v+1, linestyle=":", alpha=0.5)

        ax.set_xlabel("t")
        ax.set_ylabel("Relative L2 Error")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(title)

        fig.tight_layout()

        if save:
            plt.savefig(f"{title}-projoperror.pdf")

    @staticmethod
    def compare_projops(modellist, labels=None, relative=True, title=None, ylims=None, difference=False, testonly=True, windowlines=True, t=0):
        fig, ax = plt.subplots()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(modellist)))

        for i, x in enumerate(modellist):
            if windowlines:
                for xx in x.windowvals[:-1]:
                    if xx[-1] > t:
                        ax.axvline(xx[-1], linestyle=":", alpha=0.5, color=colors[i])

            projerrs = np.asarray(WeldHelper.get_projerr(x, testonly=testonly, relative=relative)[t+1:])
            operrs = np.asarray(WeldHelper.get_operrs(x, testonly=testonly, t=t))

            if difference:
                ax.plot(range(t, t+len(operrs)), np.log10(operrs - projerrs), label=f"{x.W if labels is None else labels[i]} Propagator Gap", c=colors[i])
            else:
                ax.plot(range(t, t+len(operrs)), np.log10(projerrs), "--", c=colors[i], alpha=0.5)
                ax.plot(range(t, t+len(operrs)), np.log10(operrs), label=f"{x.W if labels is None else labels[i]}", c=colors[i])

        ax.legend()

        ax.set_ylabel("$log_{10}$(Error)")
        ax.set_xlabel("Time")

        if ylims:
            ax.set_ylim(ylims)

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig

    @staticmethod
    def compare_projerrs(models, labels=None, windowlines=True):
        fig, ax = plt.subplots()

        if labels is None:
            labels = [x.W for x in models]

        for x, l in zip(models, labels):
            properrs = WeldHelper.get_projerr(x, testonly=True)
            ax.plot(np.log10(properrs), label=l)

            if windowlines:
                for xx in x.windowvals[:-1]:
                    ax.axvline(xx[-1] + 1, linestyle="--", alpha=0.5, color="gray")

        ax.legend()
        ax.set_title("Projection Error for Various #Windows")
        ax.set_ylabel("$log_{10}$(Projection Error)")
        ax.set_xlabel("Time")

        fig.tight_layout()
        return fig

    @staticmethod
    def compare_operrs(models, windowlines=True):
        fig, ax = plt.subplots()

        for x in models:
            properrs = WeldHelper.get_operrs(x, testonly=True)
            ax.plot(np.log10(properrs), label=x.W)

            if windowlines:
                for xx in x.windowvals[:-1]:
                    ax.axvline(xx[-1], linestyle="--", alpha=0.5, color="gray")

        ax.legend()
        ax.set_title("Operator Error for Various #Windows")
        ax.set_ylabel("$log_{10}$(Operator Error)")
        ax.set_xlabel("Time")

        fig.tight_layout()

    @staticmethod
    def compare_properrs(models, labels=None, windowlines=True):
        fig, ax = plt.subplots()

        if labels is None:
            labels = [x.W for x in models]

        for x, l in zip(models, labels):
            properrs = WeldHelper.get_properrs(x, testonly=True)
            ax.plot(np.log10(properrs), label=l)

            if windowlines:
                for xx in x.windowvals[:-1]:
                    ax.axvline(xx[-1], linestyle="--", alpha=0.5, color="gray")

        ax.legend()
        ax.set_title("Propagator Error for Various #Windows")
        ax.set_ylabel("$log_{10}$(Propagator Error)")
        ax.set_xlabel("Time")

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_latent_trajectory(weld, testnums, t=0, steps=-1, threed=False, figax=None):
        if steps == -1:
            steps = weld.T - t - 1

        data = weld.dataset.data

        arr = torch.tensor(data[testnums, :, :]).to(weld.device, dtype=torch.float32)
        outlist = weld.propagate(arr[:, t, :], t, steps)
        actual = [weld.encode_window(weld.find_window(tt), arr[:, tt, :]) for tt in range(t + 1, t + steps + 1)]

        actualpoints = torch.stack(actual, dim=2).cpu().detach()
        points = torch.stack(outlist, dim=2).cpu().detach()

        if figax is None:
            fig = plt.figure()
            if threed:
                ax = fig.add_subplot(projection="3d")
            else:
                ax = fig.add_subplot()
        else:
            fig, ax = figax
            ax.clear()

        shapes = ["*"]
        shapesactual = ["o"]

        for i in range(points.shape[0]):
            if ax.name == "3d":
                ax.scatter(actualpoints[i, 0, :], actualpoints[i, 1, :], actualpoints[i, 2, :], marker=shapesactual[i], c=range(actualpoints.shape[2]), cmap="cool", s=15)
                sc = ax.scatter(points[i, 0, :], points[i, 1, :], points[i, 2, :], marker=shapes[i], c=range(points.shape[2]), cmap="cool", s=10)
            else:
                ax.scatter(actualpoints[i, 0], actualpoints[i, 1], marker=shapesactual[i], edgecolor="black", linewidths=0.5, c=range(actualpoints.shape[2]), cmap="cool", s=15)
                sc = ax.scatter(points[i, 0], points[i, 1], marker=shapes[i], c=range(points.shape[2]), cmap="cool", s=10)

        fig.colorbar(sc, ax=ax)
        ax.set_title("o predicted, * exact")

        return (fig, ax)

    @staticmethod
    def plot_coordinate_props(weld, title="", num=None, steps=-1, i=0, difference=True, allwindows=True):
        figs = []

        if allwindows:
            ws = range(len(weld.windowvals))
        else:
            ws = [0]

        for w in ws:
            if steps == -1:
                steps = weld.windowvals[w][-1]

            data = weld.dataset.data

            if num is None:
                num = np.random.randint(data.shape[0])

            tstart = weld.windowvals[w][0]
            arr = torch.tensor(data[num, tstart:(tstart+1), :]).to(weld.device).float()

            predicted = torch.stack(weld.propagate(arr, tstart, steps)).cpu().detach()
            correct = weld.encode_window(w, torch.tensor(data[num, tstart+1:tstart+steps+1, :]).float().to(weld.device)).cpu().detach()

            colors = cm.get_cmap('tab20', predicted.shape[2])

            if difference:
                fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                ax = axes[0]
                ax1 = axes[1]
            else:
                fig, ax = plt.subplots(figsize=(5, 3))

            for j in range(predicted.shape[2]):
                ax.plot(predicted[:, 0, j], color=colors(j), label=f"Dimension {j}")
                ax.plot(correct[:, j], linestyle=":", color=colors(j))
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")

                if difference:
                    ax1.plot(np.abs(predicted[:, 0, j] - correct[:, j]), color=colors(j), label=f"Dimension {j}")
                    ax1.set_ylabel("|Predict - Correct|")
                    ax1.set_xlabel("Time")

            ax.legend()
            fig.suptitle(weld.dataset.name + " " + title + " window " + str(w+1))

            figs.append(fig)

        return figs

    @staticmethod
    def compare_errorparams(welds, labels=None, param=-1):
        if param == -1:
            param = 0
            P = welds[0].dataset.params.shape[1]

            for p in range(P):
                if np.abs(welds[0].dataset.params[0, p] - welds[0].dataset.params[1, p]) > 0:
                    param = p
                    break

        if labels is None:
            labels = [len(x.aes) for x in welds]

        fig, ax = plt.subplots()
        for lbl, weld in zip(labels, welds):
            op = WeldHelper.get_operrs(weld, fullerror=True)
            l2error = np.linalg.norm(np.asarray(op), axis=0)

            ax.scatter(weld.dataset.params[:, param], l2error, label=lbl, s=2)

        ax.set_title(f"Error vs. Parameter {param}")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Operator Error")
        ax.legend()

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_errorparams(weld, param=-1):
        if param == -1:
            param = 0
            P = weld.dataset.params.shape[1]

            for p in range(P):
                if np.abs(weld.dataset.params[0, p] - weld.dataset.params[1, p]) > 0:
                    param = p
                    break

        op = WeldHelper.get_operrs(weld, fullerror=True)
        l2error = np.linalg.norm(np.asarray(op), axis=0)

        fig, ax = plt.subplots()
        ax.scatter(weld.dataset.params[:, param], l2error, s=2)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Operator Error")

        fig.tight_layout()
        return fig

