#!/usr/bin/env python3
# Copyright 2021 UCLouvain
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE

# Attack on the ASCAD database, multiple attacks and caching version.
# See attack.py for a more simple script.

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools as it
import functools as ft
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scalib.metrics import SNR
import scalib.modeling
import scalib.attacks
import scalib.postprocessing
from tqdm import tqdm

def parse_args():
    "Parse command line arguments."
    parser = argparse.ArgumentParser(
            description=
            "Attack against the ASCAD dataset. "
            "Runs attacks with many parameter sets at once by making a grid "
            "of all given parameters combinations. "
            "Parameters that can have multiple values are number of attack "
            "traces, number of profiling traces, number of POIs, LDA output "
            "dimension."
            )
    parser.add_argument(
        "--window",
        type=str,
        default="0,250000",
        help="'start,end' window for taking only a part of the traces (default: whole trace).",
    )
    parser.add_argument(
        "--ntracesattack",
        type=str,
        default="1,2",
        help=
        "Number of traces for the attack. "
        "Must be integers separated with comma (default: %(default)s).",
    )
    parser.add_argument(
        "--averageattack",
        type=int,
        default=100,
        help="Number of runs for each attack parameter set (default: %(default)s).",
    )
    parser.add_argument(
        "--nbytes",
        type=int,
        default=14,
        help="Number of bytes in the attack (1 to 14, default: %(default)s).",
    )
    strategy_help = """Which variables to attack (default=1):
    1: all
    2: only s-box input shares (before re-masking)
    3: only s-box input (re-masked with common rin mask)
    4: only s-box output shares (with common mask)
    5: only s-box output shares (after re-masking)
    6: all except s-box input before re-masking"""
    parser.add_argument(
        "--strategy",
        type=int,
        default=1,
        help=strategy_help,
    )
    parser.add_argument(
        "--ntracesprofile",
        type=str,
        default='5000',
        help="Number of traces used for profiling "
        "(comma-separated ints, default: %(default)s).",
    )
    parser.add_argument(
        "--poi",
        type=str,
        default='512',
        help="Number of POIs for each variable "
        "(comma-separated ints, default: %(default)s).",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default='8',
        help="Dimensionality of projected space for LDA "
        "(comma-separated ints, default: %(default)s).",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="atmega8515-raw-traces.h5",
        help="Location of the 'raw traces' ASCAD database file (default: %(default)s).",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default="./res_store",
        help="Location of the result store directory (default: %(default)s).",
    )
    parser.add_argument(
        "--fast-snr",
        action="store_true",
        help="Use a faster SNR computation at the expense of large RAM usage (default: false).",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the result figures (default: don't show).",
    )
    parser.add_argument(
        "computations",
        type=str,
        help=
        f"Which computations to perform: {', '.join(Settings.all_computations)} and/or all. "
        "Computations which are not performed are loaded from the result store."
    )
    return parser.parse_args(namespace=Settings())

class Settings:
    all_computations = {"snr", "model", "attack", "sr-boxplot", "sr-map"}
    def parse_settings(self):
        self.w_start, self.w_end = [int(x.strip()) for x in self.window.split(",")]
        self.window = np.arange(self.w_start, self.w_end, dtype=np.int)
        self.ns = len(self.window)
        self.ntraces_attacks = [int(x.strip()) for x in self.ntracesattack.split(",")]
        # var to include in graph
        included_variables = {
            1: ["x0", "x1", "rin", "xrin", "yrout", "rout", "y0", "y1"],
            2: ["x0", "x1"],
            3: ["rin", "xrin"],
            4: ["yrout", "rout"],
            5: ["y0", "y1"],
            6: ["rin", "xrin", "yrout", "rout", "y0", "y1"],
        }
        self.included_variables = included_variables.get(self.strategy)
        if self.included_variables is None:
            raise ValueError(f"Unknown strategy {strategy}.")
        self.ntraces_profiles = [int(x.strip()) for x in self.ntracesprofile.split(",")]
        self.npois = [int(x.strip()) for x in self.poi.split(",")]
        self.lda_dims = [int(x.strip()) for x in self.dim.split(",")]
        # Storage directory
        os.makedirs(self.store_dir, exist_ok=True)
        self.computations = set(x.strip().lower() for x in self.computations.split(","))
        for x in self.computations - self.all_computations - {"all"}:
            raise ValueError(f"Computation '{x}' not supported.")
        if "all" in self.computations:
            self.computations = self.all_computations

    def snr_suffix(self, ntraces_profile):
        return f"{self.w_start}_{self.w_end}_p{ntraces_profile}"

    def snr_file(self, ntraces_profile):
        return os.path.join(self.store_dir, f"snr_{self.snr_suffix(ntraces_profile)}.pkl")

    def model_suffix(self, ntraces_profile, npoi, lda_dim):
        return f"{self.snr_suffix(ntraces_profile)}_poi{npoi}_dim{lda_dim}"

    def model_file(self, ntraces_profile, npoi, lda_dim):
        return os.path.join(
                self.store_dir,
                f"models_{self.model_suffix(ntraces_profile, npoi, lda_dim)}.pkl"
                )

    def attack_suffix(self, ntraces_profile, ntraces_attack, npoi, lda_dim):
        return "{ms}_{nb}_s{strat}_a{av}_na{ntra}".format(
                ms=self.model_suffix(ntraces_profile, npoi, lda_dim),
                nb=self.nbytes,
                strat=self.strategy,
                av=self.averageattack,
                ntra=ntraces_attack,
                )

    def attack_file(self, *args):
        return os.path.join(self.store_dir,
                f"attack_{self.attack_suffix(*args)}.pkl")

    def variables(self):
        # All shares labels
        return [
                f"{base}_{x}"
                for base in ("x0", "x1", "xrin", "yrout", "y0", "y1")
                for x in range(self.nbytes)
                ] + ["rin", "rout"]

# fmt: off
SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB,
        0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4,
        0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71,
        0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
        0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6,
        0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB,
        0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45,
        0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44,
        0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A,
        0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49,
        0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
        0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 0xBA, 0x78, 0x25,
        0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E,
        0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1,
        0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB,
        0x16,
    ],
    dtype=np.uint16,
)
# fmt: on

@ft.lru_cache(maxsize=None)
def load_database(settings):
    return h5py.File(settings.database, "r")

def var_labels(key, plaintext, masks, rin, rout):
    "Compute value of variables of interest based on ASCAD metadata."
    x0 = key ^ plaintext ^ masks
    x1 = masks
    xrin = ((key ^ plaintext).T ^ rin).T
    y0 = SBOX[key ^ plaintext] ^ masks
    y1 = masks
    yrout = (SBOX[(key ^ plaintext).T] ^ rout).T
    labels = {}
    for i in range(14):
        labels[f"k_{i}"] = key[:, i]
        labels[f"p_{i}"] = plaintext[:, i]
        labels[f"x0_{i}"] = x0[:, i]
        labels[f"x1_{i}"] = x1[:, i]
        labels[f"y0_{i}"] = y0[:, i]
        labels[f"y1_{i}"] = y1[:, i]
        labels[f"xrin_{i}"] = xrin[:, i]
        labels[f"yrout_{i}"] = yrout[:, i]
    labels[f"rout"] = rout[:]
    labels[f"rin"] = rin[:]
    return labels

@ft.lru_cache(maxsize=None)
def get_traces(settings, start, l, fixed_key=False):
    """Load traces and labels from ASCAD database.

    fixed_key: transform a variable key dataset into a fixed key one by
    exploiting key^plaintext leakage invariance (for multi-trace attacks)
    """
    I = np.arange(start, start + l)
    f_database = load_database(settings)
    traces = f_database["traces"][start : start + l, settings.window].astype(np.int16)
    key = f_database["metadata"]["key"][I, 2:].astype(np.uint16)
    plaintext = f_database["metadata"]["plaintext"][I, 2:].astype(np.uint16)
    masks = f_database["metadata"]["masks"][I, 2:16].astype(np.uint16)
    rin = f_database["metadata"]["masks"][I, 16].astype(np.uint16)
    rout = f_database["metadata"]["masks"][I, 17].astype(np.uint16)
    if fixed_key:
        k = np.random.randint(0, 256, 14, dtype=np.uint16)
        k = np.tile(k, (l, 1))
        plaintext = plaintext ^ k ^ key
        key = k
    labels = var_labels(key, plaintext, masks, rin, rout)
    return traces, labels

def target_variables(byte):
    """variables that will be profiled"""
    return ["rin", "rout"] + [
            f"{base}_{byte}" for base in ("x0", "x1", "xrin", "yrout", "y0", "y1")
            ]

def compute_snr(settings, ntraces_profile):
    """Returns the SNR of the traces samples for each target variable."""
    snrs = {v: dict() for i in range(settings.nbytes) for v in target_variables(i)}
    traces, labels = get_traces(settings, start=0, l=ntraces_profile, fixed_key=False)
    if settings.fast_snr:
        snr = SNR(np=len(snrs), nc=256, ns=settings.ns)
        labels_full = np.zeros((traces.shape[0], len(snrs)))
        variables = list(settings.variables())
        x = np.array([labels[v] for v in variables]).T
        snr.fit_u(traces, x)
        snrs_raw = snr.get_snr()
        # Avoid NaN in case of scope over-range
        np.nan_to_num(snrs_raw, nan=0.0)
        for i, v in enumerate(variables):
            snrs[v]["SNR"] = snrs_raw[i, :]
    else:
        for v, m in tqdm(snrs.items(), total=len(snrs), desc="SNR Variables"):
            snr = SNR(np=1, nc=256, ns=settings.ns)
            x = labels[v].reshape((ntraces_profile, 1))
            # Note: if the traces do not fit in RAM, you can call multiple times fit_u
            # on the same SNR object to do incremental SNR computation.
            snr.fit_u(traces, x)
            m["SNR"] = snr.get_snr()[0, :]
            # Avoid NaN in case of scope over-range
        np.nan_to_num(m["SNR"], nan=0.0)
    return snrs

def make_snr(settings):
    for ntraces_profile in tqdm(settings.ntraces_profiles, desc="SNR_np"):
        snrs = compute_snr(settings, ntraces_profile)
        with open(settings.snr_file(ntraces_profile), 'wb') as f:
            pickle.dump(snrs, f)

@ft.lru_cache(maxsize=None)
def load_snr(settings, ntraces_profile):
    "Load SNR from the store."
    with open(settings.snr_file(ntraces_profile), 'rb') as f:
        return pickle.load(f)

def preload_snr(settings):
    return {
            ntraces_profile: load_snr(settings, ntraces_profile)
            for ntraces_profile in settings.ntraces_profiles
            }

def compute_templates(settings, ntraces_profile, npoi, lda_dim):
    "LDA and gaussian templates computation"
    snrs = load_snr(settings, ntraces_profile)
    models = dict()
    for k, snr in snrs.items():
        poi =  np.argsort(snr["SNR"])[-npoi:].astype(np.uint32)
        poi.sort()
        models[k] = {"poi": poi}
    traces, labels = get_traces(settings, 0, ntraces_profile)
    vs = list(models.keys())
    mlda = scalib.modeling.MultiLDA(
        ncs=len(models) * [256],
        ps=len(models) * [lda_dim],
        pois=[models[v]["poi"] for v in vs],
        gemm_mode=4,
    )
    x = np.array([labels[v] for v in vs]).transpose()
    mlda.fit_u(traces, x)
    mlda.solve()
    for lda, v in zip(mlda.ldas, vs):
        models[v]["lda"] = lda
    return models

def make_models(settings):
    profile_cases = list(it.product(settings.ntraces_profiles, settings.npois, settings.lda_dims))
    for pc in tqdm(profile_cases, desc="profiling cases"):
        models = compute_templates(settings, *pc)
        fname = settings.model_file(*pc)
        with open(fname, 'wb') as f:
            pickle.dump(models, f)

@ft.lru_cache(maxsize=None)
def load_models(settings, ntraces_profile, npoi, lda_dim):
    fname = settings.model_file(ntraces_profile, npoi, lda_dim)
    with open(fname, 'rb') as f:
        return pickle.load(f)

def preload_models(settings):
    return {
            pc: load_models(settings, *pc)
            for pc in it.product(settings.ntraces_profiles, settings.npois, settings.lda_dims)
            }

SASCA_GRAPH = """
NC 256
TABLE sbox

VAR MULTI x0
VAR MULTI x1
VAR MULTI x
VAR MULTI xp
VAR MULTI xrin
VAR MULTI rout
VAR MULTI rin

VAR MULTI y0
VAR MULTI y1
VAR MULTI y
VAR MULTI yp
VAR MULTI yrout

VAR MULTI p
VAR SINGLE k

PROPERTY x = p ^ k
PROPERTY x = x0 ^ x1
PROPERTY x = rin ^ xrin

PROPERTY y = sbox[x]
PROPERTY y = y0 ^ y1
PROPERTY y = rout ^ yrout
"""

@ft.lru_cache(maxsize=None)
def sasca_graph(settings, nattack_traces):
    sasca = scalib.attacks.SASCAGraph(SASCA_GRAPH, n=nattack_traces)
    sasca.set_table("sbox", SBOX.astype(np.uint32))
    return sasca

def attack(settings, traces, labels, models):
    """Run a SASCA attack on the given traces and evaluate its performance.

    :param labels:
        contains the labels of all the variables
    :returns:
        the true key and the distribution estimated by the attack
    """
    # correct secret key
    secret_key = [labels[f"k_{i}"][0] for i in range(settings.nbytes)]
    # distribution for each of the key bytes
    key_distribution = []
    # Run a SASCA for each S-Box
    for i in range(settings.nbytes):
        sasca = copy.deepcopy(sasca_graph(settings, traces.shape[0]))
        # Set the labels for the plaintext byte
        sasca.set_public(f"p", labels[f"p_{i}"].astype(np.uint32))
        for var in settings.included_variables:
            if var in ("rin", "rout"):
                model = models[var]
            else:
                model = models[var + f"_{i}"]
            prs = model["lda"].predict_proba(traces[:, model["poi"]])
            sasca.set_init_distribution(var, prs)
        sasca.run_bp(it=3)
        distribution = sasca.get_distribution(f"k")[0, :]
        key_distribution.append(distribution)
    key_distribution = np.array(key_distribution)
    return secret_key, key_distribution

def run_attack(settings, traces, labels, models):
    """Run a SASCA attack on the given traces and evaluate its performance.

    :param labels:
        contains the labels of all the variables
    :returns:
        the log2 of the rank of the true key
    """
    secret_key, key_distribution = attack(settings, traces, labels, models)
    rmin, r, rmax = scalib.postprocessing.rank_accuracy(
        -np.log2(key_distribution), secret_key, max_nb_bin=2**20
    )
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    return lr

def make_attacks(settings):
    # Offset in traces to no fall on training traces
    traces, labels = get_traces(
            settings,
            max(settings.ntraces_profiles),
            settings.averageattack * max(settings.ntraces_attacks),
            fixed_key=True
            )
    attack_cases = list(it.product(
        settings.ntraces_profiles,
        settings.ntraces_attacks,
        settings.npois,
        settings.lda_dims,
        ))
    # TODO this loop seems to not be able to fully exploit parallelism.
    # (seeing a only ~8x speedup on a large CPU)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=settings.averageattack*len(attack_cases)) as progress:
        futures_all = dict()
        for attack_case in attack_cases:
            ntraces_profile, ntraces_attack, npoi, lda_dim = attack_case
            models = load_models(settings, ntraces_profile, npoi, lda_dim)
            futures = []
            for a in range(settings.averageattack):
                future = executor.submit(
                        run_attack,
                        settings,
                        traces[a*ntraces_attack:(a+1)*ntraces_attack, :],
                        {
                            k: val[a * ntraces_attack : (a+1) * ntraces_attack]
                            for k, val in labels.items()
                            },
                        models
                        )
                future.add_done_callback(lambda _: progress.update())
                futures.append(future)
            futures_all[attack_case] = futures

        for attack_case, futures in futures_all.items():
            attack_results = np.array([future.result() for future in futures])
            fname = settings.attack_file(*attack_case)
            with open(fname, 'wb') as f:
                pickle.dump(attack_results, f)

@ft.lru_cache(maxsize=None)
def load_attacks(settings, ntraces_profile, ntraces_attack, npoi, lda_dim):
    fname = settings.attack_file(ntraces_profile, ntraces_attack, npoi, lda_dim)
    with open(fname, 'rb') as f:
        return pickle.load(f)

def preload_attacks(settings):
    attack_cases = list(it.product(
        settings.ntraces_profiles,
        settings.ntraces_attacks,
        settings.npois,
        settings.lda_dims,
        ))
    return {ac: load_attacks(settings, *ac) for ac in attack_cases}

#############################
### Plots ###
#############################

def rank_boxplot(settings, ntraces_attacks, attack_results):
    plt.boxplot(
            [2**ar for ar in attack_results],
            labels=ntraces_attacks, autorange=True
    )
    plt.grid(True, which="both", ls="--")
    plt.yscale("log", base=2)
    plt.xlabel("number of attack traces")
    plt.ylabel("key rank")
    plt.ylim((1/1.1, 1.1 * 2.0 ** (8 * settings.nbytes)))
    ticks = {1: range(0, 9), 2: range(0, 17, 2)}.get(
        settings.nbytes, range(0, 8 * settings.nbytes + 1, 8)
    )
    plt.yticks([2.0 ** x for x in ticks])

def make_sr_boxplot(settings):
    plot_cases = dict()
    for attack_case, attack_res in preload_attacks(settings).items():
        ntraces_profile, ntraces_attack, npoi, lda_dims = attack_case
        plot_cases.setdefault((ntraces_profile, npoi, lda_dims),
                list()).append((ntraces_attack, attack_res))
    for plot_case, attacks in plot_cases.items():
        ntraces_profile, npoi, lda_dims = plot_case
        attack_case = ntraces_profile, 0, npoi, lda_dims
        figname = f"ranks_{settings.attack_suffix(*attack_case)}"
        plt.figure(figname)
        rank_boxplot(settings, *list(zip(*attacks)))
        plt.savefig(
                os.path.join(settings.store_dir, figname+".pdf"),
                bbox_inches="tight",
                pad_inches=0.02
                )

def success_rate(ranks, min_rank=1):
    return np.sum(ranks <= min_rank) / ranks.size

def make_sr_map(settings):
    for ntraces_profile, ntraces_attack in it.product(
            settings.ntraces_profiles, settings.ntraces_attacks
            ):
        plot_matrix = np.array([
                [
                    success_rate(2**np.array(
                        load_attacks(settings, ntraces_profile, ntraces_attack, npoi, lda_dim)
                        ))
                    for lda_dim in settings.lda_dims]
                for npoi in settings.npois
                ])
        attack_case = ntraces_profile, ntraces_attack, 0, 0
        figname = f"srs_{settings.attack_suffix(*attack_case)}"
        fig, ax = plt.subplots(num=figname)
        ax.matshow(plot_matrix, cmap=plt.cm.Greys, origin='lower', vmin=0.0, vmax=1.0)
        for i, npoi in enumerate(settings.npois):
            for j, lda_dim in enumerate(settings.lda_dims):
                c = plot_matrix[i][j]
                ax.text(j, i, f"{c:.2f}", va='center', ha='center', color='red')
        plt.xlabel("LDA dim")
        plt.ylabel("#POI")
        ax.set_xticks(range(len(settings.lda_dims)))
        ax.set_xticklabels(map(str, settings.lda_dims))
        ax.set_yticks(range(len(settings.npois)))
        ax.set_yticklabels(map(str, settings.npois))
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(
                os.path.join(settings.store_dir, figname+".pdf"),
                bbox_inches="tight",
                pad_inches=0.02
                )

if __name__ == "__main__":
    settings = parse_args()
    settings.parse_settings()
    if "snr" in settings.computations:
        print("Start SNR estimation")
        make_snr(settings)
    if "model" in settings.computations:
        print("Start modeling")
        # We use preloading of lru_caches to avoid double-loading due to
        # multi-threaded computations
        preload_snr(settings)
        make_models(settings)
    if "attack" in settings.computations:
        print("Start attack")
        preload_models(settings)
        make_attacks(settings)
    if "sr-boxplot" in settings.computations:
        preload_attacks(settings)
        make_sr_boxplot(settings)
    if "sr-map" in settings.computations:
        preload_attacks(settings)
        make_sr_map(settings)
    if {"sr-boxplot", "sr-map"}.intersection(settings.computations) and settings.show:
        plt.show()
