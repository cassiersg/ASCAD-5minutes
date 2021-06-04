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

# Attack on the ASCAD database.
# This performs a side-channel based key-recovery of the masked implementation,
# based on traces from the public ASCAD database.
# The attack is based on the tools provided by the SCAlib library (SNR
# computation for POI selection, LDA/gaussian templates modelling and SASCA).
# See:
#  https://github.com/ANSSI-FR/ASCAD
#  https://github.com/simple-crypto/SCALib/

import argparse
import itertools as it
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

##########################
### Parse command-line ###
##########################

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "-w",
    "--window",
    type=str,
    default="0,250000",
    help="'start,end' window for taking only a part of the traces (default: whole trace).",
)
parser.add_argument(
    "-n",
    "--ntracesattack",
    type=str,
    default="1,2",
    help="Number of traces for the attack. Must be integers separated with comma.",
)
parser.add_argument(
    "-a",
    "--averageattack",
    type=int,
    default=100,
    help="Number of runs for each attack parameter set (default: 100).",
)
parser.add_argument(
    "-b",
    "--nbytes",
    type=int,
    default=14,
    help="Number of bytes in the attack. (1 to 14, default 14)",
)
strategy_help = """Which variables to attack (default=1):
1: all
2: only s-box input shares (before re-masking)
3: only s-box input (re-masked with common rin mask)
4: only s-box output shares (with common mask)
5: only s-box output shares (after re-masking)
6: all except s-box input before re-masking"""
parser.add_argument(
    "-s",
    "--strategy",
    type=int,
    default=1,
    help=strategy_help,
)
parser.add_argument(
    "-p",
    "--profile",
    type=int,
    default=5000,
    help="Number of traces used for profiling",
)
parser.add_argument(
    "--database",
    type=str,
    default="atmega8515-raw-traces.h5",
    help="Location of the 'raw traces' ASCAD database file (default: ./atmega8515-raw-traces.h5)",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    help="Save intermediate results (default: fast).",
)
parser.add_argument("--no-save", dest="save", action="store_false")
parser.add_argument(
    "--show",
    dest="show",
    action="store_true",
    help="Show a plot of the final key rank (default: don't show).",
)
parser.add_argument(
    "computations",
    type=str,
    default="snr,model,attack",
    help="Which computations to perform: snr, model, attack, plot and/or all (comma-separated list).",
)
parser.set_defaults(save=True, show=False)
args = parser.parse_args()

W_START, W_END = [int(x.strip()) for x in args.window.split(",")]
WINDOW = np.arange(W_START, W_END, dtype=np.int)
NS = len(WINDOW)
NTRACES_ATTACK = [int(x.strip()) for x in args.ntracesattack.split(",")]
COMPUTATIONS = set(x.strip().lower() for x in args.computations.split(","))
for x in COMPUTATIONS:
    if x not in ("snr", "model", "attack", "plot", "all"):
        raise ValueError(f"Computation '{x}' not supported.")
if "all" in COMPUTATIONS:
    COMPUTATIONS = {"snr", "model", "attack", "plot"}


########################################
### General parameters of the attack ###
########################################

# number of pois
N_POI = 500
# number of dimensions of reduced dimensionality space for LDA
LDA_DIM = 10

# Storage directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_SUFFIX = f"{W_START}_{W_END}_p{args.profile}.pkl"
SNR_FILE = os.path.join(DATA_DIR, f"{MODEL_SUFFIX}.pkl")
MODELS_FILE = os.path.join(DATA_DIR, f"models_{MODEL_SUFFIX}.pkl")
ATTACK_NAME = "{ms}_{nb}_s{strat}_a{av}_n{ntr}".format(
    ms=MODEL_SUFFIX,
    nb=args.nbytes,
    strat=args.strategy,
    av=args.averageattack,
    ntr=",".join(map(str, NTRACES_ATTACK)),
)
ATTACK_RESULT_FILE = os.path.join(DATA_DIR, f"attack_{ATTACK_NAME}.pkl")
PLOT_FILE = os.path.join(DATA_DIR, f"plot_{ATTACK_NAME}.pdf")

# variable labels
VARIABLES = [
    f"{base}_{x}"
    for base in ("x0", "x1", "xrin", "yrout", "y0", "y1")
    for x in range(args.nbytes)
] + ["rin", "rout"]

# var to include in graph
VARIABLES_INCLUDE = {
    1: ["x0", "x1", "rin", "xrin", "yrout", "rout", "y0", "y1"],
    2: ["x0", "x1"],
    3: ["rin", "xrin"],
    4: ["yrout", "rout"],
    5: ["y0", "y1"],
    6: ["rin", "xrin", "yrout", "rout", "y0", "y1"],
}
VARIABLES_INCLUDE = VARIABLES_INCLUDE.get(args.strategy)
if VARIABLES_INCLUDE is None:
    raise ValueError(f"Unknown strategy {args.strategy}.")

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


##############################
### ASCAD Database loading ###
##############################

f_database = h5py.File(args.database, "r")


def get_traces(start, l, window, fixed_key=False, verbose=False):
    """Load traces and labels from ASCAD database."""
    I = np.arange(start, start + l)
    if verbose:
        print(f"Loading traces ({l*len(window)*2/1e9} GB)...")
    traces = f_database["traces"][start : start + l, window].astype(np.int16)
    masks = f_database["metadata"]["masks"][I, 2:16].astype(np.uint16)
    key = f_database["metadata"]["key"][I, 2:].astype(np.uint16)
    plaintext = f_database["metadata"]["plaintext"][I, 2:].astype(np.uint16)
    if fixed_key:
        k = np.random.randint(0, 256, 14, dtype=np.uint16)
        k = np.tile(k, (l, 1))
        plaintext = plaintext ^ k ^ key
        key = k
    x0 = key ^ plaintext ^ masks
    x1 = masks
    rin = f_database["metadata"]["masks"][I, 16].astype(np.uint16)
    xrin = ((key ^ plaintext).T ^ rin).T
    y0 = SBOX[key ^ plaintext] ^ masks
    y1 = masks
    rout = f_database["metadata"]["masks"][I, 17].astype(np.uint16)
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
    return traces, labels


#######################
### SNR computation ###
#######################


def compute_snr():
    # Create a model for every variable
    models = {v: dict() for v in VARIABLES}
    traces, labels = get_traces(0, args.profile, WINDOW, verbose=True)
    for v, m in tqdm(models.items(), total=len(models), desc="SNR Variables"):
        snr = SNR(np=1, nc=256, ns=NS)
        x = labels[v].reshape((args.profile, 1))
        # Note: if the traces do not fit in RAM, you can call multiple times fit_u
        # on the same SNR object to do incremental SNR computation.
        snr.fit_u(traces, x)
        m["SNR"] = snr.get_snr()[0, :]
    return models


############################################
### LDA (gaussian templates) computation ###
############################################


def compute_templates(models_snr, verbose=False):
    if verbose:
        print("Compute POIs...")
    for k, m in models_snr.items():
        # Avoid NaN in case of scope over-range
        np.nan_to_num(m["SNR"])
        m["poi"] = np.argsort(m["SNR"])[-N_POI:].astype(np.uint32)
        m["poi"].sort()
        m.pop("SNR")
    traces, labels = get_traces(0, args.profile, WINDOW, verbose=True)
    if verbose:
        print("Fit LDA...")
        vs = list(models_snr.keys())
    mlda = scalib.modeling.MultiLDA(
        ncs=len(models_snr) * [256],
        ps=len(models_snr) * [LDA_DIM],
        pois=[models_snr[v]["poi"] for v in vs],
        gemm_mode=2,
    )
    x = np.array([labels[v] for v in vs]).transpose()
    mlda.fit_u(traces, x)
    if verbose:
        print("Solve LDA...")
    mlda.solve()
    for lda, v in zip(mlda.ldas, vs):
        models_snr[v]["lda"] = lda
    return models_snr


#################################
### SASCA and rank estimation ###
#################################

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


def run_attack(traces, labels):
    """Run a SASCA attack on the given traces and evaluate its performance.

    :param labels:
        contains the labels of all the variables
    :returns:
        the log2 of the rank of the true key
    """
    # correct secret key
    secret_key = [labels[f"k_{i}"][0] for i in range(args.nbytes)]
    # distribution for each of the key bytes
    key_distribution = []
    # Run a SASCA for each S-Box
    for i in range(args.nbytes):
        sasca = scalib.attacks.SASCAGraph(SASCA_GRAPH, n=traces.shape[0])
        sasca.set_table("sbox", SBOX.astype(np.uint32))
        # Set the labels for the plaintext byte
        sasca.set_public(f"p", labels[f"p_{i}"].astype(np.uint32))
        for var in VARIABLES_INCLUDE:
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
    rmin, r, rmax = scalib.postprocessing.rank_accuracy(
        -np.log2(key_distribution), secret_key, max_nb_bin=2**20
    )
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    return lr


def run_all_attacks(models):
    # Logs of the ranks
    lrs = np.zeros((args.averageattack, len(NTRACES_ATTACK)))
    m = max(NTRACES_ATTACK)
    # Offset in tracesto no fall on training traces
    traces, labels = get_traces(
        args.profile, args.averageattack * m, WINDOW, fixed_key=True
    )
    # TODO this loop could be parallelized if the scalib backend were more
    # efficiently parallelizable (looks like it blocks somewhere in
    # python-related code).
    attack_cases = list(
        it.product(range(args.averageattack), enumerate(NTRACES_ATTACK))
    )
    for a, (i, n) in tqdm(attack_cases):
        l = {k: val[a * m : a * m + n] for k, val in labels.items()}
        lrs[a, i] = run_attack(traces[a * m : a * m + n, :], l)
    return lrs


#############################
### Plot rank estimations ###
#############################


def plot_attack(attack_res):
    n = np.tile(attack_res["ntracesattack"], (attack_res["lrs"].shape[0], 1))
    plt.boxplot(
        2 ** attack_res["lrs"], labels=attack_res["ntracesattack"], autorange=True
    )
    plt.grid(True, which="both", ls="--")
    plt.yscale("log", base=2)
    plt.xlabel("number of attack traces")
    plt.ylabel("key rank")
    plt.ylim((1/1.1, 1.1 * 2.0 ** (8 * attack_res["nbytes"])))
    ticks = {1: range(0, 9), 2: range(0, 17, 2)}.get(
        attack_res["nbytes"], range(0, 8 * attack_res["nbytes"] + 1, 8)
    )
    plt.yticks([2.0 ** x for x in ticks])


if __name__ == "__main__":
    if "snr" in COMPUTATIONS:
        print("Start SNR estimation")
        models = compute_snr()
        if args.save:
            print("Saving SNRs...")
            with open(SNR_FILE, "wb") as f:
                pickle.dump(models, f)
    if "model" in COMPUTATIONS:
        print("Start modeling")
        if "snr" not in COMPUTATIONS:
            # load models containing "SNR" field
            with open(SNR_FILE, "rb") as f:
                models = pickle.load(f)
        compute_templates(models, verbose=True)
        if args.save:
            print("Saving models...")
            with open(MODELS_FILE, "wb") as f:
                pickle.dump(models, f)
    if "attack" in COMPUTATIONS:
        print("Start attack")
        if "model" not in COMPUTATIONS:
            with open(MODELS_FILE, "rb") as f:
                models = pickle.load(f)
        lrs = run_all_attacks(models)
        print("N traces : min log2(rank) | median log2(rank) | max log2(rank)")
        for i, n in enumerate(NTRACES_ATTACK):
            min_rank = np.min(lrs[:, i])
            median_rank = np.median(lrs[:, i])
            max_rank = np.max(lrs[:, i])
            print(f"{n:8} | {min_rank:14.1f} | {median_rank:17.1f} | {max_rank:14.1f}")
        attack_res = {
            "lrs": lrs,
            "strategy": args.strategy,
            "nbytes": args.nbytes,
            "ntracesattack": NTRACES_ATTACK,
        }
        if args.save:
            print("Saving attack results...")
            with open(ATTACK_RESULT_FILE, "wb") as f:
                pickle.dump(attack_res, f)
    if "plot" in COMPUTATIONS:
        if "attack" not in COMPUTATIONS:
            with open(ATTACK_RESULT_FILE, "rb") as f:
                attack_res = pickle.load(f)
        plt.figure()
        plot_attack(attack_res)
        if args.save:
            plt.savefig(PLOT_FILE, bbox_inches="tight", pad_inches=0.02)
        if args.show:
            plt.show()
