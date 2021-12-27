#! /usr/bin/env python3
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
import copy
import collections
import functools as ft

import h5py
import numpy as np
from scalib.metrics import SNR
import scalib.modeling
import scalib.attacks
import scalib.postprocessing
from tqdm import tqdm

class Settings:
    """Command-line settings (hashable object)."""
    pass

def parse_args():
    parser = argparse.ArgumentParser(
            description="Attack against the ASCAD dataset."
            )
    parser.add_argument(
        "--attacks",
        type=int,
        default=100,
        help="Number of attack runs (default: %(default)s).",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=5000,
        help="Number of traces used for profiling (default: %(default)s).",
    )
    parser.add_argument(
        "--poi",
        type=int,
        default=512,
        help="Number of POIs for each variable (default: %(default)s).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=8,
        help="Dimensionality of projected space for LDA (default: %(default)s).",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="./atmega8515-raw-traces.h5",
        help="Location of the 'raw traces' ASCAD file (default: %(default)s).",
    )
    return parser.parse_args(namespace=Settings())

# number of bytes to attack
NBYTES = 14
def target_variables(byte):
    """variables that will be profiled"""
    return ["rin", "rout"] + [
            f"{base}_{byte}" for base in ("x0", "x1", "xrin", "yrout", "y0", "y1")
            ]
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
def get_traces(settings, start, l):
    """Load traces and labels from ASCAD database."""
    I = np.arange(start, start + l)
    f_database = load_database(settings)
    traces = f_database["traces"][start : start + l, :].astype(np.int16)
    key = f_database["metadata"]["key"][I, 2:].astype(np.uint16)
    plaintext = f_database["metadata"]["plaintext"][I, 2:].astype(np.uint16)
    masks = f_database["metadata"]["masks"][I, 2:16].astype(np.uint16)
    rin = f_database["metadata"]["masks"][I, 16].astype(np.uint16)
    rout = f_database["metadata"]["masks"][I, 17].astype(np.uint16)
    labels = var_labels(key, plaintext, masks, rin, rout)
    return traces, labels


def compute_snr(settings):
    """Returns the SNR of the traces samples for each target variable."""
    snrs = {v: dict() for i in range(NBYTES) for v in target_variables(i)}
    traces, labels = get_traces(settings, start=0, l=settings.profile)
    for v, m in tqdm(snrs.items(), total=len(snrs), desc="SNR Variables"):
        snr = SNR(np=1, nc=256, ns=traces.shape[1])
        x = labels[v].reshape((settings.profile, 1))
        # Note: if the traces do not fit in RAM, you can call multiple times fit_u
        # on the same SNR object to do incremental SNR computation.
        snr.fit_u(traces, x)
        m["SNR"] = snr.get_snr()[0, :]
        # Avoid NaN in case of scope over-range
        np.nan_to_num(m["SNR"], nan=0.0)
    return snrs

def compute_templates(settings, snrs):
    """Compute the POIs, LDA and gaussian template for all variables."""
    models = dict()
    # Select POIs
    for k, m in snrs.items():
        poi =  np.argsort(m["SNR"])[-settings.poi:].astype(np.uint32)
        poi.sort()
        models[k] = {"poi": poi}
    traces, labels = get_traces(settings, start=0, l=settings.profile)
    vs = list(models.keys())
    mlda = scalib.modeling.MultiLDA(
        ncs=len(models) * [256],
        ps=len(models) * [settings.dim],
        pois=[models[v]["poi"] for v in vs],
    )
    x = np.array([labels[v] for v in vs]).transpose()
    mlda.fit_u(traces, x)
    mlda.solve()
    for lda, v in zip(mlda.ldas, vs):
        models[v]["lda"] = lda
    return models

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
def sasca_graph():
    sasca = scalib.attacks.SASCAGraph(SASCA_GRAPH, n=1)
    sasca.set_table("sbox", SBOX.astype(np.uint32))
    return sasca

def attack(traces, labels, models):
    """Run a SASCA attack on the given traces and evaluate its performance.
    Returns the true key and the byte-wise key distribution estimated by the attack.
    """
    # correct secret key
    secret_key = [labels[f"k_{i}"][0] for i in range(NBYTES)]
    # distribution for each of the key bytes
    key_distribution = []
    # Run a SASCA for each S-Box
    for i in range(NBYTES):
        sasca = copy.deepcopy(sasca_graph())
        # Set the labels for the plaintext byte
        sasca.set_public(f"p", labels[f"p_{i}"].astype(np.uint32))
        for var in target_variables(i):
            model = models[var]
            prs = model["lda"].predict_proba(traces[:, model["poi"]])
            sasca.set_init_distribution(var.split('_')[0], prs)
        sasca.run_bp(it=3)
        distribution = sasca.get_distribution(f"k")[0, :]
        key_distribution.append(distribution)
    key_distribution = np.array(key_distribution)
    return secret_key, key_distribution

def run_attack_eval(traces, labels, models):
    """Run a SASCA attack on the given traces and evaluate its performance.
    Returns the log2 of the rank of the true key.
    """
    secret_key, key_distribution = attack(traces, labels, models)
    rmin, r, rmax = scalib.postprocessing.rank_accuracy(
        -np.log2(key_distribution), secret_key, max_nb_bin=2**20
    )
    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    return lr

def run_attacks_eval(settings, models):
    """Return the list of the rank of the true key for each attack."""
    # Offset in traces to no attack the training traces
    traces, labels = get_traces(settings, start=settings.profile, l=settings.attacks)
    return 2**np.array(list(tqdm(map(
        lambda a: run_attack_eval(
            traces[a:a+1,:],
            {k: val[a:a+1] for k, val in labels.items()},
            models
            ), 
        range(settings.attacks),
        ),
        total=settings.attacks,
        desc="attacks",
        )))

def success_rate(ranks, min_rank=1):
    return np.sum(ranks <= min_rank) / ranks.size

if __name__ == "__main__":
    settings = parse_args()
    print("Start SNR estimation")
    snr = compute_snr(settings)
    print("Start modeling")
    models = compute_templates(settings, snr)
    print("Start attack")
    ranks = run_attacks_eval(settings, models)
    print('Attack ranks', collections.Counter(ranks))
    print(f'Success rate (rank 1): {success_rate(ranks, min_rank=1)*100:.0f}%')
    print(f'Success rate (rank 2**32): {success_rate(ranks, min_rank=2**32)*100:.0f}%')

