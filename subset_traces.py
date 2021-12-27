#! /usr/bin/env python3

import argparse

import h5py

parser = argparse.ArgumentParser(
        description=
        "Export a subsed of the raw ASCAD traces in a new file. "
        "Useful for fast iteration on small attacks. "
        "Exported file has a structure identical to the original file."
        )
parser.add_argument(
    "-n",
    "--ntraces",
    type=int,
    help=
    "Number of traces in the exported dataset "
    "(always takes the n first traces).",
    required=True,
)
parser.add_argument(
    "-w",
    "--window",
    type=str,
    default="0,250000",
    help="'start,end' window for taking only a part of the traces (default: whole trace).",
)
parser.add_argument(
    "input",
    type=str,
    help="Location of the 'raw traces' ASCAD database file.",
)
parser.add_argument(
    "output",
    type=str,
    help="Location of the resulting dataset file.",
)
args = parser.parse_args()

W_START, W_END = [int(x.strip()) for x in args.window.split(",")]

f_database = h5py.File(args.input, "r")

l = args.ntraces

traces = f_database["traces"][:l,W_START:W_END]
masks = f_database["metadata"]["masks"][:l,:]
key = f_database["metadata"]["key"][:l,:]
plaintext = f_database["metadata"]["plaintext"][:l,]

with h5py.File(args.output, "w") as f:
    f.create_dataset("traces", data=traces)
    md = f.create_group("metadata")
    md.create_dataset("masks", data=masks)
    md.create_dataset("key", data=key)
    md.create_dataset("plaintext", data=plaintext)
