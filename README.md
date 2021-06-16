# Attacking ASCAD with a Single Trace

This repository contains the script associated to the paper [Give Me 5 Minutes: Attacking ASCAD with a Single Side-Channel Trace](https://eprint.iacr.org/2021/817).

## Installation

This is a simple python (>=3.7) script, with dependencies specified in `requirements.txt`.
A recent version of pip is needed if you don't want to recompile SCALib from scratch.

We suggest using a virtual environment:
```
git clone https://github.com/cassiersg/ASCAD-5minutes.git
cd ASCAD-5minutes
python3 -m venv ve

# On linux
source ve/bin/activate
# On windows
RUN ve/Scripts/activate

pip install -U pip
pip install -r requirements.txt
```

## Usage

For help:
```
python ascad_attack.py --help
```

Commands used to generate the figures in the paper:

Full trace and full key:
```
python ascad_attack.py --database ./atmega8515-raw-traces.h5 -n 1,2 -b 14 -s 1 -a 100 -p 5000 all --show
```

Partial trace and one byte of key:
```
python ascad_attack.py  --database ./atmega8515-raw-traces.h5 -n 1,2,4,8,16,32 -b 1 -w 80945,82345 -a 100 -p 100000 all --show
```

## License

MIT
