# Attacking ASCAD with a Single Trace

This repository contains the scripts associated to the paper [Give Me 5 Minutes:
Attacking ASCAD with a Single Side-Channel
Trace](https://eprint.iacr.org/2021/817).

## Installation

This repository contains simple python scripts, with known-good
dependencies specified in `requirements.txt` (for python 3.10).
A recent version of pip is needed if you want to install the pre-build version
of SCALib, otherwise pip will try to recompile SCALib from scratch (which will
fail if you don't have the build dependencies installed).

We suggest using a virtual environment:
```
git clone https://github.com/cassiersg/ASCAD-5minutes.git
cd ASCAD-5minutes
python3 -m venv ve

# On linux
source ve/bin/activate
# On windows
RUN ve/Scripts/activate

pip install -U pip # to get recent pip
pip install -r requirements.txt
```

If the provided `requirements.txt` doesn't work, the direct dependencies are `matplotlib h5py scalib tqdm`.

The ASCAD database file used  for the attack can be found at https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190730-071646/atmega8515-raw-traces.h5 .


## Usage

### attack.py

This is a simple attack script which minimally reproduces our attack and should
be fairly esay to read or modify.
To run with the default settings:
```
python3 attack.py --database /path/to/ASCAD/raw/traces/atmega8515-raw-traces.h5
```

For other attack settings:
```
python3 attack.py --help
```

### attack_multi.py

This script performs the same attacks as <attack.py>, but stores intermediate
results, run multiple attacks in parallel and produces success rate plots.

Commands used to generate the success rate figures in the paper:

Full trace and full key:
```
python3 attack_multi.py --database ./atmega8515-raw-traces.h5 --fast-snr --ntracesattack 1 --averageattack 1000 --poi 32,64,128,256,512,1024,2048 --dim 1,2,3,4,5,6,7,8,9,10,15,20  --show snr,model,attack,sr-map
```

Partial trace and one byte of key:
```
python3 attack_multi.py --database ./atmega8515-raw-traces.h5 --fast-snr --ntracesattack 1,2,4,8,16,32 --nbytes 1 --averageattack 1000  --ntracesprofile 20000 --window 80945,82345 --show snr,model,attack,sr-boxplot
```

For all attack settings:
```
python3 attack_multi.py --help
```

### subset_traces.py

A utility script to make subsets of the ASCAD database that are smaller, thus
faster to load, for faster experimentation.
```
python3 subset_traces.py --help
```

## License

MIT
