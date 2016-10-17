# Low-order reactor models

CSTR modeling approach to estimate fast pyrolysis yields from bubbling
fluidized bed reactors.

## Folders

The `data/` folder contains files related to experiments or data obtained from
literature. Particle size distribution data taken from Figure 2 in the
Carpenter 2014 paper is available in `05mm.csv` (0.5 mm sieve) and `2mm.csv` (2
mm sieve). Characteristics of the particles (surface area, volume) are provided
in `sizeinfo.txt`.

```
data/
|-- 05mm.csv
|-- 2mm.csv
|-- sizeinfo.txt
```

## Files

`cstr.py` is reactor model for one or more CSTR reactors in series at
steady-state conditions.

`cstr_dist.py` is CSTR model accounting for a distribution of particle sizes.

`cstr_rtd.py` is CSTR model and RTD model for different number of stages.

`cstr_static.py` is CSTR model accounting for a single particle size.

`cstr_taug.py` compares tar yields for different gas residence times.

