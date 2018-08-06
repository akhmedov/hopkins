ABSTRACT
======

The project is a part of R&D program that is aimed to investigate the 
applicability of RNN as a hardware unit in impulse radio receivers. It uses 
the timeseries that is a simulation of impulse propagation and receiving as 
learning-validation dataset to study the applicability. The time series can 
be generated by another open source project "Maxwell" that takes into account
all electrodynamic aspects of radiation, propagation, reflection, and adsorbing
of impulse signals. It also allows to simulate a noise of signal of different 
types with arbitrary parameters.

ADVANTAGES OF THE APPROACH
------

It is supposed that this technology allows to open new applications of impulse 
radio due to better recognition of EMP sparks then is provided by existing 
hardware. Also, we are going to investigate symbiotic usage of RNNs and existing
receiving electronics.

Generally, it is supposed that impulse signal classification by RNN may allow:
1. solving of the multipath problem in impulse radio;
2. solving of the multiuser problem in impulse radio;
3. obtaining the AWGN channel capacity gain in impulse radio;
4. obtaining the accuracy gain in radar problem;
5. cost optimization for receiving hardware units in impulse radio;
6. ability to update receiving hardware units in impulse radio for free;
7. building of high-radix communication protocol of channel level.

It is suggested that following advantages is opening new opportunists like:
1. fast and secure and low energy impulsed near field communication (NFC);
2. more accuracy and less cost of short range radar;
3. combined short range radar and remote sensing system;
4. new generation of wireless USB devises with high-radix communication protocol.

DATASET DESCRIPTION
------

The dataset is a time line of a received signal magnitudes. The signal is a 
sequential arrange of EMPs. Multipath and multiuser network scheme can be 
simulated by combination of datasets. Every point of the dataset contains not 
only the magnitude but also an information about location of point of 
observation and shape of impulse excitation. The dataset format is JSON file 
in ASCII encoding.

THEORETICAL JUSTIFICATION
------

Impulse shape depends from direction of the observation. So, the matching of 
the initial form of time dependency of excitation to the shape of EMP is a 
classification problem. Transient respond (TR) of the signal and a Duhamel's 
integral allow to build learning dataset for an arbitrary shape of the 
excitation for solving the classification problem by a supervised machine 
learning.

ENVIRONMENT INSTALATION GUIDE FOR UBUNTU 16.04 x64
======

```bash
~ $ 
```

USER INSTRUCTIONS
======

```bash
~ $ 
```
