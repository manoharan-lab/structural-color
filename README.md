# structural-color
Python package for modeling structural color in colloidal systems. See the
tutorial Jupyter notebook (tutorial.ipynb) for instructions on using the
package.

Requires the [python-mie (pymie)](https://github.com/manoharan-lab/python-mie)
package for Mie scattering calculations. To install:

```shell
git clone https://github.com/manoharan-lab/python-mie.git
pip install ./python-mie
```

To remove:

```shell
pip remove pymie
```

You might want to first set up a virtual environment and install the pymie
package there.

The original code was developed by Sofia Magkiriadou (with contributions from
Jerome Fung and others) during her research [1,2] in the
[Manoharan Lab at Harvard University](http://manoharan.seas.harvard.edu). This
research was supported by the National Science Foundation under grant number
DMR-1420570 and by an International Collaboration Grant (Grant No.
Sunjin-2010-002) from the Korean Ministry of Trade, Industry & Energy of Korea.
The code has since been updated. It now works in Python 3, and it can handle
quantities with dimensions (using [pint](https://github.com/hgrecco/pint)).

[1] Magkiriadou, S.; Park, J.-G.; Kim, Y.-S.; and Manoharan, V. N. “Absence of
Red Structural Color in Photonic Glasses, Bird Feathers, and Certain Beetles”
*Physical Review E* 90, no. 6 (2014): 62302. doi:10.1103/PhysRevE.90.062302

[2] Magkiriadou, S. “Structural Color from Colloidal Glasses” (PhD Thesis,
Harvard University, 2014): Available at
http://dash.harvard.edu/bitstream/handle/1/14226099/MAGKIRIADOU-DISSERTATION-2015.pdf?sequence=1

Additional publications based on this code:

Stephenson, A. B.; Xiao, M.; Hwang, V.; Qu, L.; Odorisio, P. A.; Burke, M.; Task, K.; Deisenroth, T.; Barkley, S.; Darji, R. H.; Manoharan, V. N. “Predicting the Structural Colors of Films of Disordered Photonic Balls.” *ACS Photonics* 10, no. 1 (2023): 58-70. doi:10.1021/acsphotonics.2c00892.

Xiao, M.; Stephenson, A. B.; Neophytou, A.; Hwang, V.; Chakrabarti, D.; Manoharan, V. N. “Investigating the Trade-off between Color Saturation and Angle-Independence in Photonic Glasses.” *Optics Express* 29, no. 14 (2021): 21212–21224. doi:10.1364/OE.425399.

Hwang, V.; Stephenson, A. B.; Barkley, S.; Brandt, S.; Xiao, M.; Aizenberg, J.; Manoharan, V. N. “Designing Angle-Independent Structural Colors Using Monte Carlo Simulations of Multiple Scattering.” *Proceedings of National Academy  Sciences* 118, no. 4 (2021): e2015551118. doi:10.1073/pnas.2015551118.

Hwang, V.*; Stephenson, A. B.*; Magkiriadou, S.; Park, J.-G.; Manoharan, V. N. “Effects of Multiple Scattering on Angle-Independent Structural Color in Disordered Colloidal Materials.” *Physical Review E* 101, no. 1 (2020): 012614. doi:10.1103/PhysRevE.101.012614. *equal contribution

