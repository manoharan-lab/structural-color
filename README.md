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

[1] Magkiriadou, S., Park, J.-G., Kim, Y.-S., and Manoharan, V. N. “Absence of
Red Structural Color in Photonic Glasses, Bird Feathers, and Certain Beetles”
*Physical Review E* 90, no. 6 (2014): 62302. doi:10.1103/PhysRevE.90.062302

[2] Magkiriadou, S. “Structural Color from Colloidal Glasses” (PhD Thesis,
Harvard University, 2014): Available at
http://dash.harvard.edu/bitstream/handle/1/14226099/MAGKIRIADOU-DISSERTATION-2015.pdf?sequence=1

