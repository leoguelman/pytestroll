# pytestroll

Profit Maximizing A/B Testing.

`pytestroll` implements the methods in the paper [Test & Roll: Profit-Maximizing A/B Tests](https://arxiv.org/abs/1811.00457) by Elea McDonnell Feit and Ron Berman.


## Installation

```bash
$ cd dist
$ tar xzf pytestroll-0.1.0.tar.gz
$ pip install pytestroll-0.1.0/   
``

## Usage

`pytestroll` computes the profit-maximizing test size for a 2-armed A/B test, along with other functionality required to reproduce the paper results and examples. 

```python
from pytestroll.pytestroll import NHST, TestRoll
import numpy as np
    
    
mu = 0.68
sigma = 0.03
N = 100000

# Profit-maximizing
tr = TestRoll(N = N, s = np.sqrt(mu*(1-mu)), mu = mu, sigma = sigma)
n_star = tr.tr_size_nn()
print("Test & Roll samples:", n_star, '\n')

# Compare to standard Null Hypothesis Significance Test paradigm

d = 0.68*0.02 # 2% lift 

nht = NHST(s=np.sqrt(mu*(1-mu)), d=d)
n_nht = nht.nht_size_nn()
print("NHST Samples:", n_nht, '\n')
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pytestroll` was created by Leo Guelman. It is licensed under the terms of the MIT license.

## Credits

`pytestroll` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
