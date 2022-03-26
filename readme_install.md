- Required packages

```shell
conda install -c conda-forge jupyter
conda install -c conda-forge ipywidgets
python -m pip install numpy scipy matplotlib
jupyter nbextension enable --py widgetsnbextension
# `python -m pip install --upgrade jupyter_client`
python -m pip install -U scikit-imag>=0.19.1
python -m pip install webcolors opencv-python
python -m pip install jupyterlab
```

- ToC for notebook

    - install: https://github.com/ipython-contrib/jupyter_contrib_nbextensions

        ```shell
        python -m pip install jupyter_contrib_nbextensions
        # conda install -c conda-forge jupyter_contrib_nbextensions
        jupyter contrib nbextension install --user
        ```

    - activate **Table of Contents (2)*
