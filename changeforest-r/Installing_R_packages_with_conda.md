# Installing the changeforest R package with conda

Conda is an open source package management system. Conda enables maintaining and switching
between environments on your computer. It was created for Python programs, but can distribute
packages for any language (including R).

I personally manage my R packages and their dependencies with conda. This is a short manual
on installing the `changeforest` R package with conda. The `changeforest` R package is
available for Mac and Linux, not Windows as of yet.
[More](https://towardsdev.com/install-r-in-conda-8b2033ec3d4f)
[detailed](https://www.biostars.org/p/450316/)
[descriptions](https://docs.anaconda.com/anaconda/packages/r-language-pkg-docs/) on how to
manage R dependencies with conda exist online. There is also a
[cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
with commonly used commands.

If you do not have conda installed on your system, download an installer corresponding to
your OS and architecture [here](https://github.com/conda-forge/miniforge). I would recommend
`Miniforge3` or `Mambaforge`. Follow the installation instructions, making sure to run
`conda init bash` or `conda init zsh` at the end as described. Also, restart your shell
(e.g., by closing and reopening the terminal). Afterwards, your terminal should look something
like this:

```bash
(base) ~ $ 
```

The `(base)` indicates that you are in the `base` environment. I personally don't recommend
installing packages directly into the `base` environment. First create a new environment `R`
and activate the new environment:

```bash
(base) ~ $ conda create --name R
(base) ~ $ conda activate R
(R) ~ $
```

Next, install `r-essentials` and `r-changeforest`:

```bash
(R) ~ $ conda install -c conda-forge -y r-essentials r-changeforest
```

The `-c conda-forge` tells `conda` to install `r-changeforest` from the open-source channel
`conda-forge`. There, R packages are available with the `r-` prefix. E.g., to install `MASS`,
run

```bash
(R) ~ $ conda install -c conda-forge -y r-mass
```

Congratulations, you installed `R` and `changeforest` with conda. Now, you should be able
to import `changeforest` in an interactive R session:

```bash
(R) ~ $ R

R version 4.1.3 (2022-03-10) -- "One Push-Up"
...
> library(changeforest)
```

If you are using R Studio, this should pick up the `conda` R installation given that you
activated the correct environment first. You can check whether R Studio is using the correct
R installation by checking `R.home()` (or `library(changeforest)`). I personally do not use
R Studio, so do not have experience with this.
[Here](https://stackoverflow.com/questions/38534383/how-to-set-up-conda-installed-r-for-use-with-rstudio)
is a StackOverflow exchange with more information.

If any of the above does not work for you, please feel free to open an issue.