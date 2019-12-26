# Contributing Guide

![Python versions](https://img.shields.io/badge/python-3.6%7C3.7-blue)
![Code style](https://img.shields.io/badge/style-black-black)
![GitHub issues](https://img.shields.io/github/issues/eigenfoo/littlemcmc)
![GitHub pull requests](https://img.shields.io/github/issues-pr/eigenfoo/littlemcmc)

> This guide was derived from the [PyMC3 contributing
> guide](https://github.com/pymc-devs/pymc3/blob/master/CONTRIBUTING.md).

As a scientific community-driven software project, LittleMCMC welcomes
contributions from interested individuals or groups. These guidelines are
provided to give potential contributors information to make their contribution
compliant with the conventions of the LittleMCMC project, and maximize the
probability of such contributions to be merged as quickly and efficiently as
possible.

There are 4 main ways of contributing to the LittleMCMC project (in descending
order of difficulty or scope):

* Adding new or improved functionality to the existing codebase
* Fixing outstanding issues (bugs) with the existing codebase. They range from
  low-level software bugs to higher-level design problems.
* Contributing or improving the documentation (`docs`)
* Submitting issues related to bugs or desired enhancements

## Opening issues

We appreciate being notified of problems with the existing LittleMCMC code. We
prefer that issues be filed on the [GitHub issue
tracker](https://github.com/eigenfoo/littlemcmc/issues), rather than on social
media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues
or pull requests by using the GitHub search tool to look for key words in the
project issue tracker.

## Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are inclined
to do so to submit patches for new or existing issues via pull requests. This is
particularly the case for simple fixes, such as typos or tweaks to
documentation, which do not require a heavy investment of time and attention.

Contributors are also encouraged to contribute new code to enhance LittleMCMC's
functionality, also via pull requests. Please consult the [LittleMCMC
documentation](https://littlemcmc.readthedocs.io) to ensure that any new
contribution does not strongly overlap with existing functionality.

The preferred workflow for contributing to LittleMCMC is to fork the [GitHub
repository](https://github.com/eigenfoo/littlemcmc/), clone it to your local
machine, and develop on a feature branch.

### Steps

1. Fork the [project repository](https://github.com/eigenfoo/littlemcmc/) by
   clicking on the 'Fork' button near the top right of the main repository page.
   This creates a copy of the code under your GitHub user account.

2. Clone your fork of the LittleMCMC repo from your GitHub account to your local
   disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/littlemcmc.git
   $ cd littlemcmc
   $ git remote add upstream git@github.com:eigenfoo/littlemcmc.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never routinely work
   on the ``master`` branch of any repository.

4.  To set up a Python virtual environment for development, you may run:

   ```bash
   $ make venv
   ```

   Alternatively, you may create a conda environment for development by running:

   ```bash
   $ make conda
   ```

5. Develop the feature on your feature branch. Add changed files using ``git
   add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally. After committing, it is a good idea to sync
   with the base repository in case there have been any changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/master
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the LittleMCMC repo. Click the
   'Pull request' button to send your changes to the project's maintainers for
   review. This will notify the developers.

### Pull request checklist

We recommended that your contribution complies with the following guidelines
before you submit a pull request:

* If your pull request addresses an issue, please use the pull request title to
  describe the issue and mention the issue number in the pull request
  description. This will make sure a link back to the original issue is created.

* All public methods must have informative docstrings with sample usage when
  appropriate.

* Please prefix the title of incomplete contributions with `[WIP]` (to indicate
  a work in progress). WIPs may be useful to
  1. indicate you are working on something to avoid duplicated work,
  1. request broad review of functionality or API, or
  1. seek collaborators.

* Documentation and high-coverage tests are necessary for enhancements to be
  accepted.

* Run any of the pre-existing examples in ``docs/_static/notebooks`` that
  contain analyses that would be affected by your changes to ensure that nothing
  breaks. This is a useful opportunity to not only check your work for bugs that
  might not be revealed by unit test, but also to show how your contribution
  improves LittleMCMC for end users.

You can check for common programming errors or stylistic issues with the
following Make rule:

  ```bash
  $ make lint
  ```

You can also run the test suite with the following Make rule:

  ```bash
  $ make test
  ```

## Code of Conduct

LittleMCMC abides by the [Contributor Covenant Code of Conduct, version
1.4](https://github.com/eigenfoo/littlemcmc/blob/master/CODE_OF_CONDUCT.md).
