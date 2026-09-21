# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
# The --jobs auto parameter has been added in the hope that it would enable parallelization.
# The --fail-on-warning parameter ensures that the warnings are not overlooked when building on CI.
SPHINXOPTS    ?= --jobs auto --fail-on-warning
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

all:
	# Cleaning helps in detecting changes to the Python files.
	$(MAKE) clean
	# $(MAKE) apidoc
	$(MAKE) linkcheck
	$(MAKE) html
	$(MAKE) latexpdf
	$(MAKE) epub

all-noplot:
	$(MAKE) clean
	$(MAKE) linkcheck-noplot
	$(MAKE) html-noplot
	$(MAKE) latexpdf-noplot
	$(MAKE) epub-noplot

# apidoc:
# 	sphinx-apidoc -o ./pttools ../pttools
#	sphinx-apidoc -o ./tests ../tests

clean:
	rm -rf $(BUILDDIR) ./auto_examples ./gen_modules ./sg_api_usage.rst ./sg_execution_times.rst

# The -noplot targets build without running the examples:
# https://sphinx-gallery.github.io/stable/configuration.html#building-without-executing-examples
# They use the make mode -M like the catch-all target below, so their output goes to $(BUILDDIR)/<builder>.
# The make mode option -M has to be the first argument, and the other options have to come after the directories.
epub-noplot:
	$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" -D plot_gallery=0 $(SPHINXOPTS) $(O)

html-noplot:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -D plot_gallery=0 $(SPHINXOPTS) $(O)

latex-noplot:
	$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" -D plot_gallery=0 $(SPHINXOPTS) $(O)

# Builds the LaTeX sources and then the PDF with latexmk.
# This is used by pttools/docs/lint.py to check both the Sphinx output and the LaTeX build for errors.
latexpdf-noplot:
	$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" -D plot_gallery=0 $(SPHINXOPTS) $(O)

linkcheck-noplot:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" -D plot_gallery=0 $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
