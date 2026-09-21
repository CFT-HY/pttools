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
	$(MAKE) latex-noplot
	$(MAKE) epub-noplot

# apidoc:
# 	sphinx-apidoc -o ./pttools ../pttools
#	sphinx-apidoc -o ./tests ../tests

clean:
	rm -rf $(BUILDDIR) ./auto_examples ./gen_modules ./sg_api_usage.rst ./sg_execution_times.rst

epub-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b epub "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-noplot:
	# https://sphinx-gallery.github.io/stable/configuration.html#building-without-executing-examples
	$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

latex-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

linkcheck-noplot:
	@$(SPHINXBUILD) -D plot_gallery=0 -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
