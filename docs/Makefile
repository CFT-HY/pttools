# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
# The -j auto parameter has been added in the hope that it would enable parallelization.
SPHINXOPTS    ?= -j auto
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
	$(MAKE) linkcheck
	$(MAKE) html
	$(MAKE) latexpdf
	$(MAKE) epub

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples/
	rm -rf gen_modules/
	rm -rf */gen_modules/
	rm -rf */*/gen_modules/

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
