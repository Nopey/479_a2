# quick n dirty makefile to build rust binary
# based off the output of `cargo build --release --verbose` and `cargo build --verbose`

SHELL := /bin/bash

ROOT := $(shell realpath .)
COMMON_FLAGS = main.rs --edition=2018 --crate-type bin
RUSTC = rustc $(COMMON_FLAGS) --emit=dep-info,link

# release build
seasat:
	$(RUSTC) --crate-name $@ -C opt-level=3
-include seasat.d

# debug build
seasat_dbg:
	$(RUSTC) --crate-name $@ -C debuginfo=2
-include seasat_dbg.d

# Generate Documentation
# (Try opening doc/seasat/index.html in a browser)
.PHONY: doc
doc:
	rustdoc $(COMMON_FLAGS) --crate-name seasat --document-private-items

# Benchmark
# (Prefer hyperfine if it is present, but fall back to time)
.PHONY: bench
ifeq (, $(shell which hyperfine))
bench: seasat
	time ./seasat A2-input1.txt
	time ./seasat A2-input2.txt
	time ./seasat A2-input3.txt
	time ./seasat A2-input4.txt
else
bench: seasat
	@# Hyperfine needs the command to be in quotes
	hyperfine './seasat A2-input1.txt'
	hyperfine './seasat A2-input2.txt'
	hyperfine './seasat A2-input3.txt'
	hyperfine './seasat A2-input4.txt'
endif

# build everything
all: seasat seasat_dbg doc

# professor-proofing the makefile by adding aliases
.PHONY: docs build ball build_dbg debug dbg benchmark
docs: doc
build: seasat
ball: seasat
build_dbg: seasat_dbg
debug: seasat_dbg
dbg: seasat_dbg
benchmark: bench


# Clean build dir
.PHONY: clean
clean:
	rm -rf seasat seasat.d seasat_dbg seasat_dbg.d doc seasat*.o
