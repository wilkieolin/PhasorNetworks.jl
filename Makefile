# Conveniences for the arc_experiments workflow (its CLAUDE.md rule 8). Stdlib make; Julia on PATH.
JULIA ?= julia
HOOKS := $(shell git rev-parse --git-common-dir)/hooks

.PHONY: help hooks envs env-cuda env-oneapi test test-gpu
help:
	@sed -n 's/^## //p' Makefile

## hooks       install scripts/hooks/* into the shared git hooks dir (covers every worktree); pre-push needs PHASOR_PUSH_OK=1
hooks:
	@mkdir -p $(HOOKS) && for h in scripts/hooks/*; do cp "$$h" "$(HOOKS)/$$(basename $$h)" && chmod +x "$(HOOKS)/$$(basename $$h)"; done && ls -l $(HOOKS)/pre-push

## envs        instantiate the pinned GPU environments in THIS checkout or worktree (envs/cuda, envs/oneapi); each may fail on a host without that backend
envs: env-cuda env-oneapi
env-cuda:
	-$(JULIA) --project=envs/cuda -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
env-oneapi:
	-$(JULIA) --project=envs/oneapi -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

## test        CPU suite (Pkg.test);  test-gpu  GPU suite on this host's backend (test/gpu.jl [cuda|oneapi])
test:
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'
test-gpu:
	$(JULIA) test/gpu.jl $(BACKEND)
