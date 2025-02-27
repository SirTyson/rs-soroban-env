all: build test

test:
	cargo hack --each-feature test

build:
	cargo hack --each-feature clippy
	cargo hack clippy --target wasm32-unknown-unknown

# We use "run" to run the soroban-env-host/src/bin/main.rs
# entrypoint, which both excludes dev-deps (noisy) and
# actually includes soroban-env-host itself (rather than
# just its deps). We want to catch ourselves using APIs
# too!
check-apis:
	cargo acl run

watch:
	cargo watch --clear --watch-when-idle --shell '$(MAKE)'

fmt:
	cargo fmt --all

clean:
	cargo clean

regenerate-test-wasms:
	make -C soroban-test-wasms regenerate-test-wasms

publish:
	cargo workspaces publish --all --force '*' --from-git --yes

publish-dry-run:
	./publish-dry-run.sh

# Requires: `cargo install cargo-llvm-cov`
coverage:
	rm -f lcov.info
	cargo llvm-cov test --all-features --tests --lcov --output-path=lcov.info
