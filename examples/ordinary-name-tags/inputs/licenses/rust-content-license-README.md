# The Rust blog

[![CI](https://github.com/rust-lang/blog.rust-lang.org/actions/workflows/main.yml/badge.svg)](https://github.com/rust-lang/blog.rust-lang.org/actions/workflows/main.yml)

This is the blog of the Rust Programming Language.

It uses [Zola](https://www.getzola.org/) and is deployed to GitHub Pages via GitHub Actions.

## Installing Zola

You need at least zola v0.22.0 to render the blog.

Compile from source:

```sh
cargo install --debug --locked --git https://github.com/getzola/zola --rev 29540e9897dbe8aca388b13f7bdf615985f6ca2c
```

You can also find a list of package managers that provide zola [here](https://www.getzola.org/documentation/getting-started/installation/).
Prebuilt binaries are available for download [here](https://github.com/getzola/zola/releases/tag/v0.22.1).
If you use [mise](https://mise.jdx.dev), you can run the pinned version of zola directly with `mise run zola`.

## Building

To serve the site locally, run the following:

```sh
zola serve --fast --open
```

The site will be reloaded automatically when you make any changes.

## Contributing

First of all, thank you!

Like everything in Rust, the blog is licensed MIT/Apache 2.0. See the two
`LICENSE-*` files for more details. We're also governed by the Rust
Code of Conduct, see `CODE_OF_CONDUCT.md` for more.

### Writing a new blog post

There is an interactive blog post generator that takes care of some boilerplate for you.
To use it, run:

```
cargo blog
```

### Snapshot testing

If you're making changes to how the site is generated, you may want to check the impact your changes have on the output.
For this purpose, there is a setup to do snapshot testing over the entire output directory.

To run these tests in CI, add the string `RUN_SNAPSHOT_TESTS` to the PR description.
You can also run these tests locally for a faster feedback cycle:

- Make sure you have [cargo-insta](https://insta.rs/docs/quickstart/) installed.

- Generate the good snapshots to compare against, usually based off the main branch:
  ```sh
  cargo insta test -p snapshot --accept --include-ignored
  ```
  Consider making a commit with these snapshots, so you can always check the diff of your changes with git:
  ```sh
  git add --force crates/snapshot/src/snapshots # snapshots are ignored by default
  git commit --message "WIP add good snapshots"
  ```
  Since we can't merge the snapshots to main, don't forget to drop this commit when opening a pull request.

- Compare the output of the branch you're working on with the good snapshots:
  ```sh
  cargo insta test -p snapshot --review --include-ignored
  ```
