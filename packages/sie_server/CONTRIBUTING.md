# Contributing to SIE Server

## Development Setup

```bash
# Clone the repo
git clone https://github.com/superlinked/sie.git
cd sie

# Install mise and dependencies
mise trust && mise install
```

## Running Tests

```bash
mise run test packages/sie_server
```

## Code Style

- Format with `mise run lint -f`
- Type check with `mise run typecheck`

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make changes with tests
4. Submit a PR

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
