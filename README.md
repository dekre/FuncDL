# Development

## Environment Setup

### Install Pre-commit and Commitizen

```
brew install pre-commit
brew install commitizen
```

Make sure they are installed correctly by running the following commands.

```
cz version
pre-commit --version
```

Then in the root directory run the following commands.

```
pre-commit install
pre-commit install --hook-type commit-msg
```
