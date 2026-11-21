# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-11-21

### Added
- Added `BioVolume`, `BioVideo`, and `BioMultiChannel` image types, along with a format-aware `BioImageReader`.
- Added `ProcessDataset` workflows for dataset processing, patch generation and splitting, and YAML-configured data loading.
- Added visualization utilities for GIF creation, distribution plots, histograms, KDEs, and statistics tables.

### Changed
- Reworked image I/O and preprocessing, with improved support for image layouts, metadata, HDF5, and multichannel data.
- Expanded MONAI transform and fastai metric interoperability.
- Updated tutorials, examples, benchmarks, and generated documentation.

### Fixed
- Improved path resolution, image creation and loading, metadata handling, patch preprocessing, and layout normalization.

## [1.2.0] - 2026-04-07

### Added
- **Enhanced documentation** across all core modules (callbacks, core, data, datasets, io, losses, metrics, nets, transforms, visualize)
- **New tutorials** demonstrating common biomedical imaging workflows
- **Benchmark examples** showing performance comparisons for image restoration and classification tasks
- **Improved data handling** with enhanced dataframe loading functionality

### Changed
- **Refactored `bioDataLoader` class** for better flexibility and cleaner API
- **Reorganized project structure** with improved directory layout for examples and benchmarks
- **Updated tutorials** with additional use cases and clearer examples
- **Improved code organization** for better readability and maintainability
- **Enhanced README** with clearer installation instructions and usage examples
- **Updated base directory handling** for better cross-platform compatibility

### Fixed
- **Data pipeline robustness** improvements for diverse biomedical imaging formats
- **Test coverage improvements** across all modules
- **Documentation consistency** updates throughout the codebase

### Dependencies
- Requires Python 3.7+
- Built on top of MONAI, fastai, and TorchIO frameworks

## [1.1.0] - Previous Release

For information about previous versions, please refer to the git history.

---

## How to Contribute

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for more details on how to get started.

## License

bioMONAI is licensed under the [Apache License 2.0](LICENSE).
