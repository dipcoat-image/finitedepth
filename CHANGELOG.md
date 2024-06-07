# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.5] - 2024-06-07

### Fixed

- Update `curvesimilarities` dependencies to v0.1.3.

## [2.0.4] - 2024-05-22

### Fixed

- `coatinglayer.sample_polyline()` now uses 1st-order B-spline. This allows sampling polylines with 2 or 3 vertices.

## [2.0.3] - 2024-05-22

### Changed

- Polyline sampling is now done by `coatinglayer.sample_polyline()`.

### Removed

- `coatinglayer.acm()` and `coatinglayer.owp()` are replaced by their counterparts in [CurveSimilarities](https://pypi.org/project/curvesimilarities/) package.
- `coatinglayer.equidistant_interpolate()` is replaced by `coatinglayer.sample_polyline()`.

## [2.0.2] - 2024-03-09

### Added

- `CoatingLayerBase.valid()` is added.

## [2.0.1] - 2024.03-06

### Added

- `finitedepth analyzers` command is added.

### Fixed

- The frame size of the video output is now determined by the analyzed frame instead of the source frame. This allows the output frame to be a different size from the source frame.

## [2.0.0] - 2024.03.05

Package published to PyPI.
