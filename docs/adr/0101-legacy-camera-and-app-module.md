# ADR-0101: `vision/camera/` and `vision/app.py` are legacy

- Date: 2026-11-09
- Status: Accepted

## Context

`vision/camera/` is the pre-0.10 copy of the guided camera (merged into `vision/app/` in 0.10, ADR-0077);
its `quality.py` has diverged by ~500 lines. `vision/app.py` (410 lines) is shadowed by the `vision/app/`
package and cannot be imported.

## Decision

Both are excluded from the parameter registry and from all PHOENIX stages. They are not moved or deleted
yet: the device bridge used in this session cannot delete files. Move them to `archive/` in the first
commit on the owner's machine (`git mv vision/camera archive/camera; git mv vision/app.py archive/app_v09.py`),
then run the test suite.

## Rollback

`git mv` back.
