# Migration Boundary

`VerdiWM-clean-v0.1.0` is a new Kernel baseline. The previous development
trees remain outside this repository as legacy material and are intentionally
untouched. Ctrl-World-specific behavior must migrate as an external adapter or
domain pack that implements `ModelAdapter`; it must not be copied into
`verdi_core`.

Cleanup gate:

1. Review this repository and run the release preflight.
2. Upload or archive this commit as the agreed public baseline.
3. Run a real target adapter through the same loop and review evidence quality.
4. Only then decide which legacy directories can be archived or deleted.

Deletion is not part of the v0.1.0 build because the old trees may still be
needed for adapter extraction and audit provenance.
