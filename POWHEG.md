# POWHEG ggHH_quartic

The `POWHEG-BOX-V2` submodule contains the reviewed FCChh changes. Its nested
`ggHH_quartic` submodule contains the process changes. Both histories are hosted
in this same GitHub repository, on the `powheg-core` and `powheg-gghh-quartic`
branches respectively. The submodule commit IDs pin exact versions; they are
independent of the HHH `main` history.

Initialise only the POWHEG components needed for this process:

```sh
git submodule update --init POWHEG-BOX-V2
git -C POWHEG-BOX-V2 submodule update --init ggHH_quartic
```

See `POWHEG-BOX-V2/ggHH_quartic/REVIEWED_BUILD.md` for the reviewed changes,
build command, validation results and grid-regeneration requirements. Build
products, integration grids, event samples and Condor campaign outputs are not
versioned. The card and submission scripts retain the local Oxford dependency
and run paths; configure these for a different installation.

When updating source, publish the process commit to `powheg-gghh-quartic`,
then publish the core commit containing its new gitlink to `powheg-core`, then
update and push the outer gitlink on HHH `main`. Preserve the upstream GitLab
remotes for comparing with the original POWHEG repositories.
