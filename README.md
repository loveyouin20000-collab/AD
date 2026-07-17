# AD

Anomaly detection research repository.

## Release

Push a version tag to trigger an automatic GitHub Release with a generated changelog:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Commit messages following [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.) are grouped in the release notes. Pre-release tags containing `-` (for example `v0.2.0-beta.1`) are marked as pre-releases.
