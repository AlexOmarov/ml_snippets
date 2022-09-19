## Contributing

New branch with story slack (`ML-...`) as a name should be created for each new task.

Each contribution to master branch MUST include following steps:

1. Changelog update with number of each task written under the new version
2. New unit tests for each new functionality (or changed class)
3. Increment version of service (highly recommended to use [Semver standard](https://semver.org/))
4. Updates of README (if some serious changes have been made)
5. Updates of python comments on each changed public method (if changes are major)
6. Update environment.yaml files when new deps are added (either via conda export or manually)
```bash 
conda env export --no-builds > win_environment.yaml
```

