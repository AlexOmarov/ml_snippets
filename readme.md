Export conda env 
```bash
conda env export --no-builds > environment.yaml
```

Import conda env
```bash
conda env update --file environment.yaml
```