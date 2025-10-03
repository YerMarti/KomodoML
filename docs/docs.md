# KomodoML documentation generation

This documentation was generated using **pdoc** with a custom template from [nxtlo](https://github.com/nxtlo/syn).

## How to generate it

1. Install `pdoc` and the template.

```
pip install pdoc
```

2. Generated docs in `/docs` using the template contained in `/template`.

```
pdoc komodoml -t template -o docs
```

3. Host on GitHub Pages.