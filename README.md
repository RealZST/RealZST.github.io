# RealZST.github.io

Personal homepage of **Sitong Zhang** — [realzst.github.io](https://realzst.github.io).

Built with Jekyll + `jekyll-scholar`. Deployed via GitHub Actions.

## Local development

```bash
bundle install
bundle exec jekyll serve
```

Site available at `http://localhost:4000`.

To test on another device on the same LAN:

```bash
bundle exec jekyll serve --host 0.0.0.0 --port 4000
```

Then visit `http://<your-LAN-IP>:4000` from the other device.

## Structure

- `_pages/about.md` — homepage content (hero bio, bulletin, research paragraph)
- `_layouts/home.liquid` — homepage layout
- `_includes/research-map.liquid` — interactive research-stack diagram
- `_bibliography/papers.bib` — publication list (uses `jekyll-scholar`)
- `_sass/` — custom theme (tokens, base, per-section styles)
- `.github/workflows/deploy.yml` — CI: builds Jekyll and publishes to GitHub Pages

Publications / Projects / Notes pages exist in `_pages/` but are set to
`published: false` for v1 — re-enable by removing that front-matter line.

## Credits

Originally scaffolded from [al-folio](https://github.com/alshedivat/al-folio)
and then rewritten top-to-bottom with a custom theme.
