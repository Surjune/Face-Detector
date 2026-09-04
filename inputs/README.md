# Probe images

`probe.jpg` is the photograph the pipeline searches for.

**To search for someone else, replace this one file.** Nothing else needs
changing — no configuration, no arguments, no code. The subject's identity is
derived from the search results at run time, so the pipeline has no idea who it
is looking for until it looks.

```bash
python -m src.pipeline run
```

Any common format works as long as exactly one face is clearly visible; the
largest face is used if there are several. If the file is not `probe.jpg`, or
lives elsewhere, pass `--image path/to/photo.jpg` instead.

## How the image reaches the search engine

Google Lens accepts an image *URL*, never an upload, so the probe has to be
publicly fetchable first. Two routes, chosen automatically:

- **Already published here.** If this repository serves byte-identical content
  at this file's raw URL, that URL is used and nothing is uploaded anywhere.
- **Otherwise, uploaded to catbox.moe**, an anonymous keyless host.

The published bytes are verified, not assumed. Replacing `probe.jpg` locally
without committing leaves the *previous* photograph published at the same URL,
and searching that would confidently return results for the wrong person while
every later stage appeared to work perfectly. A hash comparison rules that out.

So in practice: swap in your own image and it is uploaded to a public host for
the search. That is inherent to using a hosted reverse image search — use your
own photograph, or one whose subject has agreed to it.

## The committed image

| File | Subject | Source | Licence | Author |
| --- | --- | --- | --- | --- |
| `probe.jpg` | Sundar Pichai | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sundar_Pichai_-_2023_(cropped).jpg) | CC BY 4.0 | Lukasz Kobus (European Commission) |

A public figure was chosen deliberately. The pipeline can only find what a
search engine has already indexed, so a private individual's photograph would
usually return nothing — a correct result, but one that demonstrates nothing.
Everything the run discovers about this subject was already public.
