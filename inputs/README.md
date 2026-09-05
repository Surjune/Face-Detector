# Probe images

`probe.jpg` is the photograph the pipeline searches for.

**To search for someone else, replace this one file.** Nothing else needs
changing — no configuration, no arguments, no code. The pipeline has no idea
who it is looking for until it looks: a name is guessed from the search results
only to drive the per-platform queries, and identity is not reported until the
face check has accepted or rejected every candidate.

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

## The committed images

| File | Subject | Run it with |
| --- | --- | --- |
| `probe.jpg` | Suresh Raina, cricketer | `run` — this is the default |
| `probe1.jpg` | Sundar Pichai | `run --image inputs/probe1.jpg` |
| `probe2.jpg` | a private individual | `run --image inputs/probe2.jpg` |

Only `probe1.jpg` carries a documented free licence: [Wikimedia
Commons](https://commons.wikimedia.org/wiki/File:Sundar_Pichai_-_2023_(cropped).jpg),
CC BY 4.0, by Lukasz Kobus (European Commission). The other two were supplied
for testing and come with no licence grant, so treat them as examples to
replace rather than assets to reuse.

## Why both a public figure and a private individual

The two kinds of subject demonstrate opposite halves of the same claim.

**A public figure succeeds.** Their photographs are indexed, so the reverse
image search returns real posts, the face check confirms them, and a social
media post is anchored on chain. That is the happy path the brief asks for.

**A private individual returns nothing — and says so.** Their photographs are
usually not indexed at all, and the visually similar pictures a search engine
offers instead are of other people. Every candidate is rejected by the face
check, the run ends with `no_match_found`, and the identity verdict reads
`NOT IDENTIFIED`.

That second case is worth running deliberately. A pipeline that can only ever
report a match is indistinguishable from one that invents them, and the search
results are genuinely tempting: a run on `probe2.jpg` once scored a stranger at
0.6548 because both photographs showed a young man beside a motorcycle. Being
able to watch it refuse is what makes the successes mean anything.
