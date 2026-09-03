# Probe images

`probe.jpg` is the photograph the demo run searches for.

A public figure was chosen deliberately. The pipeline can only find what a
search engine has already indexed, so a private individual's photograph would
usually return nothing — a correct result, but one that demonstrates nothing.
Everything the run discovers about this subject was already public.

| File | Subject | Source | Licence | Author |
| --- | --- | --- | --- | --- |
| `probe.jpg` | Sundar Pichai | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sundar_Pichai_-_2023_(cropped).jpg) | CC BY 4.0 | Lukasz Kobus (European Commission) |

This file is committed so that Google Lens can fetch it by URL. Lens accepts an
image URL rather than an upload, so the probe has to be reachable on the public
web before it can be searched; serving it from this repository avoids uploading
anything to a third-party host.

To search for a different face, pass any image with `--image`. A file outside
this repository is uploaded to catbox.moe instead, which publishes it — use your
own photograph, or one whose subject has agreed to it.
