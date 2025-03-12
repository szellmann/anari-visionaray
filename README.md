anari-visionaray
================

ANARI device over the ray tracing library [visionaray](https://github.com/szellmann/visionaray)

Disclaimer (read this first)
----------------------------

This is an experimental implementation of the [ANARI
Spec](https://github.com/KhronosGroup/ANARI-Docs). At the time of writing this
(March 2025) I do spent a fair amount of time keeping this project up-to-date,
so chances are it builds and works for you on Linux, Mac, or Windows with the
typical ANARI apps out there.

That said, this project is mostly meant as a playground for me, so its future
is totally unclear at this point. For any "serious" ANARI users I would rather
refer to implementations like [Barney](https://github.com/ingowald/barney) or
[VisRTX](https://github.com/NVIDIA/VisRTX).

This device might be of interest if you're curious experimenting with some
features the other implementations don't have, like "omnidirectional camera",
"matrix camera" (compatible with OpenGL projection), "clipPlanes" on the
renderer, etc. etc. Though I'd also rather these features found their way into
the above devices, or, eventually into the spec if they're of general use.

Building
--------

If you decided to bear with me so far and _still_ want to use this project, I
refer you to the [github workflow
file](/.github/workflows/anari-visionaray-ci.yml) as that is most easy to
follow. You need to install
[visionaray](https://github.com/szellmann/visionaray) as a 3rd party library
(and the [ANARI-SDK](https://github.com/KhronosGroup/ANARI-SDK) of course).
Visionaray installs as header-only if you deactivate all the features as in the
CI script (you don't even need CUDA).

`anari-visionaray` into compiles separate devices for CPU, CUDA, and an
experimental but seldom tested HIP version. The latter two must be enabled with
CMake for them to compile.

