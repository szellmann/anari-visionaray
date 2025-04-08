anari-visionaray
================

ANARI device over the ray tracing library [visionaray](https://github.com/szellmann/visionaray)

Disclaimer (read this first)
----------------------------

This is an experimental implementation of the [ANARI
Spec](https://registry.khronos.org/ANARI/specs/1.0/ANARI-1.0.html). At the time
of writing this (March 2025) I do spend a fair amount of time keeping this
project up-to-date, so chances are it builds and works for you on Linux, Mac,
or Windows with the typical ANARI apps out there.

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

`anari-visionaray` compiles into separate devices for CPU, CUDA, and an
experimental but seldom tested HIP version. The latter two must be enabled with
CMake for them to compile.

List of Proprietary ANARI Extensions
------------------------------------

`anari-visionaray` adds the following extensions not specified by ANARI:

- `VSNRAY_CAMERA_MATRIX`: adds a camera type (subtype: "matrix") that takes two
  `ANARI_FLOAT32_MAT4` parameters "view" and "proj" that corresponds to
  OpenGL's `MODELVIEW` and `PROJECTION` matrices. This is intended to be used
  for interoperability with OpenGL.
- `VSNRAY_SAMPLER_VOLUME`: adds a sampler type that has an `ANARI_VOLUME`
  parameter named "volume" that is used to compose the input tuple.
- `VSNRAY_RENDERER_CLIP_PLANE`: allows to set an array of clip planes evaluated
  by the ray tracing renderers. The list is comprised of `float4` tuples with
  the `xyz` coordinates the normal vector and `w` an offset to the clip plane.
