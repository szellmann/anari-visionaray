#pragma once

#include "renderer/common.h"

namespace visionaray {

struct VisionarayRendererDirectLight
{
  VSNRAY_FUNC
  PixelSample renderSample(Ray ray, PRD &prd, unsigned worldID,
        VisionarayGlobalState::DeviceObjectRegistry onDevice,
        VisionarayGlobalState::ObjectCounts objCounts) {

    auto debug = [=]() {
      return prd.x == prd.frameSize.x/2 && prd.y == prd.frameSize.y/2;
    };

    if (debug()) printf("Rendering frame ====\n");

    PixelSample result;
    result.color = rendererState.bgColor;
    result.depth = 1.f;

    if (onDevice.TLSs[worldID].num_primitives() == 0)
      return result; // happens eg with TLSs of unsupported objects

    // Need at least one light..
    if (objCounts.lights == 0)
      return result;

    float3 throughput{1.f};

    auto hr = intersect(ray, onDevice.TLSs[worldID]);

    if (hr.hit) {

      auto inst = onDevice.instances[hr.inst_id];
      const auto &geom = onDevice.groups[inst.groupID].geoms[hr.geom_id];

      // TODO: currently, this will arbitrarily pick a volume _or_
      // surface BVH if both are present and do overlap
      if (geom.type != dco::Geometry::Volume) {

        vec3f gn(1.f,0.f,0.f);
        // TODO: doesn't work for instances yet
        if (geom.type == dco::Geometry::Triangle) {
          auto tri = geom.asTriangle.data[hr.prim_id];
          gn = normalize(cross(tri.e1,tri.e2));
        } else if (geom.type == dco::Geometry::Sphere) {
          auto sph = geom.asSphere.data[hr.prim_id];
          vec3f hitPos = ray.ori + hr.t * ray.dir;
          gn = normalize((hitPos-sph.center) / sph.radius);
        } else if (geom.type == dco::Geometry::Cylinder) {
          auto cyl = geom.asCylinder.data[hr.prim_id];
          vec3f hitPos = ray.ori + hr.t * ray.dir;
          vec3f axis = normalize(cyl.v2-cyl.v1);
          if (length(hitPos-cyl.v1) < cyl.radius)
            gn = -axis;
          else if (length(hitPos-cyl.v2) < cyl.radius)
            gn = axis;
          else {
            float t = dot(hitPos-cyl.v1, axis);
            vec3f pt = cyl.v1 + t * axis;
            gn = normalize(hitPos-pt);
          }
        }

        int lightID = uniformSampleOneLight(prd.random, objCounts.lights);
        assert(onDevice.lights[lightID].type == dco::Light::Point);
        auto pl = onDevice.lights[lightID].asPoint;

        vec3f hitPos = ray.ori + ray.dir * hr.t;
        auto ls = pl.sample(hitPos+1e-4f, prd.random);

        shade_record<float> sr;
        sr.normal = gn;
        sr.geometric_normal = gn;
        sr.view_dir = -ray.dir;
        sr.tex_color = float3(1.f);
        sr.light_dir = ls.dir;
        sr.light_intensity = pl.intensity(hitPos);

        // That doesn't work for instances..
        const auto &mat = onDevice.materials[hr.geom_id];
        float3 shadedColor = to_rgb(mat.asMatte.data.shade(sr));

        Ray shadowRay;
        shadowRay.ori = hitPos;
        shadowRay.dir = ls.dir;
        shadowRay.tmin = 1e-4f;
        shadowRay.tmax = ls.dist-1e-4f;
        auto shadowHR = intersect(shadowRay, onDevice.TLSs[worldID]);
        if (shadowHR.hit)
          throughput = float3{0.f};
        else
          throughput *= shadedColor;

        result.color = float4(throughput,1.f);
      }
    }

    if (prd.x == prd.frameSize.x/2 || prd.y == prd.frameSize.y/2) {
      result.color = float4(1.f) - result.color;
    }

    return result;
  }

  RendererState rendererState;
};

} // namespace visionaray
