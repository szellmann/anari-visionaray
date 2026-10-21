// Copyright 2023-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// visionaray
#include "visionaray/matrix_camera.h"
#include "visionaray/thin_lens_camera.h"
// ours
#include "dco/common.h"

namespace visionaray::dco {

// Camera //

struct Camera
{
  enum Type { Matrix, Pinhole, Omni, Ortho, Unknown, };
  Type type;
  unsigned camID;
  box1 shutter;
  float rollingShutterDuration;
  enum RollingShutterDirection { Down, Up, Right, Left, None, } rollingShutterDirection;
  union {
    matrix_camera asMatrixCam;

    struct {
      void init(float3 pos, float3 dir, float3 up, float fovy, float aspect,
                box2f image_region, float lensRadius, float focalDistance)
      {
        this->pos  = pos;
        this->dir  = dir;
        this->up   = up;
        this->fovy = fovy;
        this->aspect = aspect;
        this->image_region = image_region;
        this->lensRadius = lensRadius;
        this->focalDistance = focalDistance;

        float3 f = normalize(-dir);
        float3 s = normalize(cross(up,f));
        float3 u =           cross(f,s);

        U = s * tanf(fovy*0.5f) * aspect;
        V = u * tanf(fovy*0.5f);
        W = -f;

        view = mat4(
            s.x, u.x, f.x, 0.0f,
            s.y, u.y, f.y, 0.0f,
            s.z, u.z, f.z, 0.0f,
            -dot(pos, s), -dot(pos, u), -dot(pos, f), 1.0f
            );

        float near = 0.001f, far = 1000.f;

        float fd = cot(fovy * 0.5f);

        proj(0, 0) = fd / aspect;
        proj(0, 1) = 0.0f;
        proj(0, 2) = 0.0f;
        proj(0, 3) = 0.0f;

        proj(1, 0) = 0.0f;
        proj(1, 1) = fd;
        proj(1, 2) = 0.0f;
        proj(1, 3) = 0.0f;

        proj(2, 0) = 0.0f;
        proj(2, 1) = 0.0f;
        proj(2, 2) = (far + near) / (near - far);
        proj(2, 3) = (2.0f * far * near) / (near - far);

        proj(3, 0) = 0.0f;
        proj(3, 1) = 0.0f;
        proj(3, 2) = -1.0f;
        proj(3, 3) = 0.0f;
      }

      template <typename RNG>
      VSNRAY_FUNC
      inline Ray primary_ray(Ray/**/, RNG &rng, float x, float y, float width, float height) const
      {
        float2 screen((x+0.5f)/width, (y+0.5f)/height);
        screen = (float2(1.f) - screen) * image_region.min + screen * image_region.max;

        float u = 2.f * screen.x - 1.f;
        float v = 2.f * screen.y - 1.f;

        Ray r;
        r.ori = pos;
        r.dir = normalize(U*u+V*v+W);
        r.tmin = 0.f;
        r.tmax = FLT_MAX;

        float3 lens_du(normalize(U));
        float3 lens_dv(normalize(V));
        float3 lens_normal = cross(lens_du, lens_dv);

        float3 p = r.dir * (focalDistance / abs(dot(r.dir, lens_normal)));

        auto lens_sample = concentric_sample_disk(rng(), rng());

        float3 lens_offset = (lensRadius * lens_sample.x) * lens_du
                           + (lensRadius * lens_sample.y) * lens_dv;

        r.ori += lens_offset;
        r.dir = normalize(p - lens_offset);

        return r;
      }

      mat4 get_view_matrix() const
      {
        return view;
      }

      mat4 get_proj_matrix() const
      {
        return proj;
      }

      void init_matrices()
      {
      }
      float3 pos, dir, up;
      float fovy, aspect;
      float lensRadius, focalDistance;
      box2f image_region;
      float3 U, V, W;
      mat4 view, proj;
    } asPinholeCam;

    struct {
      void init(float3 pos, float3 dir, float3 up)
      {
        this->pos = pos;
        this->dir = dir;
        this->up  = up;

        float2 imgPlaneSize(1.f, 1.f);

        U = normalize(cross(dir, up)) * imgPlaneSize.x;
        V = normalize(cross(U, dir)) * imgPlaneSize.y;
        W = pos - 0.5f * U - 0.5f * V;
      }

      VSNRAY_FUNC
      inline Ray primary_ray(Ray/**/, float x, float y, float width, float height) const
      {
        float2 screen((x + 0.5f) / width, (y + 0.5f) / height);

        Ray ray;

        float theta = float(M_PI) * screen.y;
        float phi = float(2*M_PI) * screen.x;

        float3 localDir(sinf(theta) * cosf(phi),
                        cosf(theta),
                        sinf(theta) * sinf(phi));

        ray.ori = U * screen.x + V * screen.y + W;
        ray.dir = localDir * float3(1,-1,1);

        ray.tmin = 0.f;
        ray.tmax = FLT_MAX;
        return ray;
      }

      float3 dir,pos,up;
      float3 U, V, W;
    } asOmniCam;

    struct {
      void init(float3 pos, float3 dir, float3 up, float aspect, float height,
                box2f image_region)
      {
        this->pos = pos;
        this->dir = dir;
        this->up  = up;
        this->image_region = image_region;

        float2 imgPlaneSize(height * aspect, height);

        U = normalize(cross(dir, up)) * imgPlaneSize.x;
        V = normalize(cross(U, dir)) * imgPlaneSize.y;
        W = pos - 0.5f * U - 0.5f * V;
      }

      VSNRAY_FUNC
      inline Ray primary_ray(Ray/**/, float x, float y, float width, float height) const
      {
        float2 screen((x + 0.5f) / width, (y + 0.5f) / height);
        screen = (float2(1.0) - screen) * float2(image_region.min)
                               + screen * float2(image_region.max);

        Ray ray;
        ray.ori = U * screen.x + V * screen.y + W;
        ray.dir = dir;
        ray.tmin = 0.f;
        ray.tmax = FLT_MAX;
        return ray;
      }
      float3 dir,pos,up;
      float3 U, V, W;
      box2f image_region;
    } asOrthoCam;
  };

  template <typename RNG>
  VSNRAY_FUNC
  inline Ray primary_ray(RNG &rng, float x, float y, float width, float height) const
  {
    Ray ray;
    if (type == Pinhole)
      ray = asPinholeCam.primary_ray(Ray{}, rng, x, y, width, height);
    else if (type == Omni)
      ray = asOmniCam.primary_ray(Ray{}, x, y, width, height);
    else if (type == Ortho)
      ray = asOrthoCam.primary_ray(Ray{}, x, y, width, height);
    else if (type == Matrix)
      ray = asMatrixCam.primary_ray(Ray{}, x, y, width, height);

    // compute rolling shutter
    float2 screen((x + 0.5f) / width, (y + 0.5f) / height);

    float spatialFactor = 0.f;

    if (rollingShutterDirection == Left) spatialFactor = 1.f-screen.x;
    else if (rollingShutterDirection == Right) spatialFactor = screen.x;
    else if (rollingShutterDirection == Down) spatialFactor = screen.y;
    else if (rollingShutterDirection == Up) spatialFactor = 1.f-screen.y;

    float lineStartTime = shutter.min + (spatialFactor * rollingShutterDuration);

    ray.time = lineStartTime + (shutter.max - shutter.min) * rng();

    return ray;
  }
};

VSNRAY_FUNC
inline Camera createCamera()
{
  Camera cam;
  memset(&cam,0,sizeof(cam));
  cam.type = Camera::Unknown;
  cam.camID = UINT_MAX;
  cam.shutter = {0.5f, 0.5f};
  return cam;
}

} // namespace visionaray::dco
