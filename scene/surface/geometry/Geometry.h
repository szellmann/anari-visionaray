// Copyright 2022 The Khronos Group
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "array/Array1D.h"
// std
#include <array>

namespace visionaray {

struct VisionarayGeometry
{
  enum Type { Triangle, Instance, };
  Type type;
  union {
    struct {
      basic_triangle<3,float> *data;
      size_t len;
    } asTriangle;
    struct {
      mat4 xfm;
      VisionarayGeometry *vgeom;
    } asInstance;
  };
};

struct Geometry : public Object
{
  Geometry(VisionarayGlobalState *s);
  ~Geometry() override;

  static Geometry *createInstance(
      std::string_view subtype, VisionarayGlobalState *s);

  VisionarayGeometry visionarayGeometry() const;

  void commit() override;
  void markCommitted() override;

  //virtual float4 getAttributeValue(
  //    const Attribute &attr, const Ray &ray) const;

 protected:

  VisionarayGeometry vgeom;

  std::array<helium::IntrusivePtr<Array1D>, 5> m_attributes;
};

} // namespace visionaray

VISIONARAY_ANARI_TYPEFOR_SPECIALIZATION(visionaray::Geometry *, ANARI_GEOMETRY);
