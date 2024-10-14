#include <iostream>
#include <GL/glew.h>
#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>
#include <common/imgui/imgui.h>
#include "renderer/common.h"
#include "AnariCamera.h"

using box3_t = std::array<anari::math::float3, 2>;
namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(box3_t, ANARI_FLOAT32_BOX3);
ANARI_TYPEFOR_DEFINITION(box3_t);
} // namespace anari

using namespace visionaray;

static  float  g_groundPlaneOpacity = { 0.5f };
static const char* g_selectedMaterial = "Matte";
static  float3 g_lightDir = { 1.f, 1.f, 0.f };
static  float  g_metallic = { 0.f };
static  float  g_roughness = { 0.f };
static  float  g_clearcoat = { 0.f };
static  float  g_clearcoatRoughness = { 0.f };
static  float  g_ior = { 1.f };
static  bool   g_showGroundPlane = { true };
static  bool   g_showLightDir = { true };
static  bool   g_showAxes = { true };
static box3_t  g_bounds = { anari::math::float3{-3.f, 0.f, -3.f},
                            anari::math::float3{3.f, 1.f, 3.f} };

void statusFunc(const void *userData,
    ANARIDevice device,
    ANARIObject source,
    ANARIDataType sourceType,
    ANARIStatusSeverity severity,
    ANARIStatusCode code,
    const char *message)
{
  (void)userData;
  if (severity == ANARI_SEVERITY_FATAL_ERROR)
    fprintf(stderr, "[FATAL] %s\n", message);
  else if (severity == ANARI_SEVERITY_ERROR)
    fprintf(stderr, "[ERROR] %s\n", message);
  else if (severity == ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[WARN ] %s\n", message);
  else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING)
    fprintf(stderr, "[PERF ] %s\n", message);
  else if (severity == ANARI_SEVERITY_INFO)
    fprintf(stderr, "[INFO] %s\n", message);
}

static void anari_free(const void * /*user_data*/, const void *ptr)
{
  std::free(const_cast<void *>(ptr));
}

static anari::Array2D makeTextureData(anari::Device d, int dim)
{
  using texel = std::array<uint8_t, 3>;
  texel *data = (texel *)std::malloc(dim * dim * sizeof(texel));

  auto makeTexel = [](uint8_t v) -> texel { return {v, v, v}; };

  for (int h = 0; h < dim; h++) {
    for (int w = 0; w < dim; w++) {
      bool even = h & 1;
      if (even)
        data[h * dim + w] = w & 1 ? makeTexel(255) : makeTexel(0);
      else
        data[h * dim + w] = w & 1 ? makeTexel(0) : makeTexel(255);
    }
  }

  return anariNewArray2D(
      d, data, &anari_free, nullptr, ANARI_UFIXED8_VEC3, dim, dim);
}

static anari::Surface makePlane(anari::Device d, box3_t bounds)
{
  anari::math::float3 vertices[4];
  vertices[0] = { bounds[0][0], bounds[0][1], bounds[1][2] };
  vertices[1] = { bounds[1][0], bounds[0][1], bounds[1][2] };
  vertices[2] = { bounds[1][0], bounds[0][1], bounds[0][2] };
  vertices[3] = { bounds[0][0], bounds[0][1], bounds[0][2] };

  anari::math::float2 texcoords[4] = {
      {0.f, 0.f},
      {0.f, 1.f},
      {1.f, 1.f},
      {1.f, 0.f},
  };

  auto geom = anari::newObject<anari::Geometry>(d, "quad");
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.position",
      anari::newArray1D(d, vertices, 4));
  anari::setAndReleaseParameter(d,
      geom,
      "vertex.attribute0",
      anari::newArray1D(d, texcoords, 4));
  anari::commitParameters(d, geom);

  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);

  auto tex = anari::newObject<anari::Sampler>(d, "image2D");
  anari::setAndReleaseParameter(d, tex, "image", makeTextureData(d, 8));
  anari::setParameter(d, tex, "inAttribute", "attribute0");
  anari::setParameter(d, tex, "wrapMode1", "clampToEdge");
  anari::setParameter(d, tex, "wrapMode2", "clampToEdge");
  anari::setParameter(d, tex, "filter", "nearest");
  anari::commitParameters(d, tex);

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setAndReleaseParameter(d, mat, "color", tex);
  anari::setParameter(d, mat, "alphaMode", "blend");
  anari::setParameter(d, mat, "opacity", g_groundPlaneOpacity);
  anari::commitParameters(d, mat);
  anari::setAndReleaseParameter(d, surface, "material", mat);

  anari::commitParameters(d, surface);

  return surface;
}

static anari::Instance makePlaneInstance(anari::Device d, const box3_t &bounds)
{
  auto surface = makePlane(d, bounds);

  auto group = anari::newObject<anari::Group>(d);
  anari::setAndReleaseParameter(
      d, group, "surface", anari::newArray1D(d, &surface));
  anari::commitParameters(d, group);

  anari::release(d, surface);

  auto inst = anari::newObject<anari::Instance>(d, "transform");
  anari::setAndReleaseParameter(d, inst, "group", group);
  anari::commitParameters(d, inst);

  return inst;
}

static anari::Instance makeArrowInstance(anari::Device d,
                                         anari::math::float3 v1,
                                         anari::math::float3 v2,
                                         anari::math::float3 color)
{
  // Cylinder geometry:
  anari::math::float3 cylPositions[] = { v1, v2 };
  auto cylGeom = anari::newObject<anari::Geometry>(d, "cylinder");
  anari::setAndReleaseParameter(d,
      cylGeom,
      "vertex.position",
      anari::newArray1D(d, cylPositions, 2));
  anari::setParameter(d, cylGeom, "radius", 0.02f);
  anari::commitParameters(d, cylGeom);

  // Cone geometry:
  anari::math::float3 dir = v1 + v2;
  anari::math::float3 conePositions[] = { v2, v2+normalize(dir)/6.f };
  float coneRadii[] = { 0.05f, 0.0f };
  auto coneGeom = anari::newObject<anari::Geometry>(d, "cone");
  anari::setAndReleaseParameter(d,
      coneGeom,
      "vertex.position",
      anari::newArray1D(d, conePositions, 2));
  anari::setAndReleaseParameter(d,
      coneGeom,
      "vertex.radius",
      anari::newArray1D(d, coneRadii, 2));
  anari::commitParameters(d, coneGeom);

  // Surfaces and material:

  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", color);
  anari::commitParameters(d, mat);

  auto cylSurface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, cylSurface, "geometry", cylGeom);
  anari::setParameter(d, cylSurface, "material", mat);
  anari::commitParameters(d, cylSurface);

  auto coneSurface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, coneSurface, "geometry", coneGeom);
  anari::setParameter(d, coneSurface, "material", mat);
  anari::commitParameters(d, coneSurface);

  anari::release(d, mat);

  anari::Surface surface[2];
  surface[0] = cylSurface;
  surface[1] = coneSurface;

  auto group = anari::newObject<anari::Group>(d);
  anari::setAndReleaseParameter(
      d, group, "surface", anari::newArray1D(d, surface, 2));
  anari::commitParameters(d, group);

  anari::release(d, cylSurface);
  anari::release(d, coneSurface);

  auto inst = anari::newObject<anari::Instance>(d, "transform");
  anari::setAndReleaseParameter(d, inst, "group", group);
  anari::commitParameters(d, inst);

  return inst;
}

static anari::Geometry generateSphereMesh(anari::Device device, dco::Material mat)
{
  float3 viewDir{0.f,1.f,0.f};
  float3 lightDir = normalize(g_lightDir);
  float3 lightIntensity{1.f};
  float3 Ng{0.f,1.f,0.f}, Ns{0.f,1.f,0.f};
  int primID{0};
  dco::Sampler *samplers{nullptr};
  float4 *attribs{nullptr};

  int segments = 400;
  int vertexCount = (segments - 1) * segments;
  int indexCount = ((segments - 2) * segments) * 2;

  auto positionArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, vertexCount);
  auto *position = anari::map<anari::math::float3>(device, positionArray);

  auto indexArray =
      anari::newArray1D(device, ANARI_UINT32_VEC3, indexCount);
  auto *index = anari::map<anari::math::uint3>(device, indexArray);

  int cnt = 0;
  for (int i = 0; i < segments-1; ++i) {
    for (int j = 0; j < segments; ++j) {
      float phi = M_PI * (i+1) / float(segments);
      float theta = 2.f * M_PI * j / float(segments);

      anari::math::float3 v(
        sinf(phi) * cosf(theta),
        cosf(phi),
        sinf(phi) * sinf(theta));

      viewDir = normalize(vec3(v.x,v.y,v.z));
      float3 value = evalMaterial(mat,samplers,attribs,primID,Ng,Ns,
                                  normalize(viewDir),lightDir,lightIntensity);
      float scale = fabsf(value.y);
      position[cnt++] = v * scale;
    }
  }

  cnt = 0;
  for (int j = 0; j < segments-2; ++j) {
    for (int i = 0; i < segments; ++i) {
      int j0 = j * segments + 1;
      int j1 = (j+1) * segments + 1;
      unsigned idx0 = j0 + i;
      unsigned idx1 = j0 + (i+1) % segments;
      unsigned idx2 = j1 + (i+1) % segments;
      unsigned idx3 = j1 + i;
      index[cnt++] = anari::math::uint3(idx0,idx1,idx2);
      index[cnt++] = anari::math::uint3(idx0,idx2,idx3);
    }
  }

  anari::unmap(device, positionArray);
  anari::unmap(device, indexArray);

  //auto geometry = anari::newObject<anari::Geometry>(device, "quad");
  auto geometry = anari::newObject<anari::Geometry>(device, "triangle");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionArray);
  anari::setAndReleaseParameter(
      device, geometry, "primitive.index", indexArray);
  anari::commitParameters(device, geometry);

  return geometry;
}

static anari::Surface makeBRDFSurface(anari::Device device, dco::Material mat)
{
  auto geometry = generateSphereMesh(device, mat);
  anari::commitParameters(device, geometry);

  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::commitParameters(device, material);

  auto quadSurface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, quadSurface, "geometry", geometry);
  anari::setAndReleaseParameter(device, quadSurface, "material", material);
  anari::commitParameters(device, quadSurface);
  return quadSurface;
}

static void addPlaneAndArrows(anari::Device device, anari::World world)
{
  std::vector<anari::Instance> instances;

  // ground plane
  if (g_showGroundPlane) {
    auto planeInst = makePlaneInstance(device, g_bounds);
    instances.push_back(planeInst);
  }

  // light dir:
  if (g_showLightDir) {
    if (length(g_lightDir) > 0.f) {
      auto ld = normalize(g_lightDir);
      anari::math::float3 origin(0.f, 0.f, 0.f);
      anari::math::float3 lightDir(ld.x, ld.y, ld.z);
      auto lightDirInst = makeArrowInstance(device,
                                            origin,
                                            (lightDir-origin) * 1.2f,
                                            anari::math::float3(1.f, 1.f, 0.f));
      instances.push_back(lightDirInst);
    }
  }

  // basis vectors:
  if (g_showAxes) {
    auto xInst = makeArrowInstance(device,
                                   anari::math::float3(0.f, 0.f, 0.f),
                                   anari::math::float3(1.2f, 0.f, 0.f),
                                   anari::math::float3(1.f, 0.f, 0.f));
    auto yInst = makeArrowInstance(device,
                                   anari::math::float3(0.f, 0.f, 0.f),
                                   anari::math::float3(0.f, 1.2f, 0.f),
                                   anari::math::float3(0.f, 1.f, 0.f));
    auto zInst = makeArrowInstance(device,
                                   anari::math::float3(0.f, 0.f, 0.f),
                                   anari::math::float3(0.f, 0.f, 1.2f),
                                   anari::math::float3(0.f, 0.f, 1.f));
    instances.push_back(xInst);
    instances.push_back(yInst);
    instances.push_back(zInst);
  }

  if (!instances.empty()) {
    anari::setAndReleaseParameter(
        device, world, "instance",
        anari::newArray1D(device, instances.data(), instances.size()));
    for (auto &i : instances) {
      anari::release(device, i);
    }
  }

  anari::commitParameters(device, world);
}

dco::Material generateMaterial()
{
  dco::Material::Type type;
  if (std::string(g_selectedMaterial) == "Matte")
    type = dco::Material::Matte;
  else if (std::string(g_selectedMaterial) == "PBM")
    type = dco::Material::PhysicallyBased;

  dco::Material mat;
  mat.type = type;

  if (type == dco::Material::Matte) {
    mat = dco::makeDefaultMaterial();
    mat.asMatte.color.rgb = {1.f,1.f,1.f};
  }
  else if (type == dco::Material::PhysicallyBased) {
    mat.asPhysicallyBased.baseColor.rgb = {1.f,1.f,1.f};
    mat.asPhysicallyBased.baseColor.samplerID = UINT_MAX;
    mat.asPhysicallyBased.baseColor.attribute = dco::Attribute::None;

    mat.asPhysicallyBased.opacity.f = 1.f;
    mat.asPhysicallyBased.opacity.samplerID = UINT_MAX;
    mat.asPhysicallyBased.opacity.attribute = dco::Attribute::None;

    mat.asPhysicallyBased.metallic.f = g_metallic;
    mat.asPhysicallyBased.metallic.samplerID = UINT_MAX;
    mat.asPhysicallyBased.metallic.attribute = dco::Attribute::None;

    mat.asPhysicallyBased.roughness.f = g_roughness;
    mat.asPhysicallyBased.roughness.samplerID = UINT_MAX;
    mat.asPhysicallyBased.roughness.attribute = dco::Attribute::None;

    mat.asPhysicallyBased.normal.samplerID = UINT_MAX;

    mat.asPhysicallyBased.alphaMode = dco::AlphaMode::Opaque;
    mat.asPhysicallyBased.alphaCutoff = 0.f;

    mat.asPhysicallyBased.clearcoat.f = g_clearcoat;
    mat.asPhysicallyBased.clearcoat.samplerID = UINT_MAX;
    mat.asPhysicallyBased.clearcoat.attribute = dco::Attribute::None;

    mat.asPhysicallyBased.clearcoatRoughness.f = g_clearcoatRoughness;
    mat.asPhysicallyBased.clearcoatRoughness.samplerID = UINT_MAX;
    mat.asPhysicallyBased.clearcoatRoughness.attribute = dco::Attribute::None;

    mat.asPhysicallyBased.ior = g_ior;
  }
  return mat;
}

struct Renderer : viewer_glut
{
  Renderer();
  ~Renderer();

  void on_display() override;
  void on_mouse_move(const visionaray::mouse_event &event) override;
  void on_key_press(const visionaray::key_event &event) override;
  void on_resize(int w, int h) override;

  AnariCamera::SP cam;

  struct {
    anari::Library library{nullptr};
    anari::Device device{nullptr};
    anari::World world{nullptr};
    anari::Renderer renderer{nullptr};
    anari::Frame frame{nullptr};
    anari::Light light{nullptr};
  } anari;
};

Renderer::Renderer()
{
  anari.library = anari::loadLibrary("visionaray", statusFunc);
  anari.device = anariNewDevice(anari.library, "default");
  anari.renderer = anari::newObject<anari::Renderer>(anari.device, "default");
  //anari.renderer = anari::newObject<anari::Renderer>(anari.device, "raycast");

  anari.world = anari::newObject<anari::World>(anari.device);

  auto surf = makeBRDFSurface(anari.device, generateMaterial());
  //auto surf = makeCylinders(anari.device);
  //auto surf = makeCurves(anari.device);
  //auto surf = makeBezierCurves(anari.device);
  anari::setAndReleaseParameter(
      anari.device, anari.world, "surface", anari::newArray1D(anari.device, &surf));
  anari::commitParameters(anari.device, anari.world);

  addPlaneAndArrows(anari.device, anari.world);

  anari.light = anari::newObject<anari::Light>(anari.device, "directional");
  anari::setParameter(anari.device, anari.light, "direction",
      anari::math::float3(1.f, -1.f, -1.f));
  anari::setParameter(anari.device, anari.light, "irradiance", 1.f);
  anari::setParameter(anari.device, anari.light, "color",
      anari::math::float3(1.f, 1.f, 1.f));
  anari::setAndReleaseParameter(anari.device,
      anari.world,
      "light",
      anari::newArray1D(anari.device, &anari.light, 1));

  anari::commitParameters(anari.device, anari.world);

  cam = std::make_shared<AnariCamera>(anari.device);

  float aspect = width() / float(height());
  cam->perspective(60.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
  cam->set_viewport(0, 0, width(), height());

  cam->viewAll(g_bounds);
  cam->commit();

  anari::setParameter(anari.device, anari.renderer, "background",
      anari::math::float4(0.6f, 0.6f, 0.6f, 1.0f));

  anari::setParameter(anari.device, anari.renderer, "ambientRadiance", 0.f);
  //anari::setParameter(anari.device, anari.renderer, "mode", "Ng");
  //anari::setParameter(anari.device, anari.renderer, "mode", "Ns");
  //anari::setParameter(anari.device, anari.renderer, "heatMapEnabled", true);
  //anari::setParameter(anari.device, anari.renderer, "taa", true);
  //anari::setParameter(anari.device, anari.renderer, "taaAlpha", 0.1f);
  //anari::math::float4 clipPlane[] = {{ 0.f, 0.f, -1.f, 0.f }};
  //anari::math::float4 clipPlane[] = {{ 0.707f, 0.f, -0.707f, 0.4f }};
  //anari::setAndReleaseParameter(
  //    anari.device, anari.renderer, "clipPlane",
  //    anari::newArray1D(anari.device, clipPlane, 1));

  anari::commitParameters(anari.device, anari.renderer);
}

Renderer::~Renderer()
{
  //cam.reset(nullptr);
  anari::release(anari.device, anari.world);
  anari::release(anari.device, anari.renderer);
  anari::release(anari.device, anari.frame);
  anari::release(anari.device, anari.device);
  anari::unloadLibrary(anari.library);
}

void Renderer::on_display()
{
  anari::render(anari.device, anari.frame);
  anari::wait(anari.device, anari.frame);

  auto channelColor = anari::map<uint32_t>(anari.device, anari.frame, "channel.color");

  glDrawPixels(width(), height(), GL_RGBA, GL_UNSIGNED_BYTE, channelColor.data);

  anari::unmap(anari.device, anari.frame, "channel.color");

  ImGui::Begin("Parameters");
  bool updated = false;
  if (ImGui::DragFloat3("Light dir", (float *)g_lightDir.data())) {
    addPlaneAndArrows(anari.device, anari.world);
    updated = true;
  }

  const char *selected = g_selectedMaterial;
  if (ImGui::BeginCombo("Material Type", g_selectedMaterial)) {
    if (ImGui::Selectable("Matte", std::string(g_selectedMaterial)=="Matte")) {
      selected = "Matte";
    }
    else if (ImGui::Selectable("PBM", std::string(g_selectedMaterial)=="PBM")) {
      selected = "PBM";
    }
    if (std::string(selected) != std::string(g_selectedMaterial)) {
      g_selectedMaterial = selected;
      updated = true;
    }
    ImGui::EndCombo();
  }

  if (std::string(g_selectedMaterial)=="PBM") {
    updated |= ImGui::DragFloat("Metallic", &g_metallic, g_metallic, 0.f, 1.f);
    updated |= ImGui::DragFloat("Roughness", &g_roughness, g_roughness, 0.f, 1.f);
    updated |= ImGui::DragFloat("Clearcoat", &g_clearcoat, g_clearcoat, 0.f, 1.f);
    updated |= ImGui::DragFloat("Clearcoat roughness",
        &g_clearcoatRoughness, g_clearcoatRoughness, 0.f, 1.f);
    updated |= ImGui::DragFloat("IOR", &g_ior, g_ior, 0.f, 10.f);
  }

  if (ImGui::Checkbox("Show axes", &g_showAxes)) {
    addPlaneAndArrows(anari.device, anari.world);
  }

  ImGui::SameLine();
  if (ImGui::Checkbox("Show light dir", &g_showLightDir)) {
    addPlaneAndArrows(anari.device, anari.world);
  }

  ImGui::SameLine();
  if (ImGui::Checkbox("Show ground plane", &g_showGroundPlane)) {
    addPlaneAndArrows(anari.device, anari.world);
  }

  ImGui::End();

  if (updated) {
    dco::Material mat = generateMaterial();

    auto surf = makeBRDFSurface(anari.device, mat);
    anari::setAndReleaseParameter(
        anari.device, anari.world, "surface", anari::newArray1D(anari.device, &surf));
    anari::commitParameters(anari.device, anari.world);
  }
  viewer_glut::on_display();
}

void Renderer::on_mouse_move(const visionaray::mouse_event &event)
{
  if (event.buttons() == mouse::NoButton)
    return;

  viewer_glut::on_mouse_move(event);

  cam->commit();

  // HACK:
  anari::setParameter(anari.device, cam->getAnariHandle(), "apertureRadius", 0.f);
  anari::commitParameters(anari.device, cam->getAnariHandle());
}

void Renderer::on_key_press(const visionaray::key_event &event)
{
  viewer_glut::on_key_press(event);

  // static float D = 0.4f;

  // if (event.key() == 'a')
  //   D -= .1f;
  // else if (event.key() == 'b')
  //   D += .1f;

  // anari::math::float4 clipPlane[] = {{ 0.707f, 0.f, -0.707f, D }};
  // anari::setAndReleaseParameter(
  //     anari.device, anari.renderer, "clipPlane",
  //     anari::newArray1D(anari.device, clipPlane, 1));

  // anari::commitParameters(anari.device, anari.renderer);
}

void Renderer::on_resize(int w, int h)
{
  if (!anari.frame) {
    anari.frame = anari::newObject<anari::Frame>(anari.device);
  }

  cam->set_viewport(0, 0, w, h);
  float aspect = w / float(h);
  cam->perspective(60.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
  cam->commit();

  // HACK:
  anari::setParameter(anari.device, cam->getAnariHandle(), "apertureRadius", 0.f);
  anari::commitParameters(anari.device, cam->getAnariHandle());

  anari::math::uint2 size(w, h);

  anari::setParameter(anari.device, anari.frame, "world", anari.world);
  anari::setParameter(anari.device, anari.frame, "renderer", anari.renderer);
  anari::setParameter(anari.device, anari.frame, "camera", cam->getAnariHandle());
  //anari::setParameter(anari.device, anari.frame, "channel.color", ANARI_UFIXED8_VEC4);
  anari::setParameter(anari.device, anari.frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(anari.device, anari.frame, "size", size);
  anari::commitParameters(anari.device, anari.frame);

  viewer_glut::on_resize(w, h);
}

int main(int argc, char *argv[]) {
  Renderer rend;
  try {
    rend.init(argc, argv);
  } catch (...) {
    return EXIT_FAILURE;
  }

  rend.add_manipulator(std::make_shared<arcball_manipulator>(*rend.cam, mouse::Left));
  rend.add_manipulator(std::make_shared<pan_manipulator>(*rend.cam, mouse::Middle) );
  // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
  rend.add_manipulator(std::make_shared<pan_manipulator>(*rend.cam, mouse::Left, keyboard::Alt));
  rend.add_manipulator(std::make_shared<zoom_manipulator>(*rend.cam, mouse::Right));
  rend.event_loop();
}
