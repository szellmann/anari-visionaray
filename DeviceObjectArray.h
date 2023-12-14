
#pragma once

namespace visionaray {

typedef uint32_t DeviceObjectHandle;

template <typename T>
struct DeviceObjectArray : public std::vector<T>
{
 public:
  typedef typename std::vector<T>::value_type value_type;
  typedef std::vector<T> Base;

  DeviceObjectArray() = default;
  ~DeviceObjectArray() = default;

  DeviceObjectHandle alloc(const T &obj)
  {
    Base::push_back(obj);
    return (DeviceObjectHandle)(Base::size()-1);
  }

  void free(DeviceObjectHandle handle)
  {
  }

  void update(DeviceObjectHandle handle, const T &obj)
  {
    Base::data()[handle] = obj;
  }

  std::vector<DeviceObjectHandle> freeHandles;
};

} // namespace visionaray
