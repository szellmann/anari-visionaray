
#pragma once

// std
#include <vector>
// ours
#include "DeviceCopyableObjects.h"

namespace visionaray {

// ==================================================================
// host/device array storing device object handles
// ==================================================================

struct DeviceHandleArray : public std::vector<DeviceObjectHandle>
{
 public:
  typedef DeviceObjectHandle value_type;
  typedef std::vector<DeviceObjectHandle> Base;

  DeviceHandleArray() = default;
  ~DeviceHandleArray() = default;

  void set(size_t index, DeviceObjectHandle handle)
  {
    if (index >= Base::size())
      Base::resize(index+1);

    Base::operator[](index) = handle;
  }

  const DeviceObjectHandle *hostPtr() const
  {
    return Base::data();
  }

  DeviceObjectHandle *devicePtr()
  {
    if (updated) {
      deviceData.resize(Base::size());
      // TODO: assert trivially copyable
      memcpy(deviceData.data(), Base::data(), Base::size() * sizeof(DeviceObjectHandle));
      updated = false;
    }
    return deviceData.data();
  }

  Base deviceData;
  bool updated = true;
};

// ==================================================================
// Array type capable of managing device-copyable objects via handles
// ==================================================================

template <typename T>
struct DeviceObjectArray : private std::vector<T>
{
 public:
  typedef typename std::vector<T>::value_type value_type;
  typedef std::vector<T> Base;

  DeviceObjectArray() = default;
  ~DeviceObjectArray() = default;

  DeviceObjectHandle alloc(const T &obj)
  {
    Base::push_back(obj);
    updated = true;
    return (DeviceObjectHandle)(Base::size()-1);
  }

  void free(DeviceObjectHandle handle)
  {
    updated = true;
  }

  void update(DeviceObjectHandle handle, const T &obj)
  {
    Base::data()[handle] = obj;
    updated = true;
  }

  size_t size() const
  {
    return Base::size();
  }

  void clear()
  {
    Base::clear();
    freeHandles.clear();
    updated = true;
  }

  const T &operator[](size_t index) const
  {
    return Base::operator[](index);
  }

  const T *hostPtr() const
  {
    return Base::data();
  }

  T *devicePtr()
  {
    if (updated) {
      deviceData.resize(Base::size());
      // TODO: assert trivially copyable
      memcpy(deviceData.data(), Base::data(), Base::size() * sizeof(T));
      updated = false;
    }
    return deviceData.data();
  }

  std::vector<DeviceObjectHandle> freeHandles;
  std::vector<T> deviceData;
  bool updated = true;
};

} // namespace visionaray
