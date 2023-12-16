
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
    updated = true;
    return (DeviceObjectHandle)(Base::size()-1);
  }

  bool allocAt(DeviceObjectHandle handle, const T &obj)
  {
    if (handle >= Base::size()) {
      Base::resize(handle+1);
      (*this)[handle] = obj;
      updated = true;
      return true;
    } else {
      auto it = std::find(freeHandles.begin(), freeHandles.end(), handle);
      if (it == freeHandles.end())
        return false;
      else {
        (*this)[handle] = obj;
        updated = true;
        freeHandles.erase(it);
        return true;
      }
    }
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

  void clear()
  {
    Base::clear();
    freeHandles.clear();
    updated = true;
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
