// Headless Vulkan compute scaffold. No surface, no swapchain, no validation
// layers: instance -> discrete GPU -> compute queue -> one dispatch filling
// a linear-HDR buffer -> host readback -> existing PPM writer. Every later
// GPU slice (intersection kernels, wavefront stages) reuses this exact
// setup/teardown shape with different shaders.
#include "../io/ppm.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define VK_CHECK(x)                                                                               \
    do                                                                                            \
    {                                                                                             \
        VkResult vk_check_r = (x);                                                                \
        if (vk_check_r != VK_SUCCESS)                                                             \
        {                                                                                         \
            std::fprintf(stderr, "Vulkan error %d at %s:%d\n", vk_check_r, __FILE__, __LINE__);    \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

namespace
{

// Directory holding this executable: shaders ship next to the binary (see
// CMake POST_BUILD copy), so the app runs from anywhere.
std::filesystem::path exe_dir(const char *argv0)
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0)
        return std::filesystem::path(std::string(buf, n)).parent_path();
#endif
    return std::filesystem::path(argv0).parent_path();
}

std::vector<char> read_file(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", path.string().c_str());
        std::exit(1);
    }
    std::vector<char> bytes(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

// First queue family on device that serves compute (dedicated compute
// preferred over a shared graphics+compute one: async-friendly later).
uint32_t find_compute_family(VkPhysicalDevice device)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
    for (uint32_t i = 0; i < count; ++i)
        if ((props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            return i;
    for (uint32_t i = 0; i < count; ++i)
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    std::fprintf(stderr, "no compute queue family\n");
    std::exit(1);
}

uint32_t find_memory_type(VkPhysicalDevice device, uint32_t bits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties mem = {};
    vkGetPhysicalDeviceMemoryProperties(device, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    std::fprintf(stderr, "no suitable memory type\n");
    std::exit(1);
}

} // namespace

int main(int, char **argv)
{
    const uint32_t width = 800;
    const uint32_t height = 450;

    // --- instance (no extensions: headless compute needs none) ---
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance instance = nullptr;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    // --- physical device: first discrete GPU, else first compute-capable ---
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
    if (gpu_count == 0)
    {
        std::fprintf(stderr, "no Vulkan physical devices\n");
        return 1;
    }
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data()));
    VkPhysicalDevice gpu = gpus[0];
    for (auto g : gpus)
    {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(g, &props);
        std::printf("GPU: %s\n", props.deviceName);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            gpu = g;
            break;
        }
    }
    VkPhysicalDeviceProperties chosen = {};
    vkGetPhysicalDeviceProperties(gpu, &chosen);
    std::printf("using: %s\n", chosen.deviceName);

    uint32_t qfamily = find_compute_family(gpu);

    // --- logical device + queue ---
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice device = nullptr;
    VK_CHECK(vkCreateDevice(gpu, &dci, nullptr, &device));
    VkQueue queue = nullptr;
    vkGetDeviceQueue(device, qfamily, 0, &queue);

    // --- storage buffer: W*H RGB floats, host-visible for direct readback.
    // A real renderer stages through device-local memory; the scaffold skips
    // that copy on purpose (one allocation, zero transfers to get wrong).
    const VkDeviceSize buf_size = static_cast<VkDeviceSize>(width) * height * 3 * sizeof(float);
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = buf_size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = nullptr;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buffer));
    VkMemoryRequirements reqs = {};
    vkGetBufferMemoryRequirements(device, buffer, &reqs);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = reqs.size;
    mai.memoryTypeIndex = find_memory_type(gpu, reqs.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory memory = nullptr;
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    // --- descriptor set: binding 0 = the frame buffer ---
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dsli = {};
    dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = 1;
    dsli.pBindings = &binding;
    VkDescriptorSetLayout layout = nullptr;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &layout));

    // Push constants carry width/height: 8 bytes, no buffer round-trip.
    VkPushConstantRange push = {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = 8;
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout = nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout));

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.pPoolSizes = &pool_size;
    dpi.poolSizeCount = 1;
    VkDescriptorPool pool = nullptr;
    VK_CHECK(vkCreateDescriptorPool(device, &dpi, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout;
    VkDescriptorSet set = nullptr;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &set));
    VkDescriptorBufferInfo dbi = {};
    dbi.buffer = buffer;
    dbi.range = buf_size;
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    // --- compute pipeline ---
    std::vector<char> spirv = read_file(exe_dir(argv[0]) / "shaders" / "fill.spv");
    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size();
    smci.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
    VkShaderModule module = nullptr;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &module));
    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage = stage;
    pci.layout = pipeline_layout;
    VkPipeline pipeline = nullptr;
    VK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &pci, nullptr, &pipeline));

    // --- record, submit, wait ---
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = qfamily;
    VkCommandPool cmd_pool = nullptr;
    VK_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool));
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = nullptr;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    uint32_t dims[2] = {width, height};
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dims), dims);
    vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    // Make the shader write visible to the host before the fence signals.
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = qfamily;
    barrier.dstQueueFamilyIndex = qfamily;
    barrier.buffer = buffer;
    barrier.size = buf_size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = nullptr;
    VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    // --- readback -> existing PPM writer (tonemap applies as usual) ---
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(device, memory, 0, buf_size, 0, &mapped));
    const float *pixels = static_cast<const float *>(mapped);
    std::vector<vec3> framebuffer(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < framebuffer.size(); ++i)
        framebuffer[i] = vec3(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
    vkUnmapMemory(device, memory);
    std::uint64_t hash = write_ppm("gpu_fill.ppm", framebuffer, width, height, 1.0, true);
    std::printf("wrote gpu_fill.ppm hash=%llu\n", static_cast<unsigned long long>(hash));

    // --- teardown, reverse order of creation ---
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
