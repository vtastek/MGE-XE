#include "vkrender.h"
#include "vkshaders.h"
#include "support/log.h"

#include <vulkan/vulkan.h>

#include <cstring>
#include <vector>
#include <windows.h>

// Milestone A: offscreen render + CPU readback. Everything is single-buffered and
// fully serialized (submit, wait fence, memcpy) — correctness over latency. The
// transport is replaced wholesale in Milestone B; this validates the rest of the
// pipeline (process + IPC + Vulkan + composite + present) first.

namespace {
    struct VK {
        bool ready = false;
        std::uint32_t width = 0, height = 0;

        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice phys = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        std::uint32_t gfxFamily = 0;
        VkQueue queue = VK_NULL_HANDLE;

        VkImage colorImage = VK_NULL_HANDLE;
        VkDeviceMemory colorMem = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE;

        VkRenderPass renderPass = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;

        VkBuffer readback = VK_NULL_HANDLE;
        VkDeviceMemory readbackMem = VK_NULL_HANDLE;
        void* readbackMapped = nullptr;
        VkDeviceSize readbackBytes = 0;
    };
    VK g;

    inline double msPerTick() {
        static const double v = [] {
            LARGE_INTEGER f; QueryPerformanceFrequency(&f);
            return 1000.0 / static_cast<double>(f.QuadPart);
        }();
        return v;
    }

    bool findMemoryType(std::uint32_t typeBits, VkMemoryPropertyFlags want, std::uint32_t& out) {
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(g.phys, &props);
        for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags & want) == want) {
                out = i;
                return true;
            }
        }
        return false;
    }

    VkShaderModule makeShader(const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(g.device, &ci, nullptr, &m) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return m;
    }

    bool createInstanceAndDevice() {
        VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "mgeHost64-spike";
        app.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &g.instance) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateInstance failed");
            return false;
        }

        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(g.instance, &count, nullptr);
        if (count == 0) {
            LOG::logline("!! [vk] no Vulkan physical devices");
            return false;
        }
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(g.instance, &count, devs.data());

        // Prefer a discrete GPU, else take the first.
        g.phys = devs[0];
        for (auto d : devs) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                g.phys = d;
                break;
            }
        }
        {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(g.phys, &p);
            LOG::logline(">> [vk] device: %s", p.deviceName);
        }

        std::uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qfCount, qfs.data());
        bool found = false;
        for (std::uint32_t i = 0; i < qfCount; ++i) {
            if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                g.gfxFamily = i;
                found = true;
                break;
            }
        }
        if (!found) {
            LOG::logline("!! [vk] no graphics queue family");
            return false;
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = g.gfxFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(g.phys, &dci, nullptr, &g.device) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateDevice failed");
            return false;
        }
        vkGetDeviceQueue(g.device, g.gfxFamily, 0, &g.queue);
        return true;
    }

    bool createImageTargets() {
        const VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;   // matches D3DFMT_X8R8G8B8 byte order

        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = { g.width, g.height, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g.device, &ici, nullptr, &g.colorImage) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateImage failed");
            return false;
        }

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(g.device, g.colorImage, &mr);
        std::uint32_t memType = 0;
        if (!findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memType)) {
            LOG::logline("!! [vk] no device-local memory type for color image");
            return false;
        }
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memType;
        if (vkAllocateMemory(g.device, &mai, nullptr, &g.colorMem) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkAllocateMemory (color) failed");
            return false;
        }
        vkBindImageMemory(g.device, g.colorImage, g.colorMem, 0);

        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = g.colorImage;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(g.device, &vci, nullptr, &g.colorView) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateImageView failed");
            return false;
        }

        // Readback buffer (host-visible, persistently mapped).
        g.readbackBytes = static_cast<VkDeviceSize>(g.width) * g.height * 4;
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = g.readbackBytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(g.device, &bci, nullptr, &g.readback) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateBuffer (readback) failed");
            return false;
        }
        VkMemoryRequirements bmr;
        vkGetBufferMemoryRequirements(g.device, g.readback, &bmr);
        std::uint32_t bMemType = 0;
        if (!findMemoryType(bmr.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bMemType)) {
            LOG::logline("!! [vk] no host-visible memory type for readback");
            return false;
        }
        VkMemoryAllocateInfo bmai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        bmai.allocationSize = bmr.size;
        bmai.memoryTypeIndex = bMemType;
        if (vkAllocateMemory(g.device, &bmai, nullptr, &g.readbackMem) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkAllocateMemory (readback) failed");
            return false;
        }
        vkBindBufferMemory(g.device, g.readback, g.readbackMem, 0);
        if (vkMapMemory(g.device, g.readbackMem, 0, g.readbackBytes, 0, &g.readbackMapped) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkMapMemory (readback) failed");
            return false;
        }
        return true;
    }

    bool createRenderPassAndPipeline() {
        const VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;

        VkAttachmentDescription color{};
        color.format = fmt;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // ready for copy-to-buffer

        VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &colorRef;

        // Guarantee the copy (TRANSFER) sees the color writes and the layout
        // transition to TRANSFER_SRC happens-before the copy reads.
        VkSubpassDependency dep{};
        dep.srcSubpass = 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        rpci.attachmentCount = 1;
        rpci.pAttachments = &color;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &sub;
        rpci.dependencyCount = 1;
        rpci.pDependencies = &dep;
        if (vkCreateRenderPass(g.device, &rpci, nullptr, &g.renderPass) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateRenderPass failed");
            return false;
        }

        VkFramebufferCreateInfo fci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fci.renderPass = g.renderPass;
        fci.attachmentCount = 1;
        fci.pAttachments = &g.colorView;
        fci.width = g.width;
        fci.height = g.height;
        fci.layers = 1;
        if (vkCreateFramebuffer(g.device, &fci, nullptr, &g.framebuffer) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateFramebuffer failed");
            return false;
        }

        VkShaderModule vs = makeShader(kTriVertSpv, sizeof(kTriVertSpv));
        VkShaderModule fs = makeShader(kTriFragSpv, sizeof(kTriFragSpv));
        if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
            LOG::logline("!! [vk] shader module creation failed");
            return false;
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{ 0.0f, 0.0f, (float)g.width, (float)g.height, 0.0f, 1.0f };
        VkRect2D sc{ {0, 0}, { g.width, g.height } };
        VkPipelineViewportStateCreateInfo vps{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        vps.viewportCount = 1;
        vps.pViewports = &vp;
        vps.scissorCount = 1;
        vps.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        if (vkCreatePipelineLayout(g.device, &plci, nullptr, &g.pipeLayout) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreatePipelineLayout failed");
            vkDestroyShaderModule(g.device, vs, nullptr);
            vkDestroyShaderModule(g.device, fs, nullptr);
            return false;
        }

        VkGraphicsPipelineCreateInfo gp{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb;
        gp.layout = g.pipeLayout;
        gp.renderPass = g.renderPass;
        gp.subpass = 0;
        VkResult pr = vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &gp, nullptr, &g.pipeline);
        vkDestroyShaderModule(g.device, vs, nullptr);
        vkDestroyShaderModule(g.device, fs, nullptr);
        if (pr != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateGraphicsPipelines failed");
            return false;
        }
        return true;
    }

    bool createCommandsAndFence() {
        VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = g.gfxFamily;
        if (vkCreateCommandPool(g.device, &pci, nullptr, &g.cmdPool) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateCommandPool failed");
            return false;
        }
        VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = g.cmdPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(g.device, &cai, &g.cmd) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkAllocateCommandBuffers failed");
            return false;
        }
        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(g.device, &fci, nullptr, &g.fence) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkCreateFence failed");
            return false;
        }
        return true;
    }
}

namespace VKRender {
    bool isReady() { return g.ready; }

    bool init(std::uint32_t width, std::uint32_t height) {
        if (g.ready && g.width == width && g.height == height) {
            return true;
        }
        if (g.ready) {
            shutdown();
        }
        g.width = width;
        g.height = height;

        if (!createInstanceAndDevice()) { shutdown(); return false; }
        if (!createImageTargets())      { shutdown(); return false; }
        if (!createRenderPassAndPipeline()) { shutdown(); return false; }
        if (!createCommandsAndFence())  { shutdown(); return false; }

        g.ready = true;
        LOG::logline(">> [vk] renderer ready (%ux%u, B8G8R8A8, offscreen+readback)", width, height);
        return true;
    }

    bool renderFrame(void* outPixels, std::uint32_t outBytes, double* cpuMs) {
        if (!g.ready) return false;
        if (outBytes < g.readbackBytes) {
            LOG::logline("!! [vk] renderFrame outBytes %u < needed %llu", outBytes,
                         (unsigned long long)g.readbackBytes);
            return false;
        }

        LARGE_INTEGER t0; QueryPerformanceCounter(&t0);

        vkResetCommandBuffer(g.cmd, 0);
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(g.cmd, &bi);

        VkClearValue clear{};
        clear.color = { { 0.08f, 0.08f, 0.12f, 1.0f } };  // dark slate so the triangle reads clearly
        VkRenderPassBeginInfo rp{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rp.renderPass = g.renderPass;
        rp.framebuffer = g.framebuffer;
        rp.renderArea = { {0, 0}, { g.width, g.height } };
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;
        vkCmdBeginRenderPass(g.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pipeline);
        vkCmdDraw(g.cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(g.cmd);

        // Image is now TRANSFER_SRC_OPTIMAL (render-pass finalLayout). Copy to buffer.
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;     // tightly packed
        region.bufferImageHeight = 0;
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { g.width, g.height, 1 };
        vkCmdCopyImageToBuffer(g.cmd, g.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               g.readback, 1, &region);

        vkEndCommandBuffer(g.cmd);

        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g.cmd;
        vkResetFences(g.device, 1, &g.fence);
        if (vkQueueSubmit(g.queue, 1, &si, g.fence) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkQueueSubmit failed");
            return false;
        }
        if (vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            LOG::logline("!! [vk] vkWaitForFences failed");
            return false;
        }

        std::memcpy(outPixels, g.readbackMapped, g.readbackBytes);

        LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
        if (cpuMs) *cpuMs = double(t1.QuadPart - t0.QuadPart) * msPerTick();
        return true;
    }

    void shutdown() {
        if (g.device) {
            vkDeviceWaitIdle(g.device);
            if (g.fence)        vkDestroyFence(g.device, g.fence, nullptr);
            if (g.cmdPool)      vkDestroyCommandPool(g.device, g.cmdPool, nullptr);
            if (g.pipeline)     vkDestroyPipeline(g.device, g.pipeline, nullptr);
            if (g.pipeLayout)   vkDestroyPipelineLayout(g.device, g.pipeLayout, nullptr);
            if (g.framebuffer)  vkDestroyFramebuffer(g.device, g.framebuffer, nullptr);
            if (g.renderPass)   vkDestroyRenderPass(g.device, g.renderPass, nullptr);
            if (g.readbackMapped) vkUnmapMemory(g.device, g.readbackMem);
            if (g.readback)     vkDestroyBuffer(g.device, g.readback, nullptr);
            if (g.readbackMem)  vkFreeMemory(g.device, g.readbackMem, nullptr);
            if (g.colorView)    vkDestroyImageView(g.device, g.colorView, nullptr);
            if (g.colorImage)   vkDestroyImage(g.device, g.colorImage, nullptr);
            if (g.colorMem)     vkFreeMemory(g.device, g.colorMem, nullptr);
            vkDestroyDevice(g.device, nullptr);
        }
        if (g.instance) vkDestroyInstance(g.instance, nullptr);

        VK fresh;
        g = fresh;
    }
}
