extern crate ash;
extern crate hassle_rs;
extern crate sdl2;

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::{vk, vk::Handle, Device, Entry, Instance};
use hassle_rs::compile_hlsl;
use sdl2::video::Window;
use std::borrow::Cow;
use std::ffi::{CStr, CString};

pub struct VkBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) buffer_memory: vk::DeviceMemory,
}

pub struct VkImage {
    pub(crate) image: vk::Image,
    pub(crate) image_memory: vk::DeviceMemory,
}

pub struct VkTexture {
    pub(crate) image: VkImage,
    pub(crate) image_view: vk::ImageView,
    pub(crate) sampler: vk::Sampler,
}

pub fn create_sdl_window(width: u32, height: u32) -> (sdl2::Sdl, sdl2::video::Window) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("Ocean", width, height)
        .vulkan()
        .build()
        .unwrap();

    return (sdl_context, window);
}

pub fn create_instance(window: &Window) -> (Entry, Instance) {
    let application_name = CString::new("Rust Vulkan Renderer").unwrap();

    let mut instance_extensions = window.vulkan_instance_extensions().unwrap();
    instance_extensions.push("VK_EXT_debug_utils");
    let mut extension_names_raw = instance_extensions
        .iter()
        .map(|ext| ext.as_ptr() as *const i8)
        .collect::<Vec<_>>();
    extension_names_raw.push(DebugUtils::name().as_ptr());

    let entry = unsafe { Entry::load().unwrap() };

    let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
    let layer_names_raw: Vec<*const i8> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(&application_name)
        .application_version(0)
        .engine_name(&application_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_2);

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names_raw)
        .enabled_extension_names(&extension_names_raw);

    let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

    return (entry, instance);
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    return vk::FALSE;
}

pub fn create_debug_layer(
    entry: &Entry,
    instance: &Instance,
) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = DebugUtils::new(entry, instance);
    let debug_callback = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap()
    };

    return (debug_utils_loader, debug_callback);
}

pub fn create_surface(
    window: &sdl2::video::Window,
    entry: &Entry,
    instance: &Instance,
) -> (vk::SurfaceKHR, Surface) {
    let surface_handle = window
        .vulkan_create_surface(vk::Instance::as_raw(instance.handle()) as usize)
        .unwrap();
    let surface = vk::SurfaceKHR::from_raw(surface_handle);
    let surface_loader = Surface::new(entry, instance);

    return (surface, surface_loader);
}

pub fn get_physical_device_and_graphics_queue_family_index(
    instance: &Instance,
    surface_loader: &Surface,
    surface: &vk::SurfaceKHR,
) -> (vk::PhysicalDevice, u32) {
    let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };

    let (physical_device, queue_family_index) = unsafe {
        physical_devices
            .iter()
            .map(|physical_device| {
                return instance
                    .get_physical_device_queue_family_properties(*physical_device)
                    .iter()
                    .enumerate()
                    .find_map(|(index, ref info)| {
                        let supports_graphic_and_surface =
                            info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                && surface_loader
                                    .get_physical_device_surface_support(
                                        *physical_device,
                                        index as u32,
                                        *surface,
                                    )
                                    .unwrap();
                        if supports_graphic_and_surface {
                            return Some((*physical_device, index));
                        } else {
                            return None;
                        }
                    });
            })
            .find_map(|v| v)
            .unwrap()
    };

    return (physical_device, queue_family_index as u32);
}

pub fn create_device(
    queue_family_index: u32,
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> Device {
    let device_extensions_names_raw = [
        Swapchain::name().as_ptr(),
        vk::KhrShaderNonSemanticInfoFn::name().as_ptr(),
    ];

    let priorities = [1.0];

    let queue_info = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)
        .build()];

    let mut features16 = vk::PhysicalDevice16BitStorageFeatures::builder()
        .storage_buffer16_bit_access(true)
        .uniform_and_storage_buffer16_bit_access(true);
    let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
        .descriptor_binding_partially_bound(true)
        .runtime_descriptor_array(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .descriptor_binding_uniform_buffer_update_after_bind(true)
        .descriptor_binding_storage_image_update_after_bind(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_variable_descriptor_count(true)
        .descriptor_binding_update_unused_while_pending(true);
    let mut features = vk::PhysicalDeviceFeatures2::builder().features(
        vk::PhysicalDeviceFeatures::builder()
            .shader_clip_distance(true)
            .shader_int16(true)
            .fill_mode_non_solid(true)
            .build(),
    );

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&device_extensions_names_raw)
        .push_next(&mut indexing_features)
        .push_next(&mut features16)
        .push_next(&mut features);

    let device = unsafe {
        instance
            .create_device(*physical_device, &device_create_info, None)
            .unwrap()
    };

    return device;
}

pub fn create_swapchain_loader_and_swapchain(
    surface_loader: &Surface,
    physical_device: &vk::PhysicalDevice,
    surface: &vk::SurfaceKHR,
    width: u32,
    height: u32,
    instance: &Instance,
    device: &Device,
    surface_format: &vk::SurfaceFormatKHR,
) -> (Swapchain, vk::SwapchainKHR) {
    let surface_capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(*physical_device, *surface)
            .unwrap()
    };
    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && desired_image_count > surface_capabilities.max_image_count
    {
        desired_image_count = surface_capabilities.max_image_count;
    }
    let surface_resolution = match surface_capabilities.current_extent.width {
        std::u32::MAX => vk::Extent2D { width, height },
        _ => surface_capabilities.current_extent,
    };
    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };
    let present_modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(*physical_device, *surface)
            .unwrap()
    };
    let present_mode = present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == vk::PresentModeKHR::FIFO)
        .unwrap_or(vk::PresentModeKHR::MAILBOX);
    let swapchain_loader = Swapchain::new(instance, device);

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(*surface)
        .min_image_count(desired_image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };

    return (swapchain_loader, swapchain);
}

pub fn create_swapchain_image_views(
    swapchain_images: &Vec<vk::Image>,
    surface_format: &vk::SurfaceFormatKHR,
    device: &Device,
) -> Vec<vk::ImageView> {
    return swapchain_images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);
            unsafe { device.create_image_view(&create_view_info, None).unwrap() }
        })
        .collect();
}

pub fn create_renderpass(surface_format: &vk::SurfaceFormatKHR, device: &Device) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();
    let color_attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();
    let color_attachment_references = [color_attachment_reference];

    let depth_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();
    let depth_attachment_reference = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_references)
        .depth_stencil_attachment(&depth_attachment_reference)
        .build();
    let subpasses = [subpass];

    let attachments = [color_attachment, depth_attachment];
    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses);
    return unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    };
}

pub fn create_framebuffers(
    swapchain_image_views: &Vec<vk::ImageView>,
    depth_image_view: vk::ImageView,
    render_pass: &vk::RenderPass,
    width: u32,
    height: u32,
    device: &Device,
) -> Vec<vk::Framebuffer> {
    return swapchain_image_views
        .iter()
        .map(|&image| {
            let attachments = [image, depth_image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(*render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1);
            unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .unwrap()
            }
        })
        .collect();
}

pub fn create_command_pool_and_buffer(
    queue_family_index: u32,
    device: &Device,
    command_buffer_count: u32,
) -> (vk::CommandPool, Vec<vk::CommandBuffer>) {
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .unwrap()
    };

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .command_buffer_count(command_buffer_count)
        .level(vk::CommandBufferLevel::PRIMARY);
    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
    };

    return (command_pool, command_buffers);
}

pub fn create_shader_module(
    device: &Device,
    shader_src: &str,
    shader_name: &str,
    entry_point: &str,
    target_profile: &str,
    args: &Vec<&str>,
    defines: &Vec<(&str, Option<&str>)>,
) -> vk::ShaderModule {
    let shader_binary_as_bytes = compile_hlsl(
        shader_name,
        shader_src,
        entry_point,
        target_profile,
        &args,
        &defines,
    )
    .unwrap();

    // !Note: This might bite me later due to endianess.
    let (_, shader_binary, _) = unsafe { shader_binary_as_bytes.align_to::<u32>() };

    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(shader_binary);
    let shader_module = unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .unwrap()
    };

    return shader_module;
}

pub fn pipeline_shader_stage_create_info(
    shader_stage: vk::ShaderStageFlags,
    shader_module: vk::ShaderModule,
    entry_point: &str,
) -> vk::PipelineShaderStageCreateInfo {
    return vk::PipelineShaderStageCreateInfo::builder()
        .stage(shader_stage)
        .module(shader_module)
        .name(CStr::from_bytes_with_nul(entry_point.as_bytes()).unwrap())
        .build();
}

pub fn vertex_input_state_create_info() -> vk::PipelineVertexInputStateCreateInfo {
    return vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&[])
        .vertex_attribute_descriptions(&[])
        .build();
}

pub fn input_assembly_create_info(
    primitive_topology: &vk::PrimitiveTopology,
) -> vk::PipelineInputAssemblyStateCreateInfo {
    return vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(*primitive_topology)
        .primitive_restart_enable(false)
        .build();
}

pub fn rasterization_state_create_info(
    polygon_mode: vk::PolygonMode,
) -> vk::PipelineRasterizationStateCreateInfo {
    return vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .line_width(1.0f32)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0f32)
        .depth_bias_clamp(0.0f32)
        .depth_bias_slope_factor(0.0f32)
        .build();
}

pub fn multisampling_state_create_info() -> vk::PipelineMultisampleStateCreateInfo {
    return vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0f32)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .build();
}

pub fn color_blend_attachment_state() -> vk::PipelineColorBlendAttachmentState {
    return vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)
        .build();
}

fn begin_single_time_command(device: &Device, command_pool: vk::CommandPool) -> vk::CommandBuffer {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()[0]
    };

    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .unwrap()
    };

    return command_buffer;
}

fn end_single_time_command(
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) {
    unsafe {
        device.end_command_buffer(command_buffer).unwrap();
    };

    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&[command_buffer])
        .build();
    let submit_infos = [submit_info];

    unsafe {
        device
            .queue_submit(graphics_queue, &submit_infos, vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(graphics_queue).unwrap();
        device.free_command_buffers(command_pool, &[command_buffer]);
    };
}

pub fn create_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> VkBuffer {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_create_info, None).unwrap() };

    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = memory_properties
        .memory_types
        .iter()
        .enumerate()
        .find_map(|(i, memory_type)| {
            if (memory_requirements.memory_type_bits & (1 << i)) > 0
                && memory_type.property_flags.contains(properties)
            {
                return Some(i as u32);
            } else {
                return None;
            }
        })
        .unwrap();

    let memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type);
    let buffer_memory = unsafe { device.allocate_memory(&memory_allocate_info, None).unwrap() };

    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap() };

    return VkBuffer {
        buffer,
        buffer_memory,
    };
}

pub fn copy_buffer(
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let buffer_copy_region = vk::BufferCopy::builder()
        .src_offset(0)
        .dst_offset(0)
        .size(size)
        .build();
    let buffer_copy_regions = [buffer_copy_region];
    unsafe {
        device.cmd_copy_buffer(command_buffer, src, dst, &buffer_copy_regions);
    };

    end_single_time_command(device, command_pool, graphics_queue, command_buffer);
}

pub unsafe fn free_buffer_and_memory(device: &Device, buffer: &VkBuffer) {
    device.destroy_buffer(buffer.buffer, None);
    device.free_memory(buffer.buffer_memory, None);
}

pub fn create_image_create_info(
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    extent: vk::Extent3D,
) -> vk::ImageCreateInfo {
    return vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .build();
}

pub fn create_image(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    create_info: vk::ImageCreateInfo,
    properties: vk::MemoryPropertyFlags,
) -> VkImage {
    let image = unsafe { device.create_image(&create_info, None).unwrap() };

    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = memory_properties
        .memory_types
        .iter()
        .enumerate()
        .find_map(|(i, memory_type)| {
            if (memory_requirements.memory_type_bits & (1 << i)) > 0
                && memory_type.property_flags.contains(properties)
            {
                return Some(i as u32);
            } else {
                return None;
            }
        })
        .unwrap();

    let memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type);
    let image_memory = unsafe { device.allocate_memory(&memory_allocate_info, None).unwrap() };

    unsafe { device.bind_image_memory(image, image_memory, 0).unwrap() };

    return VkImage {
        image,
        image_memory,
    };
}

pub fn transition_image_layout(
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let src_access = match old_layout {
        vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        _ => panic!("Dunno what old_layout this is"),
    };
    let dst_access = match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::GENERAL => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        _ => panic!("Dunno what new_layout this is"),
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .build();
    let barriers = [barrier];
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    };

    end_single_time_command(device, command_pool, graphics_queue, command_buffer);
}

pub fn copy_buffer_to_image(
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Image,
    size: vk::Extent3D,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let region = vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(size)
        .build();
    let regions = [region];
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            src,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    };

    end_single_time_command(device, command_pool, graphics_queue, command_buffer);
}

pub fn create_image_view_create_info(
    format: vk::Format,
    image: vk::Image,
    aspect_flags: vk::ImageAspectFlags,
) -> vk::ImageViewCreateInfo {
    return vk::ImageViewCreateInfo::builder()
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .aspect_mask(aspect_flags)
                .build(),
        )
        .build();
}

pub fn create_depth_stencil_create_info() -> vk::PipelineDepthStencilStateCreateInfo {
    return vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .min_depth_bounds(0.0f32)
        .max_depth_bounds(1.0f32)
        .stencil_test_enable(false)
        .build();
}

pub fn descriptor_set_layout_binding(
    desc_type: vk::DescriptorType,
    binding: u32,
    desc_count: u32,
    stage_flags: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayoutBinding {
    return vk::DescriptorSetLayoutBinding::builder()
        .descriptor_type(desc_type)
        .binding(binding)
        .descriptor_count(desc_count)
        .stage_flags(stage_flags)
        .build();
}

pub fn add_texture(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    extent: vk::Extent3D,
    format: vk::Format,
    staging_buffer: Option<VkBuffer>,
    name: &str,
    textures: &mut std::collections::HashMap<String, VkTexture>,
) -> vk::DescriptorImageInfo {
    let texture_img_info = create_image_create_info(
        format,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE,
        extent,
    );
    let texture_img = create_image(
        &instance,
        physical_device,
        &device,
        texture_img_info,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    let image_layout;
    if let Some(staging_buffer) = staging_buffer {
        transition_image_layout(
            &device,
            command_pool,
            graphics_queue,
            texture_img.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        copy_buffer_to_image(
            &device,
            command_pool,
            graphics_queue,
            staging_buffer.buffer,
            texture_img.image,
            extent,
        );
        unsafe { free_buffer_and_memory(&device, &staging_buffer) };
        transition_image_layout(
            &device,
            command_pool,
            graphics_queue,
            texture_img.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    } else {
        transition_image_layout(
            &device,
            command_pool,
            graphics_queue,
            texture_img.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        image_layout = vk::ImageLayout::GENERAL;
    }

    let texture_image_view_info =
        create_image_view_create_info(format, texture_img.image, vk::ImageAspectFlags::COLOR);
    let texture_image_view = unsafe {
        device
            .create_image_view(&texture_image_view_info, None)
            .unwrap()
    };

    let texture_sampler_info = vk::SamplerCreateInfo {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
        unnormalized_coordinates: vk::FALSE,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::NEVER,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        ..Default::default()
    };
    let texture_sampler = unsafe { device.create_sampler(&texture_sampler_info, None).unwrap() };

    let texture = VkTexture {
        image: texture_img,
        image_view: texture_image_view,
        sampler: texture_sampler,
    };
    textures.insert(name.to_string(), texture);

    return vk::DescriptorImageInfo {
        image_layout,
        image_view: textures[name].image_view,
        sampler: textures[name].sampler,
        ..Default::default()
    };
}

pub fn return_aligned_size(
    physical_device_properties: vk::PhysicalDeviceProperties,
    original_size: usize,
) -> usize {
    let min_ubo_alignment = physical_device_properties
        .limits
        .min_uniform_buffer_offset_alignment as usize;
    let aligned_size = if min_ubo_alignment > 0 {
        (original_size + min_ubo_alignment - 1) & !(min_ubo_alignment - 1)
    } else {
        original_size
    };

    return aligned_size;
}

pub struct VkPipelineBuilder {
    pub(crate) shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    pub(crate) vertex_input_info: vk::PipelineVertexInputStateCreateInfo,
    pub(crate) input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    pub(crate) viewport: vk::Viewport,
    pub(crate) scissor: vk::Rect2D,
    pub(crate) rasterizer: vk::PipelineRasterizationStateCreateInfo,
    pub(crate) color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    pub(crate) multisampling: vk::PipelineMultisampleStateCreateInfo,
    pub(crate) depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) dynamic_state: vk::PipelineDynamicStateCreateInfo,
}

impl VkPipelineBuilder {
    pub fn default() -> VkPipelineBuilder {
        return VkPipelineBuilder {
            shader_stages: Vec::<vk::PipelineShaderStageCreateInfo>::default(),
            vertex_input_info: vk::PipelineVertexInputStateCreateInfo::default(),
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default(),
            viewport: vk::Viewport::default(),
            scissor: vk::Rect2D::default(),
            rasterizer: vk::PipelineRasterizationStateCreateInfo::default(),
            color_blend_attachment: vk::PipelineColorBlendAttachmentState::default(),
            multisampling: vk::PipelineMultisampleStateCreateInfo::default(),
            depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo::default(),
            pipeline_layout: vk::PipelineLayout::default(),
            dynamic_state: vk::PipelineDynamicStateCreateInfo::default(),
        };
    }

    pub fn build_pipline(&self, render_pass: &vk::RenderPass, device: &Device) -> vk::Pipeline {
        let viewports = [self.viewport];
        let scissors = [self.scissor];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let attachments = [self.color_blend_attachment];
        let color_blend_state_create_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachments);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(self.shader_stages.as_slice())
            .vertex_input_state(&self.vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blend_state_create_info)
            .depth_stencil_state(&self.depth_stencil_state)
            .layout(self.pipeline_layout)
            .render_pass(*render_pass)
            .dynamic_state(&self.dynamic_state)
            .build();
        let pipeline_create_infos = [pipeline_create_info];
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .unwrap()
        };

        return pipeline[0];
    }
}
