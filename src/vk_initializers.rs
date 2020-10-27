extern crate ash;
extern crate sdl2;

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, vk::Handle, Device, Entry, Instance};
use sdl2::video::Window;
use std::borrow::Cow;
use std::ffi::{CStr, CString};

pub fn create_sdl_window(width: u32, height: u32) -> (sdl2::Sdl, sdl2::video::Window) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("Rust Vulkan Renderer", width, height)
        .vulkan()
        .build()
        .unwrap();

    return (sdl_context, window);
}

pub fn create_instance(window: &Window) -> (Entry, Instance) {
    let application_name = CString::new("Rust Vulkan Renderer").unwrap();

    let instance_extensions = window.vulkan_instance_extensions().unwrap();
    let mut extension_names_raw = instance_extensions
        .iter()
        .map(|ext| ext.as_ptr() as *const i8)
        .collect::<Vec<_>>();
    extension_names_raw.push(DebugUtils::name().as_ptr());

    let entry = Entry::new().unwrap();

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
        .api_version(vk::make_version(1, 1, 0));

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

    vk::FALSE
}

pub fn create_debug_layer(entry: &Entry, instance: &Instance) {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = DebugUtils::new(entry, instance);
    let debug_call_back = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap()
    };
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
                instance
                    .get_physical_device_queue_family_properties(*physical_device)
                    .iter()
                    .enumerate()
                    .filter_map(|(index, ref info)| {
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
                            Some((*physical_device, index))
                        } else {
                            None
                        }
                    })
                    .next()
            })
            .filter_map(|v| v)
            .next()
            .unwrap()
    };

    return (physical_device, queue_family_index as u32);
}

pub fn create_device(
    queue_family_index: u32,
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> Device {
    let device_extensions_names_raw = [Swapchain::name().as_ptr()];
    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };
    let priorities = [1.0];

    let queue_info = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)
        .build()];

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&device_extensions_names_raw)
        .enabled_features(&features);

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
        std::u32::MAX => vk::Extent2D {
            width: width,
            height: height,
        },
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
    let color_attachemt = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();
    let color_attachments = [color_attachemt];
    let color_attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();
    let color_attachment_references = [color_attachment_reference];

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_references)
        .build();
    let subpasses = [subpass];

    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&color_attachments)
        .subpasses(&subpasses)
        .build();
    return unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    };
}

pub fn create_framebuffers(
    swapchain_image_views: &Vec<vk::ImageView>,
    renderpass: &vk::RenderPass,
    width: u32,
    height: u32,
    device: &Device,
) -> Vec<vk::Framebuffer> {
    return swapchain_image_views
        .iter()
        .map(|&image| {
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(*renderpass)
                .attachments(&[image])
                .width(width)
                .height(height)
                .layers(1)
                .build();
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
) -> (vk::CommandPool, Vec<vk::CommandBuffer>) {
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    let command_pool = unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .unwrap()
    };

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();
    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
    };

    return (command_pool, command_buffer);
}

pub fn create_vertex_and_fragment_shader_modules(
    device: &Device,
) -> (vk::ShaderModule, vk::ShaderModule) {
    let mut shader_compiler = shaderc::Compiler::new().unwrap();

    let vertex_shader_source = std::fs::read_to_string("shaders/triangle.vert").unwrap();
    let vertex_shader_binary = shader_compiler
        .compile_into_spirv(
            vertex_shader_source.as_str(),
            shaderc::ShaderKind::Vertex,
            "triangle.vert",
            "main",
            None,
        )
        .unwrap();

    let vertex_shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
        .code(vertex_shader_binary.as_binary())
        .build();
    let vertex_shader_module = unsafe {
        device
            .create_shader_module(&vertex_shader_module_create_info, None)
            .unwrap()
    };

    let fragment_shader_source = std::fs::read_to_string("shaders/triangle.frag").unwrap();
    let fragment_shader_binary = shader_compiler
        .compile_into_spirv(
            fragment_shader_source.as_str(),
            shaderc::ShaderKind::Fragment,
            "triangle.frag",
            "main",
            None,
        )
        .unwrap();

    let fragment_shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
        .code(fragment_shader_binary.as_binary())
        .build();
    let fragment_shader_module = unsafe {
        device
            .create_shader_module(&fragment_shader_module_create_info, None)
            .unwrap()
    };

    return (vertex_shader_module, fragment_shader_module);
}

pub fn pipeline_shader_stage_create_info(
    shader_stage: vk::ShaderStageFlags,
    shader_module: vk::ShaderModule,
) -> vk::PipelineShaderStageCreateInfo {
    return vk::PipelineShaderStageCreateInfo::builder()
        .stage(shader_stage)
        .module(shader_module)
        .name(CStr::from_bytes_with_nul("main\0".as_bytes()).unwrap())
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
        .front_face(vk::FrontFace::CLOCKWISE)
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

pub fn pipeline_layout_create_info() -> vk::PipelineLayoutCreateInfo {
    return vk::PipelineLayoutCreateInfo::default();
}
