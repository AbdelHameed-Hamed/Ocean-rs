extern crate ash;
extern crate sdl2;

use crate::vk_initializers;

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device, Instance};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::video::Window;

struct VkPipelineBuilder {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    vertex_input_info: vk::PipelineVertexInputStateCreateInfo,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling: vk::PipelineMultisampleStateCreateInfo,
    pipeline_layout: vk::PipelineLayout,
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
            pipeline_layout: vk::PipelineLayout::default(),
        };
    }

    pub fn build_pipline(&self, render_pass: &vk::RenderPass, device: &Device) -> vk::Pipeline {
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&[self.viewport])
            .scissors(&[self.scissor])
            .build();

        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&[self.color_blend_attachment])
            .build();

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(self.shader_stages.as_slice())
            .vertex_input_state(&self.vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blend_state_create_info)
            .layout(self.pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .build();
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .unwrap()
        };

        return pipeline[0];
    }
}

pub struct VkEngine {
    sdl_context: sdl2::Sdl,
    window: Window,
    size: vk::Extent2D,
    frame_count: u32,
    instance: Instance,
    debug_utils_loader: DebugUtils,
    debug_callback: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: Surface,
    physical_device: vk::PhysicalDevice,
    device: Device,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_format: vk::Format,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    graphics_queue: vk::Queue,
    graphics_queue_family_index: u32,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    render_fence: vk::Fence,
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    triangle_vertex_shader_module: vk::ShaderModule,
    triangle_fragment_shader_module: vk::ShaderModule,
    triangle_pipeline_layout: vk::PipelineLayout,
    triangle_pipeline: vk::Pipeline,
}

impl VkEngine {
    pub fn new(width: u32, height: u32) -> VkEngine {
        let (sdl_context, window) = vk_initializers::create_sdl_window(width, height);

        let (entry, instance) = vk_initializers::create_instance(&window);

        let (debug_utils_loader, debug_callback) =
            vk_initializers::create_debug_layer(&entry, &instance);

        let (surface, surface_loader) = vk_initializers::create_surface(&window, &entry, &instance);

        let (physical_device, queue_family_index) =
            vk_initializers::get_physical_device_and_graphics_queue_family_index(
                &instance,
                &surface_loader,
                &surface,
            );

        let device =
            vk_initializers::create_device(queue_family_index, &instance, &physical_device);

        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let (swapchain_loader, swapchain) = vk_initializers::create_swapchain_loader_and_swapchain(
            &surface_loader,
            &physical_device,
            &surface,
            width,
            height,
            &instance,
            &device,
            &surface_format,
        );

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        let swapchain_image_views = vk_initializers::create_swapchain_image_views(
            &swapchain_images,
            &surface_format,
            &device,
        );

        let render_pass = vk_initializers::create_renderpass(&surface_format, &device);

        let framebuffers = vk_initializers::create_framebuffers(
            &swapchain_image_views,
            &render_pass,
            width,
            height,
            &device,
        );

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let (command_pool, command_buffers) =
            vk_initializers::create_command_pool_and_buffer(queue_family_index, &device);

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        let render_fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let render_semaphore = unsafe {
            device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap()
        };
        let present_semaphore = unsafe {
            device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap()
        };

        let triangle_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&vk_initializers::pipeline_layout_create_info(), None)
                .unwrap()
        };

        let (triangle_vertex_shader_module, triangle_fragment_shader_module) =
            vk_initializers::create_vertex_and_fragment_shader_modules(&device);

        let mut pipeline_builder = VkPipelineBuilder::default();

        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::VERTEX,
                triangle_vertex_shader_module,
                "VSMain\0",
            ));
        pipeline_builder
            .shader_stages
            .push(vk_initializers::pipeline_shader_stage_create_info(
                vk::ShaderStageFlags::FRAGMENT,
                triangle_fragment_shader_module,
                "FSMain\0",
            ));

        pipeline_builder.vertex_input_info = vk_initializers::vertex_input_state_create_info();

        pipeline_builder.input_assembly =
            vk_initializers::input_assembly_create_info(&vk::PrimitiveTopology::TRIANGLE_LIST);

        pipeline_builder.viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(width as f32)
            .height(height as f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32)
            .build();

        pipeline_builder.scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: width,
                height: height,
            })
            .build();

        pipeline_builder.rasterizer =
            vk_initializers::rasterization_state_create_info(vk::PolygonMode::FILL);

        pipeline_builder.multisampling = vk_initializers::multisampling_state_create_info();

        pipeline_builder.color_blend_attachment = vk_initializers::color_blend_attachment_state();

        pipeline_builder.pipeline_layout = triangle_pipeline_layout;

        let triangle_pipeline = pipeline_builder.build_pipline(&render_pass, &device);

        return VkEngine {
            sdl_context: sdl_context,
            window: window,
            size: vk::Extent2D { width, height },
            frame_count: 0,
            instance: instance,
            debug_utils_loader: debug_utils_loader,
            debug_callback: debug_callback,
            surface: surface,
            surface_loader: surface_loader,
            physical_device: physical_device,
            device: device,
            swapchain_loader: swapchain_loader,
            swapchain: swapchain,
            swapchain_image_format: surface_format.format,
            swapchain_images: swapchain_images,
            swapchain_image_views: swapchain_image_views,
            render_pass: render_pass,
            framebuffers: framebuffers,
            graphics_queue: graphics_queue,
            graphics_queue_family_index: queue_family_index,
            command_pool: command_pool,
            command_buffer: command_buffers[0], // Should probably do something about this.
            render_fence: render_fence,
            render_semaphore: render_semaphore,
            present_semaphore: present_semaphore,
            triangle_vertex_shader_module: triangle_vertex_shader_module,
            triangle_fragment_shader_module: triangle_fragment_shader_module,
            triangle_pipeline_layout: triangle_pipeline_layout,
            triangle_pipeline: triangle_pipeline,
        };
    }

    pub fn cleanup(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device
                .destroy_shader_module(self.triangle_vertex_shader_module, None);
            self.device
                .destroy_shader_module(self.triangle_fragment_shader_module, None);

            self.device.destroy_semaphore(self.present_semaphore, None);
            self.device.destroy_semaphore(self.render_semaphore, None);
            self.device.destroy_fence(self.render_fence, None);

            self.device.destroy_command_pool(self.command_pool, None);

            for &framebuffer in self.framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            self.device.destroy_pipeline(self.triangle_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.triangle_pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            for &image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);

            self.instance.destroy_instance(None);
        };
    }

    pub fn draw(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.render_fence], true, std::u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.render_fence]).unwrap();
        };

        let swapchain_image_index = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    std::u64::MAX,
                    self.present_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
                .0
        };

        unsafe {
            self.device
                .reset_command_buffer(
                    self.command_buffer,
                    vk::CommandBufferResetFlags::from_raw(0),
                )
                .unwrap();

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();
            self.device
                .begin_command_buffer(self.command_buffer, &command_buffer_begin_info)
                .unwrap();
        };

        let flash = ((self.frame_count as f32) / 120.0f32).sin().abs();
        let mut clear_value = vk::ClearValue::default();
        clear_value.color = vk::ClearColorValue {
            float32: [0.0f32, 0.0f32, flash, 1.0f32],
        };

        let renderpass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.size,
            })
            .framebuffer(self.framebuffers[swapchain_image_index as usize])
            .clear_values(&[clear_value])
            .build();
        unsafe {
            self.device.cmd_begin_render_pass(
                self.command_buffer,
                &renderpass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.triangle_pipeline,
            );
            self.device.cmd_draw(self.command_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(self.command_buffer);
            self.device.end_command_buffer(self.command_buffer).unwrap();
        };

        let submit_info = vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .wait_semaphores(&[self.present_semaphore])
            .signal_semaphores(&[self.render_semaphore])
            .command_buffers(&[self.command_buffer])
            .build();
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], self.render_fence)
                .unwrap();
        };

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&[self.swapchain])
            .wait_semaphores(&[self.render_semaphore])
            .image_indices(&[swapchain_image_index])
            .build();
        unsafe {
            self.swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
                .unwrap();
        };

        self.frame_count += 1;
    }

    pub fn run(&mut self) {
        let mut event_pump = self.sdl_context.event_pump().unwrap();

        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape),
                        ..
                    } => {
                        break 'running;
                    }
                    _ => {}
                }
            }

            self.draw();
        }
    }
}
