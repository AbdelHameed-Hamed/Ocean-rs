use std::process::Command;

fn main() {
    // dxc -T vs_6_0 -E vs_main -Zi -spirv ./assets/shaders/triangle.hlsl -Fo ./shaders/triangle.vert.spv
    Command::new("dxc")
        .args(&[
            "-T",
            "vs_6_0",
            "-E",
            "vs_main",
            "-Zi",
            "-spirv",
            "./assets/shaders/triangle.hlsl",
            "-Fo",
            "./shaders/triangle.vert.spv",
        ])
        .status()
        .unwrap();
    Command::new("dxc")
        .args(&[
            "-T",
            "ps_6_0",
            "-E",
            "fs_main",
            "-Zi",
            "-spirv",
            "./assets/shaders/triangle.hlsl",
            "-Fo",
            "./shaders/triangle.frag.spv",
        ])
        .status()
        .unwrap();

    println!("cargo:rerun-if-changed=assets/shaders/triangle.hlsl");
}
