use std::fs;
use std::process::Command;

// Example:
// dxc -T ms_6_5 -E ms_main -Zi -spirv ./assets/shaders/skybox.mesh.frag.hlsl -Fo ./shaders/skybox.mesh.spv

fn main() {
    // First make sure that the outermost shaders folder exists
    if let Err(_) = fs::read_dir("./shaders/") {
        fs::create_dir("./shaders").unwrap();
    }

    for entry in fs::read_dir("./assets/shaders/").expect("Path doesn't exist") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() && (path.extension().unwrap() == "hlsl") {
            let filename = path.file_name().unwrap().to_str().unwrap();

            let filename_tokens = filename.split('.').collect::<Vec<&str>>();
            let shader_name = filename_tokens[0];
            let shader_types = &filename_tokens[1..filename_tokens.len() - 1];
            for shader_type in shader_types {
                let mut command_args = [
                    "-T",
                    "INVALID",
                    "-E",
                    "INVALID",
                    "-Zi",
                    "-Od",
                    "-spirv",
                    path.to_str().unwrap(),
                    "-Fo",
                    "INVALID",
                    "-enable-16bit-types",
                ];
                let output_file_name: String;
                match *shader_type {
                    "task" => {
                        output_file_name = format!("./shaders/{}.task.spv", shader_name);
                        command_args[1] = "as_6_5";
                        command_args[3] = "ts_main";
                    }
                    "mesh" => {
                        output_file_name = format!("./shaders/{}.mesh.spv", shader_name);
                        command_args[1] = "ms_6_5";
                        command_args[3] = "ms_main";
                    }
                    "vert" => {
                        output_file_name = format!("./shaders/{}.vert.spv", shader_name);
                        command_args[1] = "vs_6_5";
                        command_args[3] = "vs_main";
                    }
                    "frag" => {
                        output_file_name = format!("./shaders/{}.frag.spv", shader_name);
                        command_args[1] = "ps_6_5";
                        command_args[3] = "fs_main";
                    }
                    "comp" => {
                        output_file_name = format!("./shaders/{}.comp.spv", shader_name);
                        command_args[1] = "cs_6_5";
                        command_args[3] = "cs_main";
                    }
                    _ => {
                        panic!("Unsupported shader type: {}", shader_type);
                    }
                }
                command_args[9] = output_file_name.as_str();
                eprintln!(
                    "{}",
                    Command::new("dxc").args(&command_args).status().unwrap()
                );
            }
        } else {
            panic!("Not an HLSL file");
        }
    }

    println!("cargo:rerun-if-changed=assets/shaders/");
}
