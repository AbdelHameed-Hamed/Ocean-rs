use crate::math::Vec3;

// Returns pair of vectors for vertices and faces
pub fn read_obj_file(file_path: &str) -> (Vec<Vec3>, Vec<(u32, u32, u32)>) {
    let file_content = std::fs::read_to_string(file_path).unwrap();

    let (mut vertices, mut faces) = (Vec::<Vec3>::new(), Vec::<(u32, u32, u32)>::new());

    'outer: for (i, line) in file_content
        .split(&['\r', '\n'][..])
        .collect::<Vec<&str>>()
        .iter()
        .enumerate()
    {
        let mut result = Vec::<f32>::new();
        // I shouldn't need to do this, but alas, UTF-8 makes things more complicated, and although
        // technically .obj files are ASCII only and I shouldn't need to worry about unicode, Rust only
        // offers unicode strings and I'm in no mood to keep casting back and forth.
        for value in line.split(&[' ', 't'][..]).collect::<Vec<&str>>() {
            match value {
                // Comment or blank line
                "#" | "" => {
                    continue 'outer;
                }
                // Face or vertex specifier
                "v" | "f" => {}
                // Other specifiers
                // ToDo: Probably wanna implement these later.
                "g" | "usemtl" | "mtllib" | "vt" | "vn" => {
                    panic!("No support for {} yet!", value);
                }
                // Actual numbers
                _ => {
                    result.push(
                        value.parse::<f32>().expect(
                            format!(
                                "Couldn't parse value: {} on line #{}: {} as a float",
                                value, i, line,
                            )
                            .as_str(),
                        ),
                    );
                }
            }
        }
        // Vertices are always made of 3 coordinates, faces are always triangles.
        assert!(result.len() == 3);

        if line.find('v') != None {
            vertices.push(Vec3 {
                x: result[0],
                y: result[1],
                z: result[2],
            });
        } else if line.find('f') != None {
            faces.push((result[0] as u32, result[1] as u32, result[2] as u32));
        } else {
            panic!("Line #{} is neither face nor vertex: {}", i, line);
        }
    }

    return (vertices, faces);
}
