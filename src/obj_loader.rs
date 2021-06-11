use crate::math::Vec3;

// Note: God this is an ugly function. Wanna nuke it at some point and do it properly.
// Returns pair of vectors for vertices and indices
pub fn read_obj_file(file_path: &str) -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>) {
    let file_content = std::fs::read_to_string(file_path).unwrap();

    let (mut vertices, mut normals, mut indices) =
        (Vec::<Vec3>::new(), Vec::<Vec3>::new(), Vec::<u32>::new());
    let mut vertex_to_normal_map = std::collections::HashMap::<u32, u32>::new();

    for (i, line) in file_content
        .split(&['\r', '\n'][..])
        .filter(|t| {
            return *t != "";
        })
        .collect::<Vec<&str>>()
        .iter()
        .enumerate()
    {
        let mut result = Vec::<f32>::new();
        let byte_array = line.as_bytes();
        let current_char = byte_array[0] as char;
        if current_char == '#' || current_char == 's' {
            continue;
        }

        let tokens = line.split(&[' ', 't'][..]).skip(1).collect::<Vec<&str>>();

        let parse_as_float = |token: &str| -> f32 {
            return token.parse::<f32>().expect(
                format!(
                    "Couldn't parse value: {} on line #{}: {} as a float",
                    token, i, line,
                )
                .as_str(),
            );
        };

        match current_char {
            'v' => {
                for token in tokens {
                    result.push(parse_as_float(token));
                }
                assert!(result.len() == 3, "Result's size is {}!", result.len());
                let result = Vec3 {
                    x: result[0],
                    y: result[1],
                    z: result[2],
                };

                match byte_array[1] as char {
                    ' ' => vertices.push(result),
                    'n' => normals.push(result),
                    _ => panic!("No support for {} yet!", line),
                }
            }
            'f' => {
                for token in tokens {
                    for token in token
                        .split('/')
                        .filter(|t| {
                            return *t != "" && *t != "";
                        })
                        .collect::<Vec<&str>>()
                    {
                        result.push(parse_as_float(token));
                    }
                }
                assert!(
                    result.len() % 3 == 0,
                    "Result's size {} is not divisible by 3!",
                    result.len()
                );

                indices.push(result[0] as u32 - 1);
                indices.push(result[1 * result.len() / 3] as u32 - 1);
                indices.push(result[2 * result.len() / 3] as u32 - 1);

                if result.len() == 6 {
                    vertex_to_normal_map.insert(result[0] as u32 - 1, result[1] as u32 - 1);
                    vertex_to_normal_map.insert(result[2] as u32 - 1, result[3] as u32 - 1);
                    vertex_to_normal_map.insert(result[4] as u32 - 1, result[5] as u32 - 1);
                } else {
                    panic!("Incorrect number of entires: {}\n", result.len());
                }
            }
            _ => panic!("No support for {} yet!", line),
        }
    }

    let mut temp_normals = normals.clone();
    if vertex_to_normal_map.len() != 0 {
        for (k, v) in vertex_to_normal_map {
            temp_normals[k as usize] = normals[v as usize];
        }
    }

    normals = temp_normals;

    assert!(
        vertices.len() == normals.len(),
        "Vertices: {}, Normals: {}\n",
        vertices.len(),
        normals.len()
    );

    return (vertices, normals, indices);
}
