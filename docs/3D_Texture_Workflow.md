# 3D Texture Generation Workflow

This workflow demonstrates how to use the new 3D texture generation nodes in MaraScott Nodes.

## Required Nodes

### 1. 🐰 3D Mesh Loader (TriMeshLoader_v1)
- **Purpose**: Load 3D mesh files (OBJ, PLY, STL, GLTF, etc.)
- **Inputs**:
  - `mesh_file`: Select from available mesh files in ComfyUI input directory
  - `scale`: Scale factor for the mesh (default: 1.0)
  - `center_mesh`: Center the mesh at origin (default: True)
  - `fix_normals`: Fix mesh normals (default: True)
  - `merge_vertices`: Merge duplicate vertices (default: True)
- **Outputs**:
  - `trimesh`: Loaded 3D mesh object

### 2. 🐰 3D Camera Config (CameraConfig3D_v1)
- **Purpose**: Create camera configuration for multiview rendering
- **Inputs**:
  - `distance`: Camera distance from object (default: 2.0)
  - `elevation`: Camera elevation angle in degrees (default: 30.0)
  - `field_of_view`: Camera field of view in degrees (default: 60.0)
  - `camera_type`: "perspective" or "orthographic" (default: "perspective")
  - `up_vector`: Camera up vector as "X,Y,Z" (default: "0,0,1")
  - `look_at`: Camera look-at point as "X,Y,Z" (default: "0,0,0")
- **Outputs**:
  - `camera_config`: Camera configuration dictionary

### 3. 🐰 3D Texture Generator (TextureGenerator3D_v1)
- **Purpose**: Generate various texture maps from 3D mesh and camera setup
- **Inputs**:
  - `trimesh`: 3D mesh object from TriMeshLoader
  - `camera_config`: Camera configuration from CameraConfig3D
  - `texture_size`: Size of generated textures (default: 1024)
  - `multiview_count`: Number of views to generate (default: 8)
  - `lighting_mode`: "uniform", "directional", "point", "ambient" (default: "uniform")
  - `normal_space`: "object", "world", "tangent" (default: "world")
  - `use_antialiasing`: Enable antialiasing (default: True)
  - `background_color`: Background color as "R,G,B,A" (default: "0.0,0.0,0.0,0.0")
- **Outputs**:
  - `normal_maps`: Normal map textures for lighting calculations
  - `position_maps`: Position map textures for spatial information
  - `exr_maps`: HDR texture maps for lighting
  - `renderer`: Configured renderer object for further use
  - `masks`: Alpha/transparency masks for compositing

## Basic Workflow Setup

```
[3D Mesh Loader] → [trimesh] → [3D Texture Generator] → [normal_maps, position_maps, exr_maps, masks]
                                         ↑
[3D Camera Config] → [camera_config] ----+
```

## Example Configuration

1. **Load a 3D Model**:
   - Place your .obj, .ply, .stl, or .gltf files in ComfyUI's input directory
   - Use TriMeshLoader_v1 to load the mesh
   - Set appropriate scale and processing options

2. **Configure Camera**:
   - Use CameraConfig3D_v1 to set camera parameters
   - Adjust distance and elevation for best view
   - Choose perspective or orthographic projection

3. **Generate Textures**:
   - Connect mesh and camera config to TextureGenerator3D_v1
   - Set texture_size based on your needs (512, 1024, 2048, etc.)
   - Choose multiview_count (4-16 views recommended)
   - Select appropriate lighting mode for your use case

## Output Usage

- **Normal Maps**: Use for realistic lighting and surface detail in 3D rendering
- **Position Maps**: Useful for spatial effects and advanced shading
- **EXR Maps**: High dynamic range lighting information
- **Masks**: Alpha channels for compositing and transparency effects

## Dependencies

The following Python packages are required:
- `trimesh[easy]`: 3D mesh processing
- `pyrender`: 3D rendering engine
- `moderngl`: OpenGL context management

Install with:
```bash
pip install trimesh[easy] pyrender moderngl
```

## Tips

1. **Mesh Quality**: Clean meshes with proper normals work best
2. **Texture Size**: Higher resolutions provide more detail but take longer to process
3. **View Count**: More views provide better coverage but increase processing time
4. **Lighting**: Choose lighting mode based on your final use case
5. **Camera Distance**: Adjust to ensure entire mesh is visible in all views

## Troubleshooting

- **Import Errors**: Install required dependencies
- **Mesh Not Loading**: Check file format and path
- **Poor Quality**: Increase texture_size and ensure good mesh normals
- **Slow Performance**: Reduce texture_size and multiview_count for testing
