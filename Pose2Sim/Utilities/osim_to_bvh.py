#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## OpenSim to BVH Exporter                      ##
    ##################################################
    
    Exports OpenSim .osim models and .mot animations to rigged animations in the BVH MoCap format.

    Heavily inspired by a script by Harri Kaimio:
    https://github.com/hkaimio/mocap-helpers/blob/main/opensim-to-bvh/opensim_to_bvh.py

    Usage:
        # Export rest pose only
        python osim_to_bvh.py --model model.osim -o

        # Export animation
        python osim_to_bvh.py --model model.osim --motion motion.mot

        # With optional parameters
        python osim_to_bvh.py --model model.osim --motion motion.mot -o output.bvh
'''

## INIT
import argparse
import sys
from pathlib import Path
import opensim as osim
import numpy as np
from bvhsdk import anim, bvh, mathutils


## AUTHORSHIP INFORMATION
__author__ = "Harri Kaimio, adapted by David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["Harri Kaimio", "David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"



## FUNCTIONS
def build_skeleton_tree(model):
    '''
    Build hierarchical skeleton structure from OpenSim model.

    INPUTS:
        model: OpenSim Model object

    OUPUTS:
        dict mapping body_name -> {
            'body': OpenSim Body object,
            'parent_body_name': str or None,
            'children': list of child body names,
            'joint': OpenSim Joint object connecting to parent
        }
    '''
    skeleton_tree = {}

    # Get all bodies
    body_set = model.getBodySet()

    # Build parent-child relationships from joints
    joint_set = model.getJointSet()

    # Initialize all bodies in tree
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        body_name = body.getName()
        skeleton_tree[body_name] = {
            'body': body,
            'parent_body_name': None,
            'children': [],
            'joint': None
        }

    # Populate parent-child relationships from joints
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        parent_body_name = joint.getParentFrame().findBaseFrame().getName()
        child_body_name = joint.getChildFrame().findBaseFrame().getName()

        # Set parent
        skeleton_tree[child_body_name]['parent_body_name'] = parent_body_name
        skeleton_tree[child_body_name]['joint'] = joint

        # Add to parent's children list
        if parent_body_name in skeleton_tree:
            skeleton_tree[parent_body_name]['children'].append(child_body_name)

    return skeleton_tree


def get_body_global_transform_matrix(model, state, body_name):
    '''
    Extract 4x4 global transform matrix for an OpenSim body.

    INPUTS:
        model: OpenSim Model object
        state: OpenSim State object
        body_name: Name of the body

    OUPUTS:
        4x4 numpy array [R | t; 0 0 0 1] where R is rotation, t is translation
    '''
    body = model.getBodySet().get(body_name)
    transform_osim = body.getTransformInGround(state)

    # Convert OpenSim::Transform to 4x4 matrix
    rotation = transform_osim.R()
    translation = transform_osim.T()

    matrix = np.eye(4)
    for i in range(3):
        for j in range(3):
            matrix[i, j] = rotation.get(i, j)
        matrix[i, 3] = translation.get(i)

    # Convert from meters to centimeters (BVH standard)
    matrix[0:3, 3] *= 100.0

    return matrix


def calculate_end_site_from_geometry(body, body_transform):
    '''
    Calculate end site position for a leaf node based on body's cylinder geometry.

    INPUTS:
        body: OpenSim Body object
        body_transform: 4x4 global transform matrix of the body

    OUPUTS:
        3D offset vector in body's local frame (in centimeters)
    '''
    # Try to get cylinder geometry
    try:
        geometry_list = body.get_attached_geometry()

        for i in range(geometry_list.getSize()):
            geometry = geometry_list.get(i)

            # Check if it's a cylinder
            if geometry.getConcreteClassName() == 'Cylinder':
                # Get cylinder properties
                cylinder = osim.Cylinder.safeDownCast(geometry)
                half_height = cylinder.get_half_height()

                # End site at cylinder tip (2 * half_height along Y axis in body frame)
                # Convert from meters to centimeters
                end_site = np.array([0.0, half_height * 2.0 * 100.0, 0.0])
                return end_site
    except:
        pass

    # Default: 10cm along Y axis if no geometry found
    return np.array([0.0, 10.0, 0.0])


def create_bvh_hierarchy(skeleton_tree, model, root_body_name, num_frames, framerate):
    '''
    Create bvhsdk Animation object with proper hierarchy.

    INPUTS:
        skeleton_tree: Skeleton structure from build_skeleton_tree()
        model: OpenSim Model object
        root_body_name: Name of the root body
        num_frames: Number of frames in animation
        framerate: Frames per second

    OUPUTS:
        anim.Animation object
    '''
    # Initialize state at default pose
    state = model.initSystem()
    model.realizePosition(state)

    # Create root joint recursively
    root_joint = create_joint_recursive(
        body_name=root_body_name,
        skeleton_tree=skeleton_tree,
        parent=None,
        depth=0,
        model=model,
        state=state
    )

    # Create animation
    animation = anim.Animation(filename="opensim_export", root=root_joint)
    animation.frames = num_frames
    animation.frametime = 1.0 / framerate

    return animation


def create_joint_recursive(body_name, skeleton_tree, parent, depth, model, state):
    '''
    Recursively create Joint objects for bvhsdk.

    INPUTS:
        body_name: Name of current body
        skeleton_tree: Skeleton structure
        parent: Parent Joint object (None for root)
        depth: Depth in hierarchy
        model: OpenSim Model object
        state: OpenSim State object (for getting transforms)

    OUPUTS:
        anim.Joints object
    '''
    node = skeleton_tree[body_name]

    # Create joint with OpenSim body name
    joint = anim.Joints(name=body_name, depth=depth, parent=parent)

    # Get global transform for this body
    body_transform = get_body_global_transform_matrix(model, state, body_name)
    body_global_pos = body_transform[0:3, 3]

    # Calculate offset in parent's local coordinate frame
    if parent is None:
        # Root offset = global position at rest pose
        offset = body_global_pos
    else:
        # Child offset = bone vector transformed to parent's local frame
        parent_body_name = node['parent_body_name']
        parent_transform = get_body_global_transform_matrix(model, state, parent_body_name)
        parent_global_pos = parent_transform[0:3, 3]
        parent_global_rot = parent_transform[0:3, 0:3]

        # Offset in global frame
        offset_global = body_global_pos - parent_global_pos

        # Transform to parent's local frame
        offset = parent_global_rot.T @ offset_global

    joint.addOffset(offset)

    # Set rotation order and channels
    joint.order = 'ZXY'  # BVH rotation order (supported by bvhsdk)
    joint.n_channels = 6 if parent is None else 3

    # Add end site for leaf nodes
    if len(node['children']) == 0:
        endsite = calculate_end_site_from_geometry(node['body'], body_transform)
        joint.addEndSite(endsite)

    # Recurse for children
    for child_body_name in node['children']:
        create_joint_recursive(child_body_name, skeleton_tree, joint, depth + 1, model, state)

    return joint


def set_frame_from_state(animation, model, state, frame_idx):
    '''
    Set animation frame data from OpenSim state.
    Used for both rest pose (frame 0) and animation frames.

    INPUTS:
        animation: bvhsdk Animation object
        model: OpenSim Model object
        state: OpenSim State object
        frame_idx: Frame index to set
    '''
    joints_list = animation.getlistofjoints()

    # Debug: Focus on specific joints
    debug_joints = ["spine", "shin.R"]
    debug_frames = [0, 1, 2]  # Only print first few frames

    for joint in joints_list:
        body_name = joint.name
        # Initialize arrays if needed (for first frame)
        if len(joint.rotation) == 0 or joint.rotation.shape[0] == 0:
            joint.rotation = np.zeros((animation.frames, 3))
        if joint.parent is None:
            if len(joint.translation) == 0 or joint.translation.shape[0] == 0:
                joint.translation = np.zeros((animation.frames, 3))

        # Get global transform from OpenSim
        global_transform = get_body_global_transform_matrix(model, state, body_name)
        global_rotation = global_transform[0:3, 0:3]
        global_position = global_transform[0:3, 3]

        # Debug output for specific joints
        if body_name in debug_joints and frame_idx in debug_frames:
            print(f"\n=== Frame {frame_idx}, Joint: {body_name} ===")

            # Print OpenSim generalized coordinates for this body's joint
            try:
                # Find the joint that has this body as child
                joint_set = model.getJointSet()
                for i in range(joint_set.getSize()):
                    joint_obj = joint_set.get(i)
                    if joint_obj.getChildFrame().findBaseFrame().getName() == body_name:
                        print(f"OpenSim Joint: {joint_obj.getName()}")
                        num_coords = joint_obj.numCoordinates()
                        coord_values = []
                        for j in range(num_coords):
                            coord = joint_obj.get_coordinates(j)
                            coord_name = coord.getName()
                            coord_value = coord.getValue(state)
                            coord_values.append(f"{coord_name}={coord_value:.4f}")
                        print(f"  Generalized coordinates: {', '.join(coord_values)}")

                        # Check for orientation offsets in the joint frames
                        try:
                            parent_frame = joint_obj.getParentFrame()
                            child_frame = joint_obj.getChildFrame()

                            # Get the transform from parent body to parent frame
                            parent_transform = parent_frame.findTransformInBaseFrame()
                            parent_rot_in_body = parent_transform.R()

                            # Get the transform from child body to child frame
                            child_transform = child_frame.findTransformInBaseFrame()
                            child_rot_in_body = child_transform.R()

                            print(f"  Parent frame orientation in parent body:")
                            parent_rot_matrix = np.eye(3)
                            for ii in range(3):
                                for jj in range(3):
                                    parent_rot_matrix[ii, jj] = parent_rot_in_body.get(ii, jj)
                            print(f"    {parent_rot_matrix}")

                            print(f"  Child frame orientation in child body:")
                            child_rot_matrix = np.eye(3)
                            for ii in range(3):
                                for jj in range(3):
                                    child_rot_matrix[ii, jj] = child_rot_in_body.get(ii, jj)
                            print(f"    {child_rot_matrix}")
                        except Exception as e:
                            print(f"    Could not get frame orientations: {e}")

                        break
            except Exception as e:
                print(f"  Could not get coordinates: {e}")

            print(f"Global transform from OpenSim:")
            print(global_transform)
            print(f"Global rotation matrix:")
            print(global_rotation)
            print(f"Global position: {global_position}")

        # Convert global rotation matrix to Euler ZXY (returns degrees)
        global_euler_deg, warning = mathutils.eulerFromMatrix(global_rotation, 'ZXY')

        # Debug: print global Euler before bvhsdk processes it
        if body_name in debug_joints and frame_idx in debug_frames:
            print(f"Global Euler (degrees): {global_euler_deg}")

        # Use bvhsdk's setGlobalRotation to automatically compute local rotation
        # setGlobalRotation expects degrees (bvhsdk operates in degrees)
        joint.setGlobalRotation(global_euler_deg, frame_idx)

        # Debug: print what bvhsdk computed as local rotation
        if body_name in debug_joints and frame_idx in debug_frames:
            print(f"Local rotation computed by bvhsdk (degrees): {joint.rotation[frame_idx]}")

            # Get bvhsdk's global rotation to verify
            if joint.parent:
                print(f"Parent joint: {joint.parent.name}")
            else:
                print(f"Root joint (no parent)")

            # Get what bvhsdk thinks the global rotation is
            # getGlobalRotation may return (euler, warning) tuple like eulerFromMatrix
            result = joint.getGlobalRotation(frame_idx)
            if isinstance(result, tuple):
                bvhsdk_global_euler_deg = result[0]
            else:
                bvhsdk_global_euler_deg = result

            print(f"Global Euler we passed to setGlobalRotation (degrees): {global_euler_deg}")
            print(f"Global Euler from bvhsdk's getGlobalRotation (degrees): {bvhsdk_global_euler_deg}")
            print(f"Difference in global Euler: {bvhsdk_global_euler_deg - global_euler_deg}")

        # For root: set global translation
        if joint.parent is None:
            joint.translation[frame_idx] = global_position - joint.offset
            if body_name in debug_joints and frame_idx in debug_frames:
                print(f"Root translation: {joint.translation[frame_idx]}")
                print(f"Root offset: {joint.offset}")


def populate_animation_from_motion(animation, model, motion_table,
                                   start_frame, end_frame):
    '''
    Populate animation frames from OpenSim motion data.

    INPUTS:
        animation: bvhsdk Animation object
        model: OpenSim Model object
        motion_table: OpenSim TimeSeriesTable with motion data
        start_frame: First frame to export
        end_frame: Last frame to export (inclusive)
    '''
    # Initialize state ONCE before the loop (matching YAML script approach)
    state = model.initSystem()
    coordinate_set = model.getCoordinateSet()

    # Get column labels (coordinate names)
    coordinate_names = motion_table.getColumnLabels()
    motion_data_np = motion_table.getMatrix().to_numpy()
    for i, c in enumerate(coordinate_names):
        try:
            if coordinate_set.get(c).getMotionType() == 1:  # 1: rotation
                if motion_table.getTableMetaDataAsString('inDegrees') == 'yes':
                    motion_data_np[:, i] = motion_data_np[:, i] * np.pi / 180
        except Exception:
            pass

    for frame_idx in range(start_frame, end_frame + 1):
        # Reset coordinate counter for this frame
        coords_changed = 0
        # Python tuple - iterate directly
        for i, coord_name in enumerate(coordinate_names):
            try:
                coord = coordinate_set.get(coord_name)
                value = motion_data_np[frame_idx, i]
                coord.setValue(state, value, enforceConstraints=False)
                coords_changed += 1
            except Exception as e:
                # Coordinate might not exist in model
                if frame_idx == start_frame:
                    print(f"    Warning: Could not set {coord_name}: {e}")
                pass

        if frame_idx == start_frame:
            print(f"  Set {coords_changed} coordinates for frame {frame_idx}")

        # Realize position to update body transforms
        model.realizePosition(state)

        # Set frame data (frame_idx - start_frame + 1 because frame 0 is rest pose)
        output_frame_idx = frame_idx - start_frame + 1
        set_frame_from_state(animation, model, state, output_frame_idx)

        if frame_idx % 100 == 0:
            print(f"  Processed frame {frame_idx - start_frame + 1}/{end_frame - start_frame + 1}")


def export_to_bvh(model_path, output_path, motion_path=None, framerate=30.0,
                  start_frame=0, end_frame=None, root_body='pelvis'):
    '''
    Main export function.

    INPUTS:
        model_path: Path to OpenSim model file
        output_path: Path to output BVH file
        motion_path: Path to motion file (optional)
        framerate: Output framerate
        start_frame: First frame to export
        end_frame: Last frame to export (None = all)
        root_body: Name of root body
    '''
    print(f"Loading OpenSim model: {model_path}")
    model = osim.Model(str(model_path))

    print("Building skeleton tree...")
    skeleton_tree = build_skeleton_tree(model)
    print(f"  Found {len(skeleton_tree)} bodies")

    # Determine number of frames
    if motion_path:
        print(f"Loading motion data: {motion_path}")
        motion_table = osim.TimeSeriesTable(str(motion_path))
        
        times = motion_table.getIndependentColumn()
        framerate = int((len(times) - 1) / (times[-1] - times[0]))

        total_frames = motion_table.getNumRows()
        if end_frame is None:
            end_frame = total_frames - 1
        else:
            end_frame = min(end_frame, total_frames - 1)
        num_output_frames = end_frame - start_frame + 2  # +1 for rest pose at frame 0
        print(f"  Exporting frames {start_frame} to {end_frame}")

    else:
        print("No motion data - exporting rest pose only")
        num_output_frames = 1
        motion_table = None

    print("Creating BVH hierarchy...")
    animation = create_bvh_hierarchy(skeleton_tree, model, root_body, num_output_frames, framerate)

    print("Setting rest pose (frame 0)...")
    state = model.initSystem()
    model.realizePosition(state)
    set_frame_from_state(animation, model, state, frame_idx=0)

    if motion_table:
        print(f"Populating animation frames...")
        populate_animation_from_motion(animation, model, motion_table, start_frame, end_frame)

    bvh.WriteBVH(
        animation=animation,
        path=str(output_path.parent),
        name=output_path.stem,
        frametime=1.0 / framerate,
        writeTranslation=False,  # Only root has translation
        refTPose=True,           # First frame is reference pose
        precision=6
    )

    print("✓ Export complete!")


def parse_arguments():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Export OpenSim model and animation to BVH format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--model', required=True, type=Path,
                        help='OpenSim model file (.osim)')
    parser.add_argument('-o', '--output', required=False, type=Path,
                        help='Output BVH file path')
    parser.add_argument('--motion', type=Path,
                        help='Motion file (.mot, .sto) - if omitted, exports rest pose only')

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")
    if args.motion and not args.motion.exists():
        parser.error(f"Motion file not found: {args.motion}")
    if not args.output:
        if not args.motion:
            args.output = args.model.with_suffix('.bvh')
        else:
            args.output = args.motion.with_suffix('.bvh')

    return args


def main():
    '''Main entry point.'''
    args = parse_arguments()

    try:
        export_to_bvh(
            model_path=args.model,
            output_path=args.output,
            motion_path=args.motion
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
