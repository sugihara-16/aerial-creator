"""Object-condition expansion with causal observations and no teacher-label reuse."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from amsrr.geometry.mass_properties import RigidBodyMassProperties
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler, ContactCandidateSamplerConfig
from amsrr.policies.high_level_requests import RequestCatalogBuilder
from amsrr.schemas.high_level import ActiveExecutionState
from amsrr.training.request_imitation import decision_context, teacher_request_scene
from amsrr.training.order9_c3_teacher import _offset_horizontal_grasp_candidates
from amsrr.simulation.task_mass_properties import inertia_matrix


def ballasted_box_mass_properties(size, mass, com):
    """Uniform 80% shell volume plus a 20% internal cuboid (0.2 side lengths).

    The shell here is a filled uniform box; the ballast adds positive density
    inside it. The requested COM is produced by moving the ballast's center.
    """
    size, com = np.asarray(size, dtype=float), np.asarray(com, dtype=float)
    if size.shape != (3,) or com.shape != (3,) or np.any(size <= 0) or mass <= 0 or not np.isfinite([*size, mass, *com]).all():
        raise ValueError('invalid box mass properties')
    if np.any(np.abs(com) > .05 * size + 1e-12):
        raise ValueError('requested COM outside ballasted-box distribution')
    def box_inertia(m, s):
        return np.diag(m / 12 * (np.dot(s, s) - s * s))
    inertia = np.zeros((3, 3))
    for fraction, dimensions, center in ((.8, size, np.zeros(3)), (.2, .2*size, com/.2)):
        displacement = center - com
        inertia += box_inertia(mass*fraction, dimensions) + mass*fraction*(np.dot(displacement, displacement)*np.eye(3)-np.outer(displacement, displacement))
    volume = float(np.prod(size))
    result = RigidBodyMassProperties(float(mass), com.tolist(), inertia[np.triu_indices(3)].tolist(), volume, mass/volume, 'positive_density_internal_ballast_box_v1')
    result.validate()
    return result


def _preserve_bottom(pose, old_size, new_size):
    result = list(pose)
    vertical = np.abs(Rotation.from_quat(result[3:]).as_matrix()[2])
    result[2] += float(vertical @ (np.asarray(new_size)-old_size)/2)
    return result


def transform_initial_robot(observation, transform):
    """Apply an authored planar rigid transform to every robot link state.

    Object/support/goal states stay in their authored world frame. This is
    training-scene randomization, never an action chosen by the policy.
    """
    yaw = float(transform['yaw_rad'])
    translation = np.asarray(transform['translation_world_m'], dtype=float)
    if (translation.shape != (3,) or not np.isfinite([yaw, *translation]).all()
            or abs(float(translation[2])) > 1e-12):
        raise ValueError('initial robot transform must be finite and planar')
    rotation = Rotation.from_euler('z', yaw)
    for state in observation.module_states:
        state.pose_world = [*(rotation.apply(state.pose_world[:3]) + translation),
                            *(rotation * Rotation.from_quat(state.pose_world[3:])).as_quat()]
        state.twist_world = [*rotation.apply(state.twist_world[:3]),
                             *rotation.apply(state.twist_world[3:])]
    return observation


def _load_bound_morphology(binding, physical):
    """Validate one upstream design against the robot that physics will load."""
    import json
    from pathlib import Path
    from amsrr.schemas.morphology import MorphologyGraph
    from amsrr.utils.hashing import hash_file
    if hash_file(binding['path']) != binding['sha256']:
        raise ValueError('designated morphology identity changed')
    replacement = MorphologyGraph.from_dict(json.loads(Path(binding['path']).read_text()))
    # A new planning graph must also change the physical robot. Reject
    # incomplete bindings before planning, rather than run the donor USD.
    from amsrr.simulation.order9_morphology_assets import (
        load_order9_morphology_asset_manifest,
        validate_order9_morphology_asset_manifest_bytes)
    asset_binding = binding.get('asset_manifest')
    if not isinstance(asset_binding, dict):
        raise ValueError('designated morphology requires a bound physical asset manifest')
    if hash_file(asset_binding['path']) != asset_binding['sha256']:
        raise ValueError('designated physical asset manifest identity changed')
    assets = load_order9_morphology_asset_manifest(asset_binding['path'])
    if assets.physical_model_hash != physical.stable_hash():
        raise ValueError('designated physical model identity differs')
    validate_order9_morphology_asset_manifest_bytes(
        assets, repository_root=Path(__file__).resolve().parents[2])
    asset = assets.entry_for(replacement)
    if asset.source_morphology_hash != replacement.stable_hash():
        raise ValueError('designated planning and physical morphologies differ')
    return replacement, asset


def expand_box_episode(episode, condition, physical):
    """Keep robot/support/goal criteria; regenerate object geometry and catalog."""
    original = decision_context(episode, 0, physical)
    task = deepcopy(episode['task'])
    obj = task.scene.objects[0]
    geometry = next(g for g in task.scene.geometry_library if g.geometry_id == obj.geometry_id)
    old_size = np.asarray(geometry.primitive_params['size_m'])
    old_mass, old_inertia = obj.mass_kg, inertia_matrix(obj.inertia_kgm2)
    old_com = np.asarray(obj.center_of_mass_object or [0, 0, 0])
    if 'size_m' in condition:
        size = np.asarray(condition['size_m'], dtype=float)
    else:
        volume = float(np.prod(old_size)) * condition['volume_factor']
        xy, zy = condition['x_over_y'], condition['z_over_y']
        y = (volume/(xy*zy))**(1/3)
        size = np.array([xy*y, y, zy*y])
    mass = old_mass * condition['mass_factor']
    com = size * np.asarray(condition['com_fraction'])
    properties = ballasted_box_mass_properties(size, mass, com)
    geometry.primitive_params['size_m'] = size.tolist()
    obj.mass_kg, obj.inertia_kgm2 = mass, properties.inertia_kgm2
    obj.center_of_mass_object, obj.density_kg_m3 = properties.center_of_mass_object, properties.density_kg_m3
    obj.pose_world = _preserve_bottom(obj.pose_world, old_size, size)
    for goal in task.goals:
        if goal.target_entity_id == obj.object_id and goal.target_pose_world is not None:
            goal.target_pose_world = _preserve_bottom(goal.target_pose_world, old_size, size)
    case = deepcopy(episode['case'])
    case['estimated_mass_kg'] *= mass/old_mass
    # Preserve the donor relative inertia estimation error by congruence.
    values, vectors = np.linalg.eigh(old_inertia)
    old_inv_half = (vectors/np.sqrt(values)) @ vectors.T
    new_values, new_vectors = np.linalg.eigh(inertia_matrix(obj.inertia_kgm2))
    new_half = (new_vectors*np.sqrt(new_values)) @ new_vectors.T
    transform = new_half @ old_inv_half
    estimate = transform @ inertia_matrix(case['estimated_inertia_body']) @ transform.T
    case['estimated_inertia_body'] = estimate[np.triu_indices(3)].tolist()
    case['estimated_com_object'] = (com + np.asarray(case['estimated_com_object']) - old_com).tolist()
    for key in ('estimated_mass_kg','estimated_inertia_body','estimated_com_object'):
        task.metadata[key] = deepcopy(case[key])
    task.metadata['object_condition'] = dict(condition)
    if 'order9_c3_articulated_teacher_precheck' in task.metadata:
        task.metadata['donor_teacher_precheck_history'] = task.metadata.pop('order9_c3_articulated_teacher_precheck')
    task.validate()
    observation = deepcopy(original.scene.runtime_observation)
    state = next(o for o in observation.object_states if o.object_id == obj.object_id)
    state.pose_world = _preserve_bottom(state.pose_world, old_size, size)
    # Object initial twist is a measured link-origin state, preserved with geometry.
    built = IRGBuilder().build_with_scene_graph(task)
    morphology = episode['bundle'].morphology
    if 'designated_morphology' in condition:
        # Upstream design input, fixed before the actor's first decision. The
        # donor supplies task/material distributions, never a target posture.
        from amsrr.robot_model.whole_structure_kinematics import (
            WholeStructureKinematics, ordered_global_dock_joint_ids)
        replacement, asset = _load_bound_morphology(condition['designated_morphology'], physical)
        case['morphology_graph'] = dict(path=asset.morphology_graph_path,
                                      sha256=asset.morphology_graph_sha256)
        case['robot_usd'] = dict(path=asset.usd_path, sha256=asset.usd_sha256)
        case['structural_hash'] = asset.structural_hash
        case['morphology_hash'] = asset.source_morphology_hash
        if ({(m.module_id, m.module_type) for m in replacement.modules}
                != {(m.module_id, m.module_type) for m in morphology.modules}
                or replacement.base_module_id != morphology.base_module_id):
            raise ValueError('design augmentation requires the same module inventory and base')
        if 'designated_grasp_port_ids' not in condition:
            raise ValueError('new upstream morphology requires designated grasp ports')
        replacement.robot_anchors = deepcopy([a for a in morphology.robot_anchors if a.anchor_type != 'grasp'])
        replacement.validate()
        morphology = replacement
        base = next(s.pose_world for s in observation.module_states
                    if s.module_id == morphology.base_module_id)
        q = dict.fromkeys(ordered_global_dock_joint_ids(morphology, physical), 0.)
        fk = WholeStructureKinematics().forward(morphology, physical, q, base, [])
        for state in observation.module_states:
            state.pose_world = fk.module_root_poses_world[state.module_id]
            state.twist_world = [0.] * 6
            for name in state.joint_positions:
                if f'module_{state.module_id}:{name}' in q:
                    state.joint_positions[name] = 0.
            state.joint_velocities = dict.fromkeys(state.joint_velocities, 0.)
        # This authored zero-velocity initial state is also the reset state;
        # later observations always come from the ordinary simulator sensors.
        observation.morphology_graph = morphology
    if 'designated_grasp_port_ids' in condition:
        from amsrr.robot_model.gripper_surfaces import with_designated_grasp_anchors
        from amsrr.schemas.irg import IRGNodeType
        slots = [int(n.feature['slot_id']) for n in built.irg.nodes
                 if n.node_type == IRGNodeType.CONTACT_SLOT and n.feature.get('contact_mode') == 'grasp']
        morphology = with_designated_grasp_anchors(morphology, physical,
            condition['designated_grasp_port_ids'], slot_ids=slots,
            max_force_n=task.safety.max_contact_force_n,
            max_torque_nm=task.safety.max_contact_torque_nm)
        observation.morphology_graph = morphology
    if 'initial_robot_transform' in condition:
        observation = transform_initial_robot(observation, condition['initial_robot_transform'])
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    candidates = ContactCandidateSampler().sample(task_spec=task, irg=built.irg, interaction_envelope=envelope, morphology_graph=morphology, geometry_descriptors=built.scene_graph.geometry_descriptors)
    offsets = {c.candidate_scores['c3_grasp_contact_height_offset_m']
               for c in episode['initial_scene'].contact_candidate_set.candidates
               if c.candidate_scores.get('c3_grasp_contact_height_offset_m') is not None}
    if len(offsets) != 1:
        raise ValueError('donor candidate height convention differs')
    candidates = _offset_horizontal_grasp_candidates(candidates, height_offset_m=offsets.pop(), tangent_offset_world_m=(0.,0.,0.))
    bundle = replace(deepcopy(episode['bundle']), morphology=morphology, contact_candidate_set=candidates)
    scene, phases = teacher_request_scene(bundle, task, plan_committed=False)
    scene = replace(scene, runtime_observation=observation)
    context = RequestCatalogBuilder().build(scene, task_spec=task, physical_model=physical, execution_state=ActiveExecutionState(phases['approach'], observation.time_s))
    expanded = dict(episode, task=task, case=case, bundle=bundle, initial_scene=scene, scene=scene, phase_ids=phases,
                    selected_group=SimpleNamespace(group_id='no_teacher_for_new_object_condition'))
    return expanded, context


def grasp_contact_expansion_limit(protocol):
    """Separate anchor ownership from the number of contacts in a group.

    Historical protocols omitted the source and expanded all free surfaces.
    New design-anchor runs explicitly retain the upstream anchor definitions.
    """
    source = protocol.get('grasp_anchor_source', 'all_free')
    if source not in ('defined', 'all_free', 'neighborhood'):
        raise ValueError('unknown grasp_anchor_source: ' + str(source))
    if source == 'neighborhood' and protocol.get('max_grasp_contacts', 2) != 2:
        raise ValueError('anchor neighborhoods require max_grasp_contacts=2')
    return None if source == 'defined' else protocol.get('max_grasp_contacts', 2)


def expand_episode_grasp_contacts(episode, physical, *, max_grasp_contacts=3, grasp_anchor_source='all_free'):
    """Fresh, causal catalog over every free physical grasp surface.

    None retains the upstream anchors and the already expanded initial scene.
    An integer explicitly expands all free surfaces, with that contact limit.
    Neither the donor trajectory nor its selected group supplies new labels.
    """
    if max_grasp_contacts is None:
        scene = episode['initial_scene']
        observation = scene.runtime_observation
        if observation is None:
            return episode, decision_context(episode, 0, physical)
        context = RequestCatalogBuilder().build(scene, task_spec=episode['task'],
            physical_model=physical, execution_state=ActiveExecutionState(
                episode['phase_ids']['approach'], observation.time_s))
        return episode, context
    from amsrr.robot_model.gripper_surfaces import with_available_grasp_anchors, with_neighbor_grasp_anchors
    if grasp_anchor_source not in ('all_free', 'neighborhood'):
        raise ValueError('invalid expanded anchor source: ' + grasp_anchor_source)
    neighborhood = grasp_anchor_source == 'neighborhood'
    if neighborhood and max_grasp_contacts != 2:
        raise ValueError('anchor neighborhoods require two contacts')
    from amsrr.schemas.irg import IRGNodeType
    # A preceding object-condition expansion has already transformed the
    # measured initial state. Re-reading donor tensors would undo that change.
    initial_observation = episode['initial_scene'].runtime_observation
    if initial_observation is None:
        initial_observation = decision_context(episode, 0, physical).observation
    task = deepcopy(episode['task'])
    built = IRGBuilder().build_with_scene_graph(task)
    slots = [int(n.feature['slot_id']) for n in built.irg.nodes
             if n.node_type == IRGNodeType.CONTACT_SLOT and n.feature.get('contact_mode') == 'grasp']
    expand_anchors = with_neighbor_grasp_anchors if neighborhood else with_available_grasp_anchors
    morphology = expand_anchors(episode['bundle'].morphology, physical,
        slot_ids=slots, max_force_n=task.safety.max_contact_force_n,
        max_torque_nm=task.safety.max_contact_torque_nm)
    observation = deepcopy(initial_observation)
    observation.morphology_graph = morphology
    from amsrr.policies.contact_group_geometry import observed_anchor_poses
    anchor_poses = observed_anchor_poses(SimpleNamespace(physical_model=physical,
        scene=SimpleNamespace(morphology_graph=morphology), observation=observation))
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    candidates = ContactCandidateSampler(ContactCandidateSamplerConfig(
        max_grasp_contacts=max_grasp_contacts, max_group_proposals_per_slot=8 if neighborhood else 32,
        grasp_neighborhoods=neighborhood)).sample(task_spec=task, irg=built.irg,
        interaction_envelope=envelope, morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
        anchor_poses_world=None if neighborhood else anchor_poses)
    offsets = {c.candidate_scores['c3_grasp_contact_height_offset_m']
               for c in episode['initial_scene'].contact_candidate_set.candidates
               if c.candidate_scores.get('c3_grasp_contact_height_offset_m') is not None}
    if len(offsets) > 1:
        raise ValueError('inconsistent contact-height convention')
    if offsets:
        candidates = _offset_horizontal_grasp_candidates(candidates,
            height_offset_m=offsets.pop(), tangent_offset_world_m=(0.,0.,0.))
    bundle = replace(deepcopy(episode['bundle']), morphology=morphology, contact_candidate_set=candidates)
    scene, phases = teacher_request_scene(bundle, task, plan_committed=False)
    scene = replace(scene, runtime_observation=observation)
    context = RequestCatalogBuilder().build(scene, task_spec=task, physical_model=physical,
        execution_state=ActiveExecutionState(phases['approach'], scene.runtime_observation.time_s))
    expanded = dict(episode, bundle=bundle, task=task, initial_scene=scene, scene=scene,
        phase_ids=phases, selected_group=SimpleNamespace(group_id='no_teacher_for_expanded_contact_catalog'))
    return expanded, context
