import torch
from amsrr.simulation.placement_support import PlacementSupportEvidence


def task():
    return dict(goals=[dict(goal_type="object_pose", target_entity_id="object", target_pose_world=[0,0,.3,0,0,0,1], tolerance_pos_m=.05, tolerance_rot_rad=.2)],
        scene=dict(objects=[dict(object_id="object", geometry_id="object")],
        geometry_library=[dict(geometry_id="object",geometry_type="box",primitive_params=dict(size_m=[.6,.6,.2]),scale=[1,1,1]),
                          dict(geometry_id="support",geometry_type="box",primitive_params=dict(size_m=[.2,.2,.2]),scale=[1,1,1])],
        environment=dict(support_surfaces=[dict(surface_id="table",geometry_id="support",pose_world=[0,0,.1,0,0,0,1],contact_allowed=True,allowed_contact_modes=["support"])])))


def sample(e, z=.3, x=0., force=2., robot=0., phase=4, origin=0.):
    return e.sample(phase_index=torch.tensor([phase]),object_pose_world=torch.tensor([[x+origin,0.,z,0,0,0,1]]),
        scene_origins=torch.tensor([[origin,0.,0.]]),net_force_world=torch.tensor([[0.,0.,force]]),
        robot_contact_force_world=torch.tensor([[[0.,0.,robot]]])).item()


def test_finite_support_reaction_and_current_goal_are_all_required():
    e=PlacementSupportEvidence(task(),force_threshold_n=.5)
    assert sample(e)  # support fully inside object footprint, no object corner inside support
    assert sample(e, origin=10.)
    assert not sample(e, force=0.)
    assert not sample(e, force=2., robot=2.)
    assert not sample(e, z=.31)  # near goal but in air
    assert not sample(e, x=.5)
    assert not sample(e, phase=3)
    t=task(); t['scene']['environment']['support_surfaces'][0]['contact_allowed']=False
    assert not sample(PlacementSupportEvidence(t,force_threshold_n=.5))
    t=task(); t['scene']['environment']['support_surfaces'][0]['pose_world'][0]=1.
    assert not sample(PlacementSupportEvidence(t,force_threshold_n=.5))


def test_support_keeps_environments_separate_and_rejects_downward_reaction():
    e=PlacementSupportEvidence(task(),force_threshold_n=.5)
    assert not sample(e, force=-2.)
    result=e.sample(phase_index=torch.tensor([4,4]),
        object_pose_world=torch.tensor([[0.,0.,.3,0,0,0,1],[0.,0.,.31,0,0,0,1]]),
        scene_origins=torch.zeros(2,3),net_force_world=torch.tensor([[0.,0.,2.],[0.,0.,2.]]),
        robot_contact_force_world=torch.zeros(2,2,3))
    assert result.tolist()==[True,False]


def test_ambiguous_tracked_object_does_not_exempt_drop():
    t=task(); t['scene']['objects'].append(dict(object_id='other', geometry_id='object'))
    assert not sample(PlacementSupportEvidence(t,force_threshold_n=.5))
