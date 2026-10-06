import numpy as np,pytest
from scipy.spatial.transform import Rotation
from amsrr.training.soft_contact_preload import patch_normal_forces,nominal_patch_radius
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from pathlib import Path

def solve(offset=0,rotation=None,translation=None):
 p=np.array([[-.1,0,0],[.1,0,0]]);n=np.array([[1.,0,0],[-1.,0,0]]);c=np.array([0,offset,0]);f=np.array([0.,0.,10.])
 if rotation is not None:p=rotation.apply(p);n=rotation.apply(n);c=rotation.apply(c);f=rotation.apply(f)
 if translation is not None:p+=translation;c+=translation
 return patch_normal_forces(positions=p,normals=n,center=c,required_force=f,friction=[1,1],radii=[.015,.015],normal_jacobian=np.eye(2),effort_limits=[10,10],minimum_normal_force=1)

@pytest.mark.parametrize('offset',[0,.01,.02,-.02])
def test_analytic_shear_torsion_ellipse(offset):
 force,e=solve(offset);np.testing.assert_allclose(force,[5*np.hypot(1,offset/.015)]*2,atol=1e-7)
 assert e['equilibrium_residual']<1e-8
 assert abs(sum(abs(x) for x in e['contact_torsional_moments_nm'])-abs(offset*10))<1e-7

def test_rigid_transform_does_not_change_load():
 a,_=solve(.02);b,_=solve(.02,Rotation.from_euler('xyz',[.5,-.2,1.]),np.array([1.,-2.,3.]));np.testing.assert_allclose(a,b,atol=1e-7)

def test_current_authored_pad_transform():
 p=build_physical_model_from_config(Path('configs/robot/robot_model.yaml'))
 r=nominal_patch_radius(p.urdf_path,'yaw_dock_mech2',(.115,0,0,0,0,0,1),p.metadata['urdf_hash'])
 assert .014<r<.015

def test_arbitrary_contact_count():
 angle=np.arange(3)*2*np.pi/3;p=np.c_[.1*np.cos(angle),.1*np.sin(angle),np.zeros(3)];n=-p/.1
 f,e=patch_normal_forces(positions=p,normals=n,center=[0,0,0],required_force=[0,0,9],friction=[1]*3,radii=[.015]*3,normal_jacobian=np.eye(3),effort_limits=[10]*3,minimum_normal_force=1)
 np.testing.assert_allclose(f,[3]*3,atol=1e-6);assert e['equilibrium_residual']<1e-8

def test_bad_geometry_rejected():
 with pytest.raises(ValueError):patch_normal_forces(positions=[[0,0,0]]*2,normals=[[0,0,0]]*2,center=[0,0,0],required_force=[0,0,10],friction=[1,1],radii=[.015,.015],normal_jacobian=np.eye(2),effort_limits=[10,10],minimum_normal_force=1)

@pytest.mark.parametrize('peak,lead',[(10.,.04),(2.,.04),(10.,.012)])
def test_extra_reserve_saturates_without_weakening_limits(peak,lead):
 from amsrr.training.soft_contact_preload import bounded_normal_reserve
 base=np.array([1.,1.]);force,wanted,applied=bounded_normal_reserve(forces=base,requested=3*base,jacobian=np.eye(2),stiffness=np.array([100.,100.]),peak=np.array([peak,peak]),contact_stiffness=1000.,maximum_utilization=.8,maximum_lead=lead,quantum=.001,margin=0)
 assert wanted==3;assert 1<=applied<=3
 assert np.max(force/peak)<=.8+1e-9
 assert np.max(force*.011)<=lead+1e-9
 np.testing.assert_allclose(force/base,[applied]*2)

def test_nonfinite_solver_result_fails_closed(monkeypatch):
 from types import SimpleNamespace
 import amsrr.training.soft_contact_preload as module
 monkeypatch.setattr(module,'minimize',lambda *a,**k:SimpleNamespace(success=True,x=np.full(9,np.nan),message='invalid',nit=1))
 with pytest.raises(ValueError,match='equilibrium failed'):solve(.01)

def test_additional_preload_never_reduces_legacy_pair_command():
 from amsrr.training.order9_actuator_aware_nominal_preload import solve_order9_actuator_aware_nominal_preload_from_jacobian as preload
 args=dict(inward_normal_joint_jacobian_m=np.array([[1.,0],[0,.3]]),joint_ids=['a','b'],joint_stiffness_nm_per_rad=[200.,200.],joint_peak_effort_limit_nm=[20.,20.],joint_continuous_effort_limit_nm=[10.,10.],object_mass_kg=.8,contact_friction=1.,contact_stiffness_n_per_m=1000.,contact_positions_world=np.array([[-.1,0,0],[.1,0,0]]),inward_normals_world=np.array([[1.,0,0],[-1.,0,0]]),object_com_world=np.array([0,.015,0]))
 baseline=preload(**args);extra=preload(**args,contact_torsion_radii_m=[.015,.015])
 assert baseline.feasible and extra.feasible
 assert baseline.commanded_inward_lead_m_by_anchor==pytest.approx([baseline.inward_lead_m]*2)
 assert min(extra.commanded_inward_lead_m_by_anchor)>=baseline.inward_lead_m
 assert extra.commanded_inward_lead_m_by_anchor[0]>extra.commanded_inward_lead_m_by_anchor[1]
