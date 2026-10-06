"""Nominal finite-patch friction model; never an exact contact-force estimate."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, LinearConstraint

def nominal_patch_radius(urdf_path, link_id, anchor_pose, urdf_hash):
    """Uniform-pressure mean radius on the authored front contact hull.

    URDF collision transforms are applied before expressing vertices in the
    anchor frame. This is a nominal pressure assumption, not observed contact.
    """
    from amsrr.feasibility.order9_posture_collision import _mesh_geometry, _mesh_path
    from amsrr.utils.hashing import hash_file
    path=Path(urdf_path)
    if hash_file(path) != urdf_hash:
        raise ValueError("contact geometry URDF changed after physical model creation")
    link=ET.parse(path).getroot().find(f"link[@name='{link_id}']")
    if link is None:raise ValueError('missing contact link')
    points=[]
    for collision in link.findall('collision'):
        mesh=collision.find('geometry/mesh')
        if mesh is None:raise ValueError('nominal patch requires authored mesh')
        origin=collision.find('origin');attrs={} if origin is None else origin.attrib
        xyz=np.fromstring(attrs.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(attrs.get('rpy','0 0 0'),sep=' ')
        scale=np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ')
        source=_mesh_path(mesh.attrib['filename'],urdf_path=path,search_directories=(Path(__file__).resolve().parents[2]/'module_urdf/mesh',))
        _,geometry=_mesh_geometry(source,scale)
        vertices=Rotation.from_euler('xyz',rpy).apply(np.array(geometry[2], copy=True))+xyz
        points.append(Rotation.from_quat(anchor_pose[3:]).inv().apply(vertices-np.array(anchor_pose[:3])))
    points=np.vstack(points);front=points[points[:,0]>=points[:,0].max()-1.e-4,1:]
    if len(front)<3:raise ValueError('contact front is not a finite patch')
    hull=ConvexHull(front);lo=front.min(0);hi=front.max(0)
    # Fixed midpoint quadrature; resolution belongs to geometry, not a case.
    xy=np.meshgrid(*[np.linspace(a,b,257)[:-1]+(b-a)/512 for a,b in zip(lo,hi)])
    samples=np.c_[xy[0].ravel(),xy[1].ravel()]
    samples=samples[np.all(samples@hull.equations[:,:2].T+hull.equations[:,2]<=1e-12,axis=1)]
    radius=float(np.linalg.norm(samples-samples.mean(0),axis=1).mean())
    if not np.isfinite(radius) or radius<=0:raise ValueError('invalid contact torsion radius')
    return radius

def patch_normal_forces(*, positions, normals, center, required_force, friction,
                        radii, normal_jacobian, effort_limits, minimum_normal_force):
    """Minimum peak normal-load effort, subject to 6D soft-contact balance.

    Each contact has normal force, two tangent forces and normal-axis moment.
    The ellipsoidal friction constraint couples shear and twisting friction.
    Joint effort here remains the existing nominal squeeze-load proxy; actual
    robot dynamics and effort safety are handled by the unchanged controller.
    """
    p,n,c,f,mu,r,j,limits=[np.asarray(x,dtype=float) for x in
        (positions,normals,center,required_force,friction,radii,normal_jacobian,effort_limits)]
    k=len(p);w=4*k+1
    if (k<2 or not np.isfinite(minimum_normal_force) or minimum_normal_force<0
        or p.shape!=(k,3) or n.shape!=p.shape or c.shape!=(3,) or f.shape!=(3,)
        or mu.shape!=(k,) or r.shape!=(k,) or j.shape!=(k,len(limits))
        or not all(np.isfinite(x).all() for x in (p,n,c,f,mu,r,j,limits))
        or np.any(mu<=0) or np.any(r<=0) or np.any(limits<=0)
        or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6)):
        raise ValueError('invalid finite contact patch inputs')
    eq=np.zeros((6,w))
    for a in range(k):
        tangent=np.cross(n[a],np.eye(3)[np.argmin(abs(n[a]))]);tangent/=np.linalg.norm(tangent)
        basis=np.column_stack((n[a],tangent,np.cross(n[a],tangent)))
        eq[:3,4*a:4*a+3]=basis
        eq[3:,4*a:4*a+3]=np.cross(np.broadcast_to(p[a]-c,(3,3)),basis.T).T
        eq[3:,4*a+3]=r[a]*n[a]
    rhs=np.r_[f,np.zeros(3)]
    # Remove only redundant equations; inconsistent RHS is rejected explicitly.
    u,s,vt=np.linalg.svd(eq,full_matrices=False);rank=int(np.sum(s>1e-10))
    if np.linalg.norm(rhs-u[:,:rank]@(u[:,:rank].T@rhs))>1e-7:raise ValueError('unbalanced patch wrench')
    eqr=u[:,:rank].T@eq;rr=u[:,:rank].T@rhs
    effort=np.zeros((2*len(limits),w));effort[::2,:4*k:4]=j.T;effort[1::2,:4*k:4]=-j.T
    effort[::2,-1]=-limits;effort[1::2,-1]=-limits
    def cone(x):
        a=x[:4*k].reshape(k,4);return mu*a[:,0]-np.linalg.norm(a[:,1:],axis=1)
    def cone_jac(x):
        a=x[:4*k].reshape(k,4);out=np.zeros((k,w));norm=np.linalg.norm(a[:,1:],axis=1)
        for i in range(k):out[i,4*i]=mu[i];out[i,4*i+1:4*i+4]=-a[i,1:]/max(norm[i],1e-12)
        return out
    objective=np.zeros(w);objective[-1]=1.;objective[:4*k:4]=1e-7
    start=np.linalg.lstsq(eqr,rr,rcond=None)[0];start[:4*k:4]=max(minimum_normal_force,np.linalg.norm(f)/mu.sum())*2
    start[-1]=max(1.,float(np.max(abs(j.T@start[:4*k:4])/limits)))
    bounds=[(None,None)]*w
    for i in range(k):bounds[4*i]=(minimum_normal_force,None)
    bounds[-1]=(0,None)
    result=minimize(lambda x:float(objective@x),start,jac=lambda x:objective,
        bounds=bounds,constraints=[LinearConstraint(eqr,rr,rr),LinearConstraint(effort,-np.inf,0),
        dict(type='ineq',fun=cone,jac=cone_jac)],method='SLSQP',options=dict(maxiter=200,ftol=1e-10))
    if (not result.success or result.x.shape!=(w,) or not np.isfinite(result.x).all()
        or np.min(result.x[:4*k:4])<minimum_normal_force-1e-7 or result.x[-1]<-1e-7
        or np.max(abs(eq@result.x-rhs))>1e-6
        or np.max(effort@result.x)>1e-6 or np.min(cone(result.x))<-1e-6):
        raise ValueError(f'nominal patch equilibrium failed: {result.message}')
    return result.x[:4*k:4],dict(equilibrium_residual=float(np.max(abs(eq@result.x-rhs))),
        contact_torsional_moments_nm=(r*result.x[3:4*k:4]).tolist(),iterations=int(result.nit))


def bounded_normal_reserve(*, forces, requested, jacobian, stiffness, peak,
                           contact_stiffness, maximum_utilization, maximum_lead,
                           quantum, margin):
    """Scale only the existing internal squeeze, within unchanged command limits.

    A finite-patch pressure model suggests reserve; it is not a hard feasibility
    certificate. Preserve gravity-support preload even when requested rotational
    reserve cannot be supplied. An already inadmissible baseline is not repaired
    by weakening a limit; the caller retains its original rejection.
    """
    forces=np.asarray(forces,dtype=float);requested=np.asarray(requested,dtype=float)
    if (requested.shape!=forces.shape or not np.isfinite(requested).all()
            or not np.isfinite(forces).all() or np.any(forces<=0) or np.any(requested<0)):
        raise ValueError('invalid torsion preload request')
    wanted=max(1.,float(np.max(requested/forces)))
    torque=jacobian.T@forces
    utilization=float(np.max(abs(torque)/peak))
    lead=float(np.max(np.maximum(jacobian@(torque/stiffness)+forces/contact_stiffness,0.)))
    quantized_limit=quantum*np.floor((maximum_lead+1e-12)/quantum)-margin
    cap=min(maximum_utilization/max(utilization,1e-12),quantized_limit/max(lead,1e-12))
    scale=max(1.,min(wanted,cap))
    return forces*scale,wanted,scale
