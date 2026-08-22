(() => {
  "use strict";
  const scene = window.AMSRR_ORDER9_SCENE;
  const library = window.AMSRR_ORDER9_MESH_LIBRARY;
  const canvas = document.getElementById("gl");
  const labelsRoot = document.getElementById("labels");
  const status = document.getElementById("status");
  const subtitle = document.getElementById("subtitle");
  if (!scene || !library) throw new Error("Order 9 viewer payload is missing");
  subtitle.textContent = `${scene.subtitle} — ${scene.semantic_scope}`;

  const params = new URLSearchParams(location.search);
  if (params.get("capture") === "1") document.body.classList.add("capture");
  const animation = scene.animation && Array.isArray(scene.animation.frames)
    && scene.animation.frames.length ? scene.animation : null;
  const requestedFrame = params.get("frame");
  let animationFrameIndex = animation && requestedFrame === "last"
    ? animation.frames.length - 1
    : Math.max(
        0,
        Math.min(
          animation ? animation.frames.length - 1 : 0,
          Number.parseInt(requestedFrame || "0", 10) || 0,
        ),
      );
  function activeAnimationFrame() {
    return animation ? animation.frames[animationFrameIndex] : null;
  }
  const urdfFk = animation && animation.frame_encoding === "urdf_fk_v1"
    ? animation.urdf_fk : null;
  const diagnosticsPanel = document.getElementById("diagnostics");
  const diagnosticsCanvas = document.getElementById("diagnostic-chart");
  const diagnosticsLive = document.getElementById("diagnostic-live");
  const diagnosticFrames = animation
    ? animation.frames.filter(frame => frame.diagnostics) : [];
  if (diagnosticFrames.length) diagnosticsPanel.hidden = false;
  let fkCacheFrameIndex = -1, fkCacheLinkMatrices = null;
  function jointMotionMatrix(joint, coordinate) {
    if (joint.joint_type === "fixed") return identity();
    const axis = normalize(joint.axis_xyz || [0, 0, 1]);
    if (joint.joint_type === "prismatic") {
      const value = identity();
      value[12] = axis[0] * coordinate;
      value[13] = axis[1] * coordinate;
      value[14] = axis[2] * coordinate;
      return value;
    }
    const half = 0.5 * coordinate, sine = Math.sin(half);
    return poseMatrix([
      0, 0, 0,
      axis[0] * sine, axis[1] * sine, axis[2] * sine, Math.cos(half)
    ]);
  }
  function activeUrdfLinkMatrices() {
    if (!urdfFk) return null;
    if (fkCacheFrameIndex === animationFrameIndex && fkCacheLinkMatrices)
      return fkCacheLinkMatrices;
    const frame = activeAnimationFrame();
    const matrices = new Array(urdfFk.links.length);
    matrices[urdfFk.root_link_index] = poseMatrix(frame.root_pose_world);
    for (const joint of urdfFk.joints) {
      const coordinate = joint.coordinate_index >= 0
        ? Number(frame.joint_positions[joint.coordinate_index]) : 0;
      matrices[joint.child_link_index] = multiply(
        multiply(
          matrices[joint.parent_link_index],
          new Float32Array(joint.origin_matrix),
        ),
        jointMotionMatrix(joint, coordinate),
      );
    }
    fkCacheFrameIndex = animationFrameIndex;
    fkCacheLinkMatrices = matrices;
    return matrices;
  }
  function activeModelMatrix(instance, index) {
    const frame = activeAnimationFrame();
    if (urdfFk && Number.isInteger(instance.link_index)) {
      const links = activeUrdfLinkMatrices();
      return multiply(
        links[instance.link_index],
        new Float32Array(instance.local_matrix),
      );
    }
    return frame && frame.model_matrices
      ? frame.model_matrices[index] : instance.model_matrix;
  }
  function activeBoxPose(box, index) {
    const frame = activeAnimationFrame();
    return frame && frame.box_poses && frame.box_poses[index]
      ? frame.box_poses[index] : box.pose_world;
  }
  const gl = canvas.getContext("webgl2", {
    alpha: false, antialias: true, depth: true, preserveDrawingBuffer: true
  });
  if (!gl) throw new Error("WebGL2 is required");

  const vertexShader = `#version 300 es
    precision highp float;
    layout(location=0) in vec3 aPosition;
    uniform mat4 uModel;
    uniform mat4 uViewProjection;
    out vec3 vWorld;
    void main() {
      vec4 world = uModel * vec4(aPosition, 1.0);
      vWorld = world.xyz;
      gl_Position = uViewProjection * world;
    }`;
  const fragmentShader = `#version 300 es
    precision highp float;
    in vec3 vWorld;
    uniform vec4 uColor;
    out vec4 outColor;
    void main() {
      vec3 dx = dFdx(vWorld), dy = dFdy(vWorld);
      vec3 n = normalize(cross(dx, dy));
      if (!gl_FrontFacing) n = -n;
      vec3 light = normalize(vec3(0.38, -0.28, 0.88));
      float diffuse = 0.30 + 0.70 * abs(dot(n, light));
      outColor = vec4(uColor.rgb * diffuse, uColor.a);
    }`;
  const lineVertexShader = `#version 300 es
    precision highp float;
    layout(location=0) in vec3 aPosition;
    layout(location=1) in vec4 aColor;
    uniform mat4 uViewProjection;
    uniform float uPointSize;
    out vec4 vColor;
    void main() {
      gl_Position = uViewProjection * vec4(aPosition, 1.0);
      gl_PointSize = uPointSize;
      vColor = aColor;
    }`;
  const lineFragmentShader = `#version 300 es
    precision highp float;
    in vec4 vColor;
    out vec4 outColor;
    void main() {
      if (gl_PointCoord.x > 0.0 || gl_PointCoord.y > 0.0) {
        vec2 d = gl_PointCoord - vec2(0.5);
        if (dot(d,d) > 0.25) discard;
      }
      outColor = vColor;
    }`;
  function shader(type, source) {
    const value = gl.createShader(type);
    gl.shaderSource(value, source); gl.compileShader(value);
    if (!gl.getShaderParameter(value, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(value));
    return value;
  }
  function program(vs, fs) {
    const value = gl.createProgram();
    gl.attachShader(value, shader(gl.VERTEX_SHADER, vs));
    gl.attachShader(value, shader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(value);
    if (!gl.getProgramParameter(value, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(value));
    return value;
  }
  const meshProgram = program(vertexShader, fragmentShader);
  const lineProgram = program(lineVertexShader, lineFragmentShader);
  const meshUniforms = {
    model: gl.getUniformLocation(meshProgram, "uModel"),
    viewProjection: gl.getUniformLocation(meshProgram, "uViewProjection"),
    color: gl.getUniformLocation(meshProgram, "uColor")
  };
  const lineUniforms = {
    viewProjection: gl.getUniformLocation(lineProgram, "uViewProjection"),
    pointSize: gl.getUniformLocation(lineProgram, "uPointSize")
  };

  function decodeBase64(value) {
    const binary = atob(value), bytes = new Uint8Array(binary.length);
    const chunk = 1 << 20;
    for (let start=0; start<binary.length; start+=chunk) {
      const end = Math.min(binary.length, start+chunk);
      for (let i=start; i<end; ++i) bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  function parseBinaryStl(record) {
    const bytes = decodeBase64(record.base64);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const triangles = view.getUint32(80, true);
    if (84 + triangles * 50 !== bytes.byteLength)
      throw new Error(`invalid binary STL ${record.filename}`);
    const positions = new Float32Array(triangles * 9);
    const min = [Infinity,Infinity,Infinity], max = [-Infinity,-Infinity,-Infinity];
    let target = 0;
    for (let triangle=0; triangle<triangles; ++triangle) {
      const base = 84 + triangle * 50 + 12;
      for (let vertex=0; vertex<3; ++vertex) {
        const offset = base + vertex * 12;
        for (let axis=0; axis<3; ++axis) {
          const value = view.getFloat32(offset + axis*4, true);
          positions[target++] = value;
          min[axis] = Math.min(min[axis], value);
          max[axis] = Math.max(max[axis], value);
        }
      }
    }
    const vao = gl.createVertexArray(), buffer = gl.createBuffer();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return {vao, count: positions.length/3, min, max, filename: record.filename};
  }

  const requiredKeys = [...new Set(scene.instances.map(item => item.mesh_key))];
  const meshes = new Map();
  for (let index=0; index<requiredKeys.length; ++index) {
    const key = requiredKeys[index], record = library.meshes[key];
    if (!record) throw new Error(`mesh ${key} absent from shared library`);
    status.textContent = `loading exact STL ${index+1}/${requiredKeys.length}: ${record.filename}`;
    meshes.set(key, parseBinaryStl(record));
  }

  function identity() { return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]); }
  function multiply(a,b) {
    const out = new Float32Array(16);
    for (let col=0; col<4; ++col) for (let row=0; row<4; ++row) {
      let value=0;
      for (let k=0; k<4; ++k) value += a[k*4+row] * b[col*4+k];
      out[col*4+row]=value;
    }
    return out;
  }
  function transformPoint(matrix, point) {
    return [
      matrix[0]*point[0]+matrix[4]*point[1]+matrix[8]*point[2]+matrix[12],
      matrix[1]*point[0]+matrix[5]*point[1]+matrix[9]*point[2]+matrix[13],
      matrix[2]*point[0]+matrix[6]*point[1]+matrix[10]*point[2]+matrix[14]
    ];
  }
  function perspective(fovy, aspect, near, far) {
    const f=1/Math.tan(fovy/2), nf=1/(near-far), out=new Float32Array(16);
    out[0]=f/aspect; out[5]=f; out[10]=(far+near)*nf; out[11]=-1;
    out[14]=2*far*near*nf; return out;
  }
  function orthographic(left,right,bottom,top,near,far) {
    const out=identity();
    out[0]=2/(right-left); out[5]=2/(top-bottom); out[10]=-2/(far-near);
    out[12]=-(right+left)/(right-left); out[13]=-(top+bottom)/(top-bottom);
    out[14]=-(far+near)/(far-near); return out;
  }
  function normalize(v) {
    const n=Math.hypot(v[0],v[1],v[2]) || 1;
    return [v[0]/n,v[1]/n,v[2]/n];
  }
  function cross(a,b) {
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  }
  function lookAt(eye,target,up) {
    const z=normalize([eye[0]-target[0],eye[1]-target[1],eye[2]-target[2]]);
    let x=normalize(cross(up,z));
    if (Math.hypot(...x)<1e-6) x=[1,0,0];
    const y=cross(z,x), out=identity();
    out[0]=x[0];out[1]=y[0];out[2]=z[0];
    out[4]=x[1];out[5]=y[1];out[6]=z[1];
    out[8]=x[2];out[9]=y[2];out[10]=z[2];
    out[12]=-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);
    out[13]=-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);
    out[14]=-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);
    return out;
  }
  function poseMatrix(pose, scale=[1,1,1]) {
    const [x,y,z,qx,qy,qz,qw]=pose;
    const xx=qx*qx, yy=qy*qy, zz=qz*qz, xy=qx*qy, xz=qx*qz, yz=qy*qz;
    const wx=qw*qx, wy=qw*qy, wz=qw*qz;
    return new Float32Array([
      (1-2*(yy+zz))*scale[0], (2*(xy+wz))*scale[0], (2*(xz-wy))*scale[0], 0,
      (2*(xy-wz))*scale[1], (1-2*(xx+zz))*scale[1], (2*(yz+wx))*scale[1], 0,
      (2*(xz+wy))*scale[2], (2*(yz-wx))*scale[2], (1-2*(xx+yy))*scale[2], 0,
      x,y,z,1
    ]);
  }

  function makeArrayBuffer(values, components, location) {
    const vao = gl.createVertexArray(), buffer = gl.createBuffer();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, components, gl.FLOAT, false, components*4, 0);
    gl.bindVertexArray(null);
    return {vao,buffer,count:values.length/components,components};
  }
  const markerPoints=[], markerColors=[], markerLines=[], markerLineColors=[];
  for (const marker of scene.markers) {
    markerPoints.push(...marker.position_world);
    markerColors.push(...marker.color_rgba);
    if (marker.direction_world) {
      const length=marker.selected ? 0.18 : 0.12;
      markerLines.push(...marker.position_world,
        marker.position_world[0]+marker.direction_world[0]*length,
        marker.position_world[1]+marker.direction_world[1]*length,
        marker.position_world[2]+marker.direction_world[2]*length);
      markerLineColors.push(...marker.color_rgba,...marker.color_rgba);
    }
  }
  function coloredGeometry(positions,colors) {
    const vao=gl.createVertexArray(), pb=gl.createBuffer(), cb=gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(colors),gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,4,gl.FLOAT,false,0,0);
    gl.bindVertexArray(null); return {vao,count:positions.length/3};
  }
  const markerPointGeometry=coloredGeometry(markerPoints,markerColors);
  const markerLineGeometry=coloredGeometry(markerLines,markerLineColors);

  const cubePositions = new Float32Array([
    -1,-1,-1, 1,-1,-1, 1,1,-1, -1,-1,-1, 1,1,-1, -1,1,-1,
    -1,-1,1, 1,1,1, 1,-1,1, -1,-1,1, -1,1,1, 1,1,1,
    -1,-1,-1, -1,1,-1, -1,1,1, -1,-1,-1, -1,1,1, -1,-1,1,
    1,-1,-1, 1,-1,1, 1,1,1, 1,-1,-1, 1,1,1, 1,1,-1,
    -1,-1,-1, -1,-1,1, 1,-1,1, -1,-1,-1, 1,-1,1, 1,-1,-1,
    -1,1,-1, 1,1,1, -1,1,1, -1,1,-1, 1,1,-1, 1,1,1
  ]);
  const cubeVao=gl.createVertexArray(), cubeBuffer=gl.createBuffer();
  gl.bindVertexArray(cubeVao);gl.bindBuffer(gl.ARRAY_BUFFER,cubeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,cubePositions,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  gl.bindVertexArray(null);

  const suppliedBounds = animation && animation.bounds;
  const bounds = suppliedBounds
    && Array.isArray(suppliedBounds.min) && suppliedBounds.min.length === 3
    && Array.isArray(suppliedBounds.max) && suppliedBounds.max.length === 3
    ? {min:[...suppliedBounds.min], max:[...suppliedBounds.max]}
    : {min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  function include(point) {
    for(let a=0;a<3;++a){bounds.min[a]=Math.min(bounds.min[a],point[a]);bounds.max[a]=Math.max(bounds.max[a],point[a]);}
  }
  const boundsFrames = animation ? animation.frames : [null];
  if (!suppliedBounds) {
    for(const frame of boundsFrames) {
      for(let index=0; index<scene.instances.length; ++index) {
        const instance=scene.instances[index];
        if(instance.layer!=="visual" || instance.detail_class!=="shape")continue;
        const mesh=meshes.get(instance.mesh_key);
        const matrix=frame && frame.model_matrices
          ? frame.model_matrices[index] : instance.model_matrix;
        for(const x of [mesh.min[0],mesh.max[0]])for(const y of [mesh.min[1],mesh.max[1]])for(const z of [mesh.min[2],mesh.max[2]])
          include(transformPoint(matrix,[x,y,z]));
      }
    }
    for(const marker of scene.markers) include(marker.position_world);
    for(const frame of boundsFrames) {
      for(let index=0; index<scene.boxes.length; ++index) {
        const box=scene.boxes[index];
        const pose=frame && frame.box_poses && frame.box_poses[index]
          ? frame.box_poses[index] : box.pose_world;
        const matrix=poseMatrix(pose,box.size_m.map(x=>x/2));
        for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])include(transformPoint(matrix,[x,y,z]));
      }
    }
  }
  if(!Number.isFinite(bounds.min[0])){bounds.min=[-1,-1,-1];bounds.max=[1,1,1];}
  const center=bounds.min.map((v,i)=>(v+bounds.max[i])/2);
  const radius=Math.max(0.3,Math.hypot(...bounds.max.map((v,i)=>(v-bounds.min[i])/2)));
  const camera={target:[...center],distance:radius*2.6,yaw:Math.PI/4,pitch:0.62,ortho:false};
  function setView(name) {
    if(name==="top"){camera.yaw=0;camera.pitch=Math.PI/2-1e-4;camera.ortho=true;}
    else if(name==="front"){camera.yaw=0;camera.pitch=0;camera.ortho=true;}
    else if(name==="side"){camera.yaw=Math.PI/2;camera.pitch=0;camera.ortho=true;}
    else {camera.yaw=Math.PI/4;camera.pitch=0.62;camera.ortho=false;}
    camera.target=[...center];camera.distance=radius*2.6;render();
  }

  const allLabels=[...scene.labels,...scene.markers.map(marker=>({
    label_id:marker.marker_id,label:marker.label,position_world:marker.position_world,
    kind:marker.kind,color_rgba:marker.color_rgba,selected:marker.selected
  }))];
  const labelElements=allLabels.map(item=>{
    const element=document.createElement("div");
    element.className=`label ${item.kind||""}${item.selected?" selected":""}`;
    element.textContent=item.label;
    const color=item.color_rgba||[0.1,0.1,0.1,1];
    element.style.color=`rgb(${color.slice(0,3).map(x=>Math.round(x*255)).join(",")})`;
    labelsRoot.appendChild(element); return [item,element];
  });
  let showLabels=true,fullDetail=params.get("detail")==="full",collision=false,showObject=true;
  document.getElementById("detail-toggle").checked=fullDetail;
  function project(point,matrix) {
    const x=point[0],y=point[1],z=point[2];
    const clip=[
      matrix[0]*x+matrix[4]*y+matrix[8]*z+matrix[12],
      matrix[1]*x+matrix[5]*y+matrix[9]*z+matrix[13],
      matrix[2]*x+matrix[6]*y+matrix[10]*z+matrix[14],
      matrix[3]*x+matrix[7]*y+matrix[11]*z+matrix[15]
    ];
    if(clip[3]<=0)return null;
    return [(clip[0]/clip[3]*.5+.5)*canvas.clientWidth,(-clip[1]/clip[3]*.5+.5)*canvas.clientHeight,clip[2]/clip[3]];
  }
  function resize() {
    const dpr=Math.min(2,window.devicePixelRatio||1),w=Math.round(canvas.clientWidth*dpr),h=Math.round(canvas.clientHeight*dpr);
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
    gl.viewport(0,0,w,h);
  }
  function render() {
    resize();
    const cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw);
    const eye=[camera.target[0]+camera.distance*cp*cy,camera.target[1]+camera.distance*cp*sy,camera.target[2]+camera.distance*sp];
    const up=Math.abs(cp)<.01?[0,1,0]:[0,0,1];
    const view=lookAt(eye,camera.target,up),aspect=canvas.width/canvas.height;
    const projection=camera.ortho?orthographic(-radius*1.3*aspect,radius*1.3*aspect,-radius*1.3,radius*1.3,.001,radius*10):
      perspective(Math.PI/4,aspect,.005,radius*20);
    const vp=multiply(projection,view);
    gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
    gl.clearColor(.96,.97,.985,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(meshProgram);gl.uniformMatrix4fv(meshUniforms.viewProjection,false,vp);
    const layer=collision?"collision":"visual";
    for(let instanceIndex=0; instanceIndex<scene.instances.length; ++instanceIndex) {
      const instance=scene.instances[instanceIndex];
      if(instance.layer!==layer || (!fullDetail&&instance.detail_class==="full"))continue;
      const mesh=meshes.get(instance.mesh_key);
      gl.bindVertexArray(mesh.vao);
      gl.uniformMatrix4fv(meshUniforms.model,false,new Float32Array(activeModelMatrix(instance,instanceIndex)));
      gl.uniform4fv(meshUniforms.color,new Float32Array(instance.color_rgba));
      gl.drawArrays(gl.TRIANGLES,0,mesh.count);
    }
    if(showObject) {
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.bindVertexArray(cubeVao);
      for(let boxIndex=0; boxIndex<scene.boxes.length; ++boxIndex) {
        const box=scene.boxes[boxIndex];
        gl.uniformMatrix4fv(meshUniforms.model,false,poseMatrix(activeBoxPose(box,boxIndex),box.size_m.map(x=>x/2)));
        gl.uniform4fv(meshUniforms.color,new Float32Array(box.color_rgba));
        gl.drawArrays(gl.TRIANGLES,0,cubePositions.length/3);
      }
      gl.depthMask(true);gl.disable(gl.BLEND);
    }
    gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineUniforms.viewProjection,false,vp);
    if(markerLineGeometry.count){gl.bindVertexArray(markerLineGeometry.vao);gl.uniform1f(lineUniforms.pointSize,1);gl.drawArrays(gl.LINES,0,markerLineGeometry.count);}
    if(markerPointGeometry.count){gl.bindVertexArray(markerPointGeometry.vao);gl.uniform1f(lineUniforms.pointSize,13);gl.drawArrays(gl.POINTS,0,markerPointGeometry.count);}
    const frame=activeAnimationFrame();
    for(let labelIndex=0;labelIndex<labelElements.length;++labelIndex) {
      const [item,element]=labelElements[labelIndex];
      let position=frame && frame.label_positions
        && labelIndex<scene.labels.length
        && frame.label_positions[labelIndex]
        ? frame.label_positions[labelIndex] : item.position_world;
      if (
        frame && !frame.label_positions && urdfFk
        && labelIndex < scene.labels.length
        && Number.isInteger(item.link_index)
      ) {
        const matrix = activeUrdfLinkMatrices()[item.link_index];
        position = [matrix[12], matrix[13], matrix[14]];
      }
      const projected=project(position,vp);
      const visible=showLabels&&projected&&projected[2]>=-1&&projected[2]<=1;
      element.style.display=visible?"block":"none";
      if(visible){element.style.left=`${projected[0]}px`;element.style.top=`${projected[1]}px`;}
    }
    const animationText=frame
      ? ` | t=${Number(frame.time_s).toFixed(2)} s | ${frame.status_text || frame.phase || ""}`
      : "";
    status.textContent=`exact STL | ${scene.instances.filter(x=>x.layer===layer&& (fullDetail||x.detail_class!=="full")).length} mesh instances${animationText} | drag: orbit, wheel: zoom`;
    drawDiagnostics();
  }

  function diagnosticSeries(key, arrayIndex=null) {
    return animation.frames.map(frame => {
      const value=frame.diagnostics&&frame.diagnostics[key];
      if(arrayIndex===null)return Number(value);
      return Array.isArray(value)&&arrayIndex<value.length?Number(value[arrayIndex]):NaN;
    });
  }
  function drawDiagnostics() {
    if(!diagnosticFrames.length)return;
    const dpr=Math.min(2,window.devicePixelRatio||1);
    const width=Math.max(320,Math.round(diagnosticsCanvas.clientWidth*dpr));
    const height=Math.max(180,Math.round(diagnosticsCanvas.clientHeight*dpr));
    if(diagnosticsCanvas.width!==width||diagnosticsCanvas.height!==height){diagnosticsCanvas.width=width;diagnosticsCanvas.height=height;}
    const ctx=diagnosticsCanvas.getContext("2d"),scale=dpr;
    ctx.clearRect(0,0,width,height);ctx.save();ctx.scale(scale,scale);
    const w=width/scale,h=height/scale,left=42,right=8,top=14,gap=24,plotH=(h-top-gap-18)/2;
    const times=animation.frames.map(frame=>Number(frame.time_s));
    const t0=times[0],t1=Math.max(t0+1e-6,times[times.length-1]);
    const root=diagnosticSeries("robot_root_height_m"),object=diagnosticSeries("object_height_m"),command=diagnosticSeries("command_body_height_m");
    const baseRoot=root.find(Number.isFinite)||0,baseObject=object.find(Number.isFinite)||0;
    const heights=[root.map(v=>(v-baseRoot)*1000),object.map(v=>(v-baseObject)*1000),command.map(v=>(v-baseRoot)*1000)];
    const f0=diagnosticSeries("selected_contact_force_n",0),f1=diagnosticSeries("selected_contact_force_n",1),support=diagnosticSeries("vertical_support_force_n"),weight=diagnosticSeries("payload_weight_n"),threshold=diagnosticSeries("contact_force_threshold_n");
    function plot(y,series,colors,labels,unit,zeroFloor=false){
      const finite=series.flat().filter(Number.isFinite);let lo=finite.length?Math.min(...finite):0,hi=finite.length?Math.max(...finite):1;
      if(zeroFloor)lo=0;if(Math.abs(hi-lo)<1e-6){hi+=1;lo-=zeroFloor?0:1;}
      const pad=(hi-lo)*.08;hi+=pad;if(!zeroFloor)lo-=pad;
      const xOf=i=>left+(times[i]-t0)/(t1-t0)*(w-left-right),yOf=v=>y+plotH-(v-lo)/(hi-lo)*plotH;
      ctx.strokeStyle="#c5ccd3";ctx.lineWidth=1;ctx.strokeRect(left,y,w-left-right,plotH);
      ctx.fillStyle="#52606d";ctx.font="11px ui-monospace,monospace";ctx.fillText(`${hi.toFixed(1)} ${unit}`,2,y+10);ctx.fillText(lo.toFixed(1),12,y+plotH);
      series.forEach((values,s)=>{ctx.beginPath();let started=false;values.forEach((v,i)=>{if(!Number.isFinite(v))return;const x=xOf(i),py=yOf(v);if(!started){ctx.moveTo(x,py);started=true;}else ctx.lineTo(x,py);});ctx.strokeStyle=colors[s];ctx.lineWidth=1.8;ctx.stroke();});
      let lx=left;labels.forEach((label,i)=>{ctx.fillStyle=colors[i];ctx.fillRect(lx,y-11,10,3);ctx.fillStyle="#263441";ctx.fillText(label,lx+14,y-6);lx+=14+ctx.measureText(label).width+14;});
      const cursor=left+(times[animationFrameIndex]-t0)/(t1-t0)*(w-left-right);ctx.strokeStyle="#111827";ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(cursor,y);ctx.lineTo(cursor,y+plotH);ctx.stroke();ctx.setLineDash([]);
    }
    plot(top,heights,["#1769aa","#dc7c12","#2f9e44"],["robot Δz","object Δz","body cmd Δz"],"mm",false);
    plot(top+plotH+gap,[f0,f1,support,weight,threshold],["#7b2cbf","#d6336c","#087f5b","#343a40","#e03131"],["contact 0","contact 1","vertical support","weight","contact gate"],"N",true);
    const frame=activeAnimationFrame(),d=frame.diagnostics,forces=d.selected_contact_force_n||[],slip=d.contact_relative_vertical_speed_mps||[];
    diagnosticsLive.textContent=`phase=${frame.phase} | root=${(d.robot_root_height_m*1000).toFixed(1)} mm | object=${(d.object_height_m*1000).toFixed(1)} mm\ncontact=[${forces.map(v=>Number(v).toFixed(2)).join(", ")}] N | vertical support=${Number(d.vertical_support_force_n).toFixed(2)} / weight=${Number(d.payload_weight_n).toFixed(2)} N | relative vz=[${slip.map(v=>(Number(v)*1000).toFixed(1)).join(", ")}] mm/s`;
    ctx.restore();
  }

  let dragging=false,last=[0,0];
  canvas.addEventListener("pointerdown",event=>{dragging=true;last=[event.clientX,event.clientY];canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener("pointerup",()=>dragging=false);
  canvas.addEventListener("pointermove",event=>{
    if(!dragging)return;
    camera.yaw-=(event.clientX-last[0])*.008;camera.pitch=Math.max(-1.52,Math.min(1.52,camera.pitch+(event.clientY-last[1])*.008));
    camera.ortho=false;last=[event.clientX,event.clientY];render();
  });
  canvas.addEventListener("wheel",event=>{event.preventDefault();camera.distance=Math.max(radius*.2,Math.min(radius*12,camera.distance*Math.exp(event.deltaY*.001)));render();},{passive:false});
  window.addEventListener("resize",render);
  document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>setView(button.dataset.view)));
  document.getElementById("fit").addEventListener("click",()=>{camera.target=[...center];camera.distance=radius*2.6;render();});
  document.getElementById("labels-toggle").addEventListener("change",event=>{showLabels=event.target.checked;render();});
  document.getElementById("detail-toggle").addEventListener("change",event=>{fullDetail=event.target.checked;render();});
  document.getElementById("collision-toggle").addEventListener("change",event=>{collision=event.target.checked;render();});
  document.getElementById("object-toggle").addEventListener("change",event=>{showObject=event.target.checked;render();});
  const animationControls=document.getElementById("animation-controls");
  const animationPlay=document.getElementById("animation-play");
  const animationSlider=document.getElementById("animation-slider");
  const animationTime=document.getElementById("animation-time");
  const animationSpeed=document.getElementById("animation-speed");
  let animationPlaying=false,animationRequest=null,animationLastTimestamp=null;
  function updateAnimationControls() {
    if(!animation)return;
    const frame=activeAnimationFrame();
    animationSlider.value=String(animationFrameIndex);
    animationPlay.textContent=animationPlaying?"Pause":"Play";
    animationTime.textContent=`t=${Number(frame.time_s).toFixed(2)} s | ${frame.status_text || frame.phase || ""}`;
  }
  function setAnimationFrame(index) {
    if(!animation)return;
    animationFrameIndex=Math.max(0,Math.min(animation.frames.length-1,Number(index)||0));
    updateAnimationControls();
    render();
  }
  function stopAnimation() {
    animationPlaying=false;
    animationLastTimestamp=null;
    if(animationRequest!==null)cancelAnimationFrame(animationRequest);
    animationRequest=null;
    updateAnimationControls();
  }
  function animationTick(timestamp) {
    if(!animationPlaying||!animation)return;
    if(animationLastTimestamp===null)animationLastTimestamp=timestamp;
    const speed=Number(animationSpeed.value)||1;
    const elapsed=(timestamp-animationLastTimestamp)*0.001*speed;
    let next=animationFrameIndex;
    while(next+1<animation.frames.length
      && Number(animation.frames[next+1].time_s)-Number(animation.frames[animationFrameIndex].time_s)<=elapsed+1e-9)next++;
    if(next>animationFrameIndex){
      const advanced=Number(animation.frames[next].time_s)-Number(animation.frames[animationFrameIndex].time_s);
      animationLastTimestamp+=advanced/speed*1000;
      setAnimationFrame(next);
    }
    if(animationFrameIndex>=animation.frames.length-1){stopAnimation();return;}
    animationRequest=requestAnimationFrame(animationTick);
  }
  if(animation) {
    animationControls.hidden=false;
    animationSlider.max=String(animation.frames.length-1);
    animationSlider.addEventListener("input",()=>{stopAnimation();setAnimationFrame(animationSlider.value);});
    animationPlay.addEventListener("click",()=>{
      if(animationPlaying){stopAnimation();return;}
      if(animationFrameIndex>=animation.frames.length-1)setAnimationFrame(0);
      animationPlaying=true;animationLastTimestamp=null;updateAnimationControls();
      animationRequest=requestAnimationFrame(animationTick);
    });
    updateAnimationControls();
  }
  const review = scene.metadata && scene.metadata.review;
  const reviewControls = document.getElementById("review-controls");
  const reviewStatus = document.getElementById("review-status");
  const reviewNote = document.getElementById("review-note");
  const reviewButtons = [...document.querySelectorAll("[data-review-action]")];
  function setReviewBusy(busy) {
    for (const button of reviewButtons) button.disabled = busy;
  }
  function downloadReviewRequest(action) {
    const payload = {
      case_id: review.case_id,
      action,
      note: reviewNote.value,
      scene_sha256: review.scene_sha256 || null,
      viewer_version: scene.viewer_version
    };
    const blob = new Blob([JSON.stringify(payload,null,2)+"\n"], {type:"application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${review.case_id}.${action}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    reviewStatus.textContent = "review server未接続: request JSONをdownloadしました";
  }
  async function reviewRequest(action) {
    if (!review) return;
    if (location.protocol === "file:") {
      downloadReviewRequest(action);
      return;
    }
    setReviewBusy(true);
    reviewStatus.textContent = `${action}を処理中…`;
    try {
      const response = await fetch(review.api_path || "/api/order9-c3-review", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          case_id: review.case_id,
          action,
          note: reviewNote.value,
          scene_sha256: review.scene_sha256 || null
        })
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      reviewStatus.textContent = `${payload.case_id}: ${payload.review_status}`;
      if (payload.note !== undefined) reviewNote.value = payload.note || "";
      if (payload.reload_url) location.replace(payload.reload_url);
    } catch (error) {
      reviewStatus.textContent = `失敗: ${error.message}`;
    } finally {
      setReviewBusy(false);
    }
  }
  async function loadReviewState() {
    if (!review || location.protocol === "file:") return;
    try {
      const endpoint = review.api_path || "/api/order9-c3-review";
      const response = await fetch(`${endpoint}?case_id=${encodeURIComponent(review.case_id)}`);
      const payload = await response.json();
      if (response.ok && payload.ok) {
        reviewStatus.textContent = `${payload.case_id}: ${payload.review_status}`;
        reviewNote.value = payload.note || "";
      }
    } catch (_error) {
      reviewStatus.textContent = "review serverの状態を取得できません";
    }
  }
  if (review) {
    reviewControls.hidden = false;
    for (const button of reviewButtons) {
      button.addEventListener("click",()=>reviewRequest(button.dataset.reviewAction));
    }
    loadReviewState();
  }
  setView(params.get("view")||scene.default_view||"iso");
  document.documentElement.dataset.ready="true";
  window.AMSRR_ORDER9_VIEWER_READY=true;
})();
