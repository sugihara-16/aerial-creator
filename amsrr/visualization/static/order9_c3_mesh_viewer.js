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

  const bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  function include(point) {
    for(let a=0;a<3;++a){bounds.min[a]=Math.min(bounds.min[a],point[a]);bounds.max[a]=Math.max(bounds.max[a],point[a]);}
  }
  for(const instance of scene.instances.filter(x=>x.layer==="visual"&&x.detail_class==="shape")){
    const mesh=meshes.get(instance.mesh_key), matrix=instance.model_matrix;
    for(const x of [mesh.min[0],mesh.max[0]])for(const y of [mesh.min[1],mesh.max[1]])for(const z of [mesh.min[2],mesh.max[2]])
      include(transformPoint(matrix,[x,y,z]));
  }
  for(const marker of scene.markers) include(marker.position_world);
  for(const box of scene.boxes) {
    const matrix=poseMatrix(box.pose_world,box.size_m.map(x=>x/2));
    for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])include(transformPoint(matrix,[x,y,z]));
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
    for(const instance of scene.instances) {
      if(instance.layer!==layer || (!fullDetail&&instance.detail_class==="full"))continue;
      const mesh=meshes.get(instance.mesh_key);
      gl.bindVertexArray(mesh.vao);
      gl.uniformMatrix4fv(meshUniforms.model,false,new Float32Array(instance.model_matrix));
      gl.uniform4fv(meshUniforms.color,new Float32Array(instance.color_rgba));
      gl.drawArrays(gl.TRIANGLES,0,mesh.count);
    }
    if(showObject) {
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.bindVertexArray(cubeVao);
      for(const box of scene.boxes) {
        gl.uniformMatrix4fv(meshUniforms.model,false,poseMatrix(box.pose_world,box.size_m.map(x=>x/2)));
        gl.uniform4fv(meshUniforms.color,new Float32Array(box.color_rgba));
        gl.drawArrays(gl.TRIANGLES,0,cubePositions.length/3);
      }
      gl.depthMask(true);gl.disable(gl.BLEND);
    }
    gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineUniforms.viewProjection,false,vp);
    if(markerLineGeometry.count){gl.bindVertexArray(markerLineGeometry.vao);gl.uniform1f(lineUniforms.pointSize,1);gl.drawArrays(gl.LINES,0,markerLineGeometry.count);}
    if(markerPointGeometry.count){gl.bindVertexArray(markerPointGeometry.vao);gl.uniform1f(lineUniforms.pointSize,13);gl.drawArrays(gl.POINTS,0,markerPointGeometry.count);}
    for(const [item,element] of labelElements) {
      const projected=project(item.position_world,vp);
      const visible=showLabels&&projected&&projected[2]>=-1&&projected[2]<=1;
      element.style.display=visible?"block":"none";
      if(visible){element.style.left=`${projected[0]}px`;element.style.top=`${projected[1]}px`;}
    }
    status.textContent=`exact STL | ${scene.instances.filter(x=>x.layer===layer&& (fullDetail||x.detail_class!=="full")).length} mesh instances | drag: orbit, wheel: zoom`;
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
  setView(params.get("view")||scene.default_view||"iso");
  document.documentElement.dataset.ready="true";
  window.AMSRR_ORDER9_VIEWER_READY=true;
})();
