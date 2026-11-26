// 헤어 저폴리(먼 시점용, 게임 줌 HLK보다 멀 때): three.js SimplifyModifier로 헤어·캡을 각각 1/5로 줄인다.
//   (tools/3d/node_modules에 three 0.160이 있어야 한다 — 다른 빌드와 같은 심볼릭 링크)
//   node hairlod.mjs            → hair01_lo.json, hair02_lo.json
import * as THREE from 'three';import {SimplifyModifier} from 'three/examples/jsm/modifiers/SimplifyModifier.js';import fs from 'fs';import path from 'path';import {fileURLToPath} from 'url';
const D=path.dirname(fileURLToPath(import.meta.url)),KEEP=0.2;
for(const n of ['hair01','hair02']){const d=JSON.parse(fs.readFileSync(path.join(D,n+'_lite.json'),'utf8'));const out={name:d.name+' 저폴리',ver:d.ver,coord:d.coord,color:d.color,normals:d.normals};
 for(const k of ['hair','cap']){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(d[k].v,3));g.setIndex(d[k].f);
  const s=new SimplifyModifier().modify(g,Math.floor(g.attributes.position.count*(1-KEEP)));
  out[k]={v:Array.from(s.attributes.position.array).map(v=>+v.toFixed(5)),f:Array.from(s.index.array)};console.log(n,k,d[k].f.length/3,'->',out[k].f.length/3);}
 fs.writeFileSync(path.join(D,n+'_lo.json'),JSON.stringify(out));}
