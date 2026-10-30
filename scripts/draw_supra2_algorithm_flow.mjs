// Render the model algorithm only; no model forward or RTL execution is used.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const sharp = require('sharp');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const W = 2140, H = 1860;
const C = { ink:'#233141', line:'#45546a', muted:'#66768a', blue:'#eaf3fc', blueEdge:'#537da9', green:'#edf5e9', greenEdge:'#688c59', rose:'#faecea', roseEdge:'#b77c73', gold:'#fff5dd', goldEdge:'#ac9050', purple:'#f1edfa', purpleEdge:'#8a75ac', gray:'#f4f5f7' };
const parts = [];
const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function text(x,y,lines,size=24,weight=400,color=C.ink,anchor='middle',gap=30) {
  const a = Array.isArray(lines) ? lines : [lines];
  parts.push(`<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${color}">${a.map((s,i)=>`<tspan x="${x}" dy="${i ? gap : 0}">${esc(s)}</tspan>`).join('')}</text>`);
}
function rect(x,y,w,h,fill='white',stroke=C.line,rx=7,dash='') {
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="1.8" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`);
}
function box(x,y,w,h,title,sub='',fill=C.blue,stroke=C.blueEdge,size=25) {
  rect(x,y,w,h,fill,stroke);
  if(sub) { text(x+w/2,y+h/2-3,title,size,600); text(x+w/2,y+h/2+25,sub,20,400,C.muted); }
  else text(x+w/2,y+h/2+8,title,size,600);
}
function arrow(points,color=C.line,dash='',width=2.1) {
  const marker = color===C.purpleEdge ? 'purple' : color===C.muted ? 'muted' : 'arrow';
  parts.push(`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ''} marker-end="url(#${marker})"/>`);
}
function diamond(x,y,w,h,label) {
  parts.push(`<path d="M${x},${y-h/2} L${x+w/2},${y} L${x},${y+h/2} L${x-w/2},${y} Z" fill="${C.gold}" stroke="${C.goldEdge}" stroke-width="1.8"/>`);
  text(x,y+8,label,24,600);
}
function add(x,y,gated=false) {
  parts.push(`<circle cx="${x}" cy="${y}" r="23" fill="${C.rose}" stroke="${C.roseEdge}" stroke-width="1.8"/>`);
  text(x,y+10,'+',35,400);
  if(gated) text(x+52,y+7,'gate × update',20,400,C.muted,'start');
}

parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="${C.line}"/></marker><marker id="purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="${C.purpleEdge}"/></marker><marker id="muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="${C.muted}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="'Times New Roman', Georgia, serif">`);
text(1070,48,'Supra2-IMG: Quantized Rectified-Flow Text-to-Image Sampling',34,700);
rect(35,83,1060,1717,'white',C.ink,0);
rect(1125,83,980,1717,'white',C.ink,0);
text(565,128,'(a) Sampling algorithm',30,700);
text(1615,128,'(b) One DiT block',30,700);

// Input preparation, once per prompt. T5 and VAE are host-side partitions.
box(85,160,355,58,'Text prompt / null condition','',C.green,C.greenEdge);
arrow([[262,218],[262,247]]);
box(85,247,355,66,'T5 text encoder','Host-side context preparation',C.green,C.greenEdge);
arrow([[262,313],[262,342]]);
box(85,342,355,77,'Context projection + mask','C_cond and C_uncond; 768 → 576',C.green,C.greenEdge,24);
box(650,160,365,58,'Gaussian latent z₀','',C.blue,C.blueEdge);
arrow([[832,218],[832,251]]);
box(650,251,365,65,'Initialize s = 0','z₀ ∈ R^(4 × 32 × 32)',C.blue,C.blueEdge);
arrow([[832,316],[832,451],[710,451],[710,480]]);

// Rectified-flow loop. The block scan is evaluated for each CFG branch.
rect(335,454,700,1000,'none','#a8b5c4',9,'8 6');
parts.push('<rect x="351" y="439" width="380" height="29" fill="white"/>');
text(360,460,'Evaluate both CFG branches at step s',22,600,C.muted,'start');
box(465,480,490,74,'Patchify → x_embed → position add','256 tokens × 576 features');
arrow([[710,554],[710,590]]);
box(465,590,490,69,'Timestep conditioning','Load scale / shift / gate for step s',C.purple,C.purpleEdge);
arrow([[710,659],[710,700]]);
text(752,691,'i = 0',21,400,C.muted,'start');
diamond(710,755,280,110,'skip[s, i] = 1?');
arrow([[710,810],[710,846]]);
text(738,833,'No',21,400,C.muted,'start');
box(465,846,490,82,'Execute DiT block i','Self attention → cross attention → MLP');
arrow([[710,928],[710,965]]);
diamond(710,1020,280,110,'i + 1 < 14?');
// Skipped blocks preserve the input feature exactly.
arrow([[850,755],[989,755],[989,1020],[850,1020]]);
text(965,790,'Yes',20,400,C.muted);
text(1005,952,'h_out = h_in',18,400,C.muted,'end');
// Next block bypasses initialization and returns to the skip predicate.
arrow([[570,1020],[410,1020],[410,755],[570,755]]);
text(400,1100,'Yes: i ← i + 1',20,400,C.muted);
arrow([[710,1075],[710,1120]]);
text(738,1104,'No',21,400,C.muted,'start');
box(465,1120,490,84,'Final LN / AdaLN → projection','576 → 16; unpatchify to 4 × 32 × 32');
arrow([[710,1204],[710,1250]]);
text(710,1233,'v_cond and v_uncond',21,400,C.muted);
box(465,1250,490,76,'Classifier-free guidance (CFG = 3)','v = v_uncond + 3 × (v_cond − v_uncond)',C.rose,C.roseEdge,24);
arrow([[710,1326],[710,1365]]);
box(465,1365,490,76,'Euler-flow update','z_(s+1) = z_s + Δt_s × v',C.rose,C.roseEdge);
arrow([[710,1441],[710,1475]]);
box(545,1475,330,48,'s ← s + 1','',C.gray,C.line,23);
arrow([[710,1523],[710,1535]]);
diamond(710,1590,280,110,'s < 50?');
arrow([[570,1590],[285,1590],[285,517],[465,517]]);
text(304,1538,['Yes','next step'],22,400,C.muted,'start');
arrow([[710,1645],[710,1690]]);
text(738,1673,'No',21,400,C.muted,'start');
box(465,1690,490,73,'VAE decoder → 256 × 256 image','Host-side decode of final latent z₅₀',C.green,C.greenEdge);
// Prompt context is a read-only side input, prepared outside the time loop.
arrow([[85,380],[64,380],[64,886],[465,886]],C.purpleEdge,'6 5');
parts.push('<rect x="78" y="847" width="200" height="58" fill="white"/>');
text(89,869,['Text context + mask','reused at each step'],20,400,C.purpleEdge,'start',26);
text(130,537,['50-step','feedback'],23,600,C.muted);

// Expanded block: three residual sublayers with explicit skip connections.
text(1615,171,'256 tokens · width 576 · 9 heads × 64 · MLP ratio 4',23,400,C.muted);
box(1395,220,450,55,'Input feature h_i','',C.gray,C.line,25);
arrow([[1620,275],[1620,330]]);
rect(1210,306,810,278,'none','#a8b5c4',9,'7 6');
text(1230,332,'Self attention',21,600,C.muted,'start');
box(1395,350,450,56,'LayerNorm + AdaLN');
arrow([[1620,406],[1620,434]]);
box(1395,434,450,79,'QKV → QKᵀ / √64 → softmax','PV → output projection',C.blue,C.blueEdge,24);
arrow([[1620,513],[1620,540]]);
add(1620,563,true);
arrow([[1395,247],[1278,247],[1278,563],[1597,563]]);
arrow([[1620,586],[1620,649]]);
// Plain cross-attention LayerNorm, without timestep AdaLN modulation.
rect(1210,620,810,262,'none','#a8b5c4',9,'7 6');
text(1230,646,'Cross attention',21,600,C.muted,'start');
box(1395,665,450,53,'LayerNorm');
arrow([[1620,718],[1620,745]]);
box(1395,745,450,78,'QKᵀ → masked softmax → PV','Output projection; Q: image, K/V: text',C.green,C.greenEdge,23);
arrow([[1620,823],[1620,849]]);
add(1620,872,false);
arrow([[1620,605],[1297,605],[1297,872],[1597,872]]);
text(1240,793,['C_b','+ mask'],21,400,C.purpleEdge);
arrow([[1285,787],[1395,787]],C.purpleEdge,'6 5');
arrow([[1620,895],[1620,960]]);
rect(1210,933,810,263,'none','#a8b5c4',9,'7 6');
text(1230,959,'Gated MLP',21,600,C.muted,'start');
box(1395,979,450,54,'LayerNorm + AdaLN');
arrow([[1620,1033],[1620,1060]]);
box(1395,1060,450,73,'MLP.0 → GELU(tanh) → MLP.2','576 → 2304 → 576',C.gold,C.goldEdge,24);
arrow([[1620,1133],[1620,1162]]);
add(1620,1185,true);
arrow([[1620,918],[1316,918],[1316,1185],[1597,1185]]);
arrow([[1620,1208],[1620,1245]]);
box(1395,1245,450,55,'Output feature h_(i+1)','',C.gray,C.line,25);

// Algorithmic quantization detail, independent of hardware scheduling.
rect(1170,1345,890,199,'white',C.blueEdge,8);
text(1615,1380,'Quantized linear operator (QLinear)',25,700);
const qx=[1193,1400,1607,1814];
const qlabels=[['Hadamard','FP16 features'],['Dynamic INT8','per-token scale'],['INT8 GEMM','INT32 sum'],['Scale + bias','FP16 output']];
for(let i=0;i<4;i++) {
  box(qx[i],1401,183,71,qlabels[i][0],qlabels[i][1],i===2?C.gold:C.blue,i===2?C.goldEdge:C.blueEdge,21);
  if(i<3) arrow([[qx[i]+183,1436],[qx[i+1],1436]]);
}
text(1615,1512,'y = FP16(acc_INT32 × s_token × s_weight + bias)',23,400);
// Optional caching policy is separate from the package skip-trace baseline.
rect(1170,1580,890,167,'white',C.purpleEdge,8,'7 5');
text(1615,1616,'Optional DiCache policy',25,700,C.purpleEdge);
text(1615,1654,'Shallow probe → refresh full path OR reuse residual anchors',23,400);
text(1615,1687,'Reuse: DCTA alignment → reconstruct branch velocity',23,400);
text(1615,1724,'Selectable alternative to the supplied block-skip schedule',20,400,C.muted);

text(1070,1832,'Algorithm specification. Full 50-step FPGA execution is an integration target; this figure does not report a verified inference run.',22,400,C.muted);
parts.push('</g></svg>');
const figureDir = path.join(root,'docs','figures');
fs.mkdirSync(figureDir,{recursive:true});
const output = path.join(figureDir,'supra2_img_algorithm_flow.svg');
fs.writeFileSync(output,parts.join('\n'),'utf8');
await sharp(Buffer.from(parts.join('\n'))).resize(W*2,H*2).png().toFile(path.join(figureDir,'supra2_img_algorithm_flow.png'));
console.log(JSON.stringify({svg:output,png:path.join(figureDir,'supra2_img_algorithm_flow.png'),width:W*2,height:H*2}));
