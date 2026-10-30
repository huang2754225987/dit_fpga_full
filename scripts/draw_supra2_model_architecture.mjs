// Vector model diagram: compact block and magnified pre-norm DiT sublayers.
// This generates illustration assets only; it does not execute the model.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const sharp = require('sharp');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const W=1800,H=1120;
const c={ink:'#151b22',blue:'#e5f0ff',blueLine:'#658fc6',green:'#e6f2dd',greenLine:'#739c58',gold:'#fff3cf',goldLine:'#caa33e',lav:'#eee8f7',lavLine:'#9b85ba',rose:'#fce7e4',roseLine:'#cb8279',bus:'#9a83ba'};
const svg=[];
const e=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function txt(x,y,s,size=25,anchor='middle',fill=c.ink,italic=false){svg.push(`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" fill="${fill}" ${italic?'font-family="Georgia, serif" font-style="italic"':''}>${e(s)}</text>`);}
function rect(x,y,w,h,fill,line,rx=6,sw=1.9){svg.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${line}" stroke-width="${sw}"/>`);}
function box(x,y,w,h,label,fill,line,size=25){rect(x,y,w,h,fill,line);txt(x+w/2,y+h/2+size*.33,label,size);}
function wire(points,{color=c.ink,dash='',end=true,width=1.8}={}){svg.push(`<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${dash?`stroke-dasharray="${dash}"`:''} ${end?`marker-end="url(#${color===c.bus?'mod':color===c.greenLine?'ctx':'arr'})"`:''}/>`);}
function op(x,y,type){svg.push(`<circle cx="${x}" cy="${y}" r="17" fill="${type==='+'?c.rose:'#f8fbfc'}" stroke="${type==='+'?c.roseLine:c.ink}" stroke-width="2"/>`);txt(x,y+8,type,28);}
function vector(x,y,n=7){for(let i=0;i<n;i++)rect(x+i*20,y,20,20,'white',c.ink,0,1.8);}
function dot(x,y){svg.push(`<circle cx="${x}" cy="${y}" r="2.6" fill="${c.ink}"/>`);}
svg.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs>${[['arr',c.ink],['mod',c.bus],['ctx',c.greenLine]].map(([id,color])=>`<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="${color}"/></marker>`).join('')}</defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial, Helvetica, sans-serif">`);

// A small abstract block on the left, matching the supplied magnification idiom.
vector(145,815);txt(215,872,'hᵢ',30,'middle',c.ink,true);
wire([[215,815],[215,748]]);
box(60,688,310,60,'Gated Self-Attention',c.gold,c.goldLine,24);
wire([[215,688],[215,638]]);
box(60,578,310,60,'Cross-Attention',c.green,c.greenLine,24);
wire([[215,578],[215,528]]);
box(60,468,310,60,'Gated MLP',c.blue,c.blueLine,24);
wire([[215,468],[215,407]]);
vector(145,387);txt(215,363,'hᵢ₊₁',30,'middle',c.ink,true);
txt(215,924,'DiT block × 14',28);
wire([[395,449],[625,107]],{dash:'7 7',end:false,width:1.9});
wire([[395,765],[625,994]],{dash:'7 7',end:false,width:1.9});

// Main expanded stream. Draw shortcuts first so they remain behind operators.
wire([[1040,995],[730,995],[730,711],[1023,711]]);dot(1040,995);
wire([[1040,688],[705,688],[705,467],[1023,467]]);dot(1040,688);
wire([[1040,435],[680,435],[680,90],[1023,90]]);dot(1040,435);

vector(970,1010);txt(1040,1065,'hᵢ · 256 × 576',29,'middle',c.ink,true);
wire([[1040,1010],[1040,975]]);
box(915,927,250,48,'AdaLN',c.lav,c.lavLine);
wire([[1040,927],[1040,894]]);
rect(805,804,470,90,c.gold,c.goldLine);
txt(1040,834,'Multi-Head Self-Attention',25);
txt(1040,861,'QKV → Attention → Projection',21);
txt(1040,884,'9 heads × 64',20);
wire([[1040,804],[1040,784]]);op(1040,767,'×');
txt(1001,775,'g_sa',22,'end',c.ink,true);
wire([[1040,750],[1040,728]]);op(1040,711,'+');
wire([[1040,694],[1040,668]]);

box(915,620,250,48,'LayerNorm',c.lav,c.lavLine);
wire([[1040,620],[1040,587]]);
rect(805,507,470,80,c.green,c.greenLine);
txt(1040,538,'Text Cross-Attention',25);
txt(1040,568,'Q: image features   K,V: text context',21);
wire([[1040,507],[1040,484]]);op(1040,467,'+');
wire([[1040,450],[1040,419]]);

box(915,371,250,48,'AdaLN',c.lav,c.lavLine);
rect(770,200,540,135,'#f0f9ff',c.blueLine,24,2.2);
txt(797,229,'Dense MLP',24,'start');
box(795,253,150,45,'Linear',c.blue,c.blueLine,23);
box(975,253,125,45,'GELU',c.blue,c.blueLine,23);
box(1130,253,150,45,'Linear',c.blue,c.blueLine,23);
txt(870,324,'576 → 2304',20);txt(1205,324,'2304 → 576',20);
wire([[1040,371],[1040,350],[750,350],[750,275],[795,275]]);
wire([[945,275],[975,275]]);wire([[1100,275],[1130,275]]);
wire([[1205,253],[1205,180],[1040,180],[1040,162]]);
op(1040,145,'×');txt(1001,153,'g_mlp',22,'end',c.ink,true);
wire([[1040,128],[1040,107]]);op(1040,90,'+');
wire([[1040,73],[1040,48]]);vector(970,28);txt(1040,22,'hᵢ₊₁',27,'middle',c.ink,true);

// Conditioning bus: only the two AdaLN sites and their two gates receive t.
rect(1460,105,285,80,c.lav,c.lavLine);
txt(1602,136,'Timestep t',25);
txt(1602,166,'Embedding + AdaLN MLP',20);
wire([[1460,145],[1390,145],[1390,951]],{color:c.bus,dash:'6 6',end:false});
for(const [y,x] of [[145,1057],[395,1165],[767,1057],[951,1165]])
  wire([[1390,y],[x,y]],{color:c.bus,dash:'6 6'});
txt(1440,350,'shift, scale, gate',21,'start',c.bus);

box(1470,517,275,60,'Text context',c.green,c.greenLine,24);
wire([[1470,547],[1275,547]],{color:c.greenLine});
txt(1340,532,'K,V',23,'middle',c.ink,true);
vector(1537,602);txt(1607,659,'projected to 576',22);

txt(1090,1106,'Supra2-IMG DiT block with timestep modulation and text conditioning.',23);
svg.push('</g></svg>');
const source=svg.join('\n');
const figureDir=path.join(root,'docs','figures');
fs.mkdirSync(figureDir,{recursive:true});
const svgFile=path.join(figureDir,'supra2_img_model_architecture_v3.svg');
const pngFile=path.join(figureDir,'supra2_img_model_architecture_v3.png');
fs.writeFileSync(svgFile,source,'utf8');
await sharp(Buffer.from(source)).resize(W*2,H*2).png().toFile(pngFile);
console.log(JSON.stringify({svg:svgFile,png:pngFile,width:W*2,height:H*2}));
